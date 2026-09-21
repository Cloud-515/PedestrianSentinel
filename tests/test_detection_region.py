"""只检测警戒区：裁片范围、坐标回映射、以及几时退回整帧。

这一档的意义是「省 CPU 又不牺牲区域内识别率」：网络输入尺寸固定，把同样的输入像素
全花在警戒区上，区域内的有效分辨率反而更高（实测区域内的人置信度 0.78，整帧只有
0.67，标准模式 FP32@640 是 0.71）。代价是区外的框不再产生，所以范围算错一点都不行：

* 裁得太紧 → 区域内的人被切掉（这套测试用坐标回映射 + 区域判定把它们钉住）；
* 回映射漏加偏移 → 框整体偏移一个裁片原点，**报警会直接失效**（锚点落不进多边形），
  所以下面有一条端到端的「进入 → 驻留 → 报警」检查。
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import supervision as sv

from detection_engine import DetectionEngine
from inference_profiles import InferencePolicy
from models import ZoneDefinition

FRAME_SHAPE = (1080, 1920, 3)
# 警戒区矩形 (600,500)-(1200,900)：宽 600、高 400；外扩 30% = 各边 180 / 120
ZONE_BOX = (600.0, 500.0, 1200.0, 900.0)
EXPECTED_CROP = (420, 380, 1380, 1020)
PERSON_IN_ZONE = (700.0, 560.0, 760.0, 840.0)  # 底线中点在区域内
PERSON_OUTSIDE = (60.0, 560.0, 120.0, 840.0)  # 画面左边，区域外


class FakeResult:
    """冒充 ultralytics 的 Result：只带一组框（坐标就是喂给模型的那张图的坐标）。"""

    def __init__(self, boxes: list[tuple[float, float, float, float, float]]) -> None:
        self.boxes = boxes


class FakeModel:
    def __init__(self, boxes=()) -> None:
        self.boxes = list(boxes)
        self.frames: list[np.ndarray] = []

    def __call__(self, frame, **kwargs):
        self.frames.append(frame)
        return [FakeResult(self.boxes)]


def from_fake_result(result: FakeResult) -> sv.Detections:
    xyxy = np.asarray([box[:4] for box in result.boxes], dtype=np.float32).reshape(-1, 4)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.asarray([box[4] for box in result.boxes], dtype=np.float32),
        class_id=np.zeros(len(result.boxes), dtype=int),
    )


def zone(name: str = "区域 1", box: tuple[float, ...] = ZONE_BOX) -> ZoneDefinition:
    x1, y1, x2, y2 = box
    return ZoneDefinition(
        name=name,
        polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        closed=True,
        dwell_seconds=1.0,
        cooldown_seconds=0.0,
    )


def build(
    policy: InferencePolicy, zones: list[ZoneDefinition], boxes=()
) -> tuple[DetectionEngine, FakeModel]:
    """建一个假模型的引擎。假模型与 from_ultralytics 的补丁由调用方维持（见用例内的 build）。"""
    model = FakeModel(boxes)
    with (
        patch("detection_engine.YOLO", return_value=model),
        patch(
            "detection_engine.sv.Detections.from_ultralytics",
            side_effect=from_fake_result,
        ),
    ):
        engine = DetectionEngine("model.pt", zones, "cpu", policy)
    return engine, model


REGION_POLICY = InferencePolicy(
    model_path="model.pt", device="cpu", imgsz=512, detector_interval=1, detect_region_margin=0.3
)
FULL_FRAME_POLICY = InferencePolicy(model_path="model.pt", device="cpu")


class DetectionRegionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)

    def build(
        self, policy: InferencePolicy, zones: list[ZoneDefinition], boxes=()
    ) -> tuple[DetectionEngine, FakeModel]:
        """假模型的补丁要**在整个用例期间**有效：检测发生在 process() 里，而那在
        构造之后才调用。"""
        model = FakeModel(boxes)
        for patcher in (
            patch("detection_engine.YOLO", return_value=model),
            patch(
                "detection_engine.sv.Detections.from_ultralytics",
                side_effect=from_fake_result,
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        return DetectionEngine("model.pt", zones, "cpu", policy), model

    def test_only_the_region_around_the_zones_is_sent_to_the_model(self) -> None:
        engine, model = self.build(REGION_POLICY, [zone()])

        engine.process(self.frame, 0.0, "test.mp4")

        self.assertEqual(len(model.frames), 1)
        self.assertEqual(
            model.frames[0].shape[:2],
            (EXPECTED_CROP[3] - EXPECTED_CROP[1], EXPECTED_CROP[2] - EXPECTED_CROP[0]),
        )

    def test_boxes_come_back_in_frame_coordinates(self) -> None:
        """模型在裁片上给的框（原点在裁片左上角）必须换算回整帧坐标。"""
        crop_origin_x, crop_origin_y = EXPECTED_CROP[0], EXPECTED_CROP[1]
        in_crop = (
            PERSON_IN_ZONE[0] - crop_origin_x,
            PERSON_IN_ZONE[1] - crop_origin_y,
            PERSON_IN_ZONE[2] - crop_origin_x,
            PERSON_IN_ZONE[3] - crop_origin_y,
            0.9,
        )
        engine, _ = self.build(REGION_POLICY, [zone()], [in_crop])
        drawn: list[sv.Detections] = []

        with patch.object(
            DetectionEngine, "_annotate", side_effect=lambda *a, **k: drawn.append(a[1]) or a[0]
        ):
            engine.process(self.frame, 0.0, "test.mp4")

        box = [float(value) for value in drawn[0].xyxy[0]]
        self.assertEqual(box, list(PERSON_IN_ZONE))

    def test_zone_alarm_still_fires_with_region_detection(self) -> None:
        """端到端：坐标回映射若漏了偏移，锚点就落不进多边形，这条会失败。"""
        crop_origin_x, crop_origin_y = EXPECTED_CROP[0], EXPECTED_CROP[1]
        in_crop = (
            PERSON_IN_ZONE[0] - crop_origin_x,
            PERSON_IN_ZONE[1] - crop_origin_y,
            PERSON_IN_ZONE[2] - crop_origin_x,
            PERSON_IN_ZONE[3] - crop_origin_y,
            0.9,
        )
        engine, _ = self.build(REGION_POLICY, [zone()], [in_crop])

        events = []
        # 第 1 帧建轨迹、第 2 帧拿到 ID（会话从这里开始计时）、第 3 帧满 0.5 秒、第 4 帧满 1 秒
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            _, transitions = engine.process(self.frame, timestamp, "test.mp4")
            events.extend(transitions)

        self.assertEqual([item.kind for item in events], ["entered", "alarmed"])
        self.assertEqual(events[-1].event.alarm_confidence, 0.9)

    def test_the_recorded_event_keeps_the_video_source_and_stays_serializable(self) -> None:
        """报警记录必须是能落盘的 —— 裁片那条路径曾经把 `source` 参数遮住了。

        现场表现：一条报警就把检测线程打停（``json.dumps`` 撞上 1080p 图像数组），
        界面上同时弹出 ``QTableWidgetItem.setText(ndarray)``。所以这里把「事件里的
        source 仍是视频源标识、整条记录能 json.dumps」钉成契约。
        """
        import json

        from alarm_service import EventStore

        crop_origin_x, crop_origin_y = EXPECTED_CROP[0], EXPECTED_CROP[1]
        in_crop = (
            PERSON_IN_ZONE[0] - crop_origin_x,
            PERSON_IN_ZONE[1] - crop_origin_y,
            PERSON_IN_ZONE[2] - crop_origin_x,
            PERSON_IN_ZONE[3] - crop_origin_y,
            0.9,
        )
        engine, _ = self.build(REGION_POLICY, [zone()], [in_crop])

        events = []
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            _, transitions = engine.process(self.frame, timestamp, "rtsp://camera/live", "monitor")
            events.extend(transitions)

        self.assertTrue(events, "至少该有一条进入记录")
        for item in events:
            self.assertEqual(item.event.source, "rtsp://camera/live")
            # 真记录写盘时走的就是这个 payload。
            json.dumps(EventStore._event_payload(item.event), ensure_ascii=False)

    def test_a_person_outside_the_region_is_never_detected(self) -> None:
        """区外的框不再产生 —— 这是这一档的取舍，测试把它说清楚。"""
        engine, _ = self.build(REGION_POLICY, [zone()])  # 模型在裁片里什么也没看见
        drawn: list[sv.Detections] = []

        with patch.object(
            DetectionEngine, "_annotate", side_effect=lambda *a, **k: drawn.append(a[1]) or a[0]
        ):
            engine.process(self.frame, 0.0, "test.mp4")

        self.assertEqual(len(drawn[0]), 0)
        crop = engine._detect_crop(self.frame)
        self.assertIsNotNone(crop)
        # 区外那一位（x 60~120）确实落在检测范围之外 —— 这一档不再产生他的框。
        self.assertGreater(crop[0], PERSON_OUTSIDE[2])

    def test_without_a_closed_zone_the_whole_frame_is_used(self) -> None:
        """用户正在画区域（还没闭合）时不能什么都不检测，否则画面上一片空白。"""
        drawing = ZoneDefinition(name="编辑中", polygon=[[10.0, 10.0], [20.0, 20.0]])
        engine, model = self.build(REGION_POLICY, [drawing])

        engine.process(self.frame, 0.0, "test.mp4")

        self.assertEqual(model.frames[0].shape[:2], FRAME_SHAPE[:2])

    def test_a_region_covering_almost_everything_falls_back_to_the_full_frame(self) -> None:
        engine, model = self.build(REGION_POLICY, [zone(box=(10.0, 10.0, 1900.0, 1070.0))])

        engine.process(self.frame, 0.0, "test.mp4")

        self.assertEqual(model.frames[0].shape[:2], FRAME_SHAPE[:2])

    def test_the_region_follows_zone_changes(self) -> None:
        engine, model = self.build(REGION_POLICY, [zone()])
        engine.process(self.frame, 0.0, "test.mp4")
        self.assertEqual(model.frames[0].shape[:2], (640, 960))

        engine.update_zones([zone(box=(1400.0, 100.0, 1800.0, 400.0))])
        engine.process(self.frame, 0.1, "test.mp4")

        # 新区域在右下角（400×300，外扩 30% = 各边 120/90，右边顶到画面边缘）：
        # 裁片跟着搬家。
        self.assertEqual(model.frames[1].shape[:2], (480, 640))
        self.assertEqual(engine._detect_crop(self.frame), (1280, 10, 1920, 490))

    def test_standard_mode_never_crops(self) -> None:
        """标准模式（没有设定外扩）永远送整帧 —— 区外的框照旧要画。"""
        engine, model = self.build(FULL_FRAME_POLICY, [zone()])

        engine.process(self.frame, 0.0, "test.mp4")

        self.assertEqual(model.frames[0].shape[:2], FRAME_SHAPE[:2])


if __name__ == "__main__":
    unittest.main()
