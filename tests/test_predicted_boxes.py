"""画面里那些「没有人形却有框」的框：三道闸门 + 看得懂的标签。

现象是操作员报的：「有时会出现这种没有人形却出现框体的情况」。查下来它不是一个 bug，
而是三处 Default 叠在一起，每一处单独看都合理：

* 检出下限放到 0.1（为了扛遮挡），于是 0.1~0.45 的垃圾检出也会进画面；
* 新建轨迹只看 0.6 —— 栏杆/橱窗模特/广告牌被认成人时经常正好落在 0.6~0.7，
  一旦拿到 ID，这个框就会一直画下去；
* 低功耗模式每 4 帧只检测一次，其余 3 帧画的是卡尔曼预测框（tracked_objects 不带
  置信度），并且预测框同样推进驻留与报警判定。

这里锁住收窄之后的行为：够自信才发 ID、预测帧不新建会话也不报警（除非最近真的
检出过）、以及标签上能一眼分清「模型确定 / 模型在猜 / 根本没检出」。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from alarm_service import EventStore
from detection_engine import DetectionEngine
from inference_profiles import InferencePolicy
from main_window import _detail_rows
from models import AlarmEvent, ZoneDefinition

ZONE = ZoneDefinition(
    name="区域 1",
    polygon=[[100.0, 100.0], [500.0, 100.0], [500.0, 500.0], [100.0, 500.0]],
    closed=True,
    dwell_seconds=1.0,
    cooldown_seconds=0.0,
)
INSIDE = [200.0, 200.0, 260.0, 400.0]  # 框底线中心落在 ZONE 内


def tracked(box: list[float], track_id: int = 1, confidence: float | None = 0.9) -> sv.Detections:
    """一帧的检测结果。``confidence=None`` 模拟预测帧（tracked_objects 不带分数）。"""
    return sv.Detections(
        xyxy=np.asarray([box], dtype=np.float32),
        confidence=None if confidence is None else np.asarray([confidence], dtype=np.float32),
        class_id=None if confidence is None else np.zeros(1, dtype=int),
        tracker_id=np.asarray([track_id]),
    )


def empty() -> sv.Detections:
    return sv.Detections.empty()


def build_engine(
    zones: tuple[ZoneDefinition, ...] = (),
    policy: InferencePolicy | None = None,
) -> DetectionEngine:
    """只用来验判定的引擎：模型与追踪器都是假的（桩法与 test_detection_device 一致）。

    ``from_ultralytics`` 也要一起打桩：真实路径下它是把 YOLO 的 Result 转成 Detections，
    而这里根本不需要经过模型 —— 检出直接由假追踪器给出来。
    """
    with (
        patch("detection_engine.YOLO", return_value=Mock(return_value=[object()])),
        patch.object(DetectionEngine, "_new_tracker", return_value=Mock()),
        patch("detection_engine.sv.Detections.from_ultralytics", return_value=object()),
    ):
        return DetectionEngine("model.pt", list(zones), "cpu", policy)


def low_power_engine() -> DetectionEngine:
    """低功耗预设（每 4 帧才检测一次）的假引擎。"""
    policy = InferencePolicy(model_path="model.pt", device="cpu", detector_interval=4)
    return build_engine((ZONE,), policy)


class PredictedFrameTests(unittest.TestCase):
    """低功耗模式下那 3 帧「没跑推理」的帧：画框可以，但不能当真。"""

    def engine(self) -> DetectionEngine:
        return low_power_engine()

    def test_a_predicted_box_does_not_open_a_session(self) -> None:
        """预测位置不能宣布「有人进入区域」——那是卡尔曼滤波器推出来的，不是看见的。"""
        engine = self.engine()
        tracker = engine.tracker
        tracker.update.return_value = empty()
        tracker.tracked_objects = tracked(INSIDE, track_id=1, confidence=None)

        frame = np.zeros((600, 700, 3), dtype=np.uint8)
        entered = []
        for step in range(4):
            _, transitions = engine.process(frame, float(step) * 0.5, "test.mp4")
            entered.extend(item for item in transitions if item.kind == "entered")

        self.assertEqual(entered, [])
        self.assertEqual(engine.active_sessions, {})

    def test_predictions_cannot_carry_a_session_to_an_alarm(self) -> None:
        """驻留时钟是「距首次进入的时间」，而预测帧同样计入 —— 所以它们不能替误检说话。

        序列（低功耗，每 4 帧一轮检测，这里的 0.1 秒一帧相当于 10fps 的素材）：
        第 1 轮真检出 → 中间 3 帧预测 → 第 2 轮**没检出**（容忍）→ 3 帧预测 →
        第 3 轮**又没检出**（连着两轮看不见）→ 之后的预测帧再也不能报警，
        哪怕按时间算早就过了 1 秒的驻留阈值。
        """
        engine = self.engine()
        tracker = engine.tracker
        tracker.update.return_value = empty()
        tracker.tracked_objects = tracked(INSIDE, track_id=1, confidence=None)

        frame = np.zeros((600, 700, 3), dtype=np.uint8)
        person = tracked(INSIDE, track_id=1, confidence=0.88)
        alarmed = []
        for step in range(13):
            tracker.update.return_value = person if step == 0 else empty()
            _, transitions = engine.process(frame, step * 0.1, "test.mp4")
            alarmed.extend(item for item in transitions if item.kind == "alarmed")

        self.assertEqual(alarmed, [], "检测连着两轮都没看见它，预测框不该撑着报警")
        self.assertEqual(len(engine.active_sessions), 1, "会话仍该留着，等人真的回来")

    def test_one_missed_detection_cycle_is_tolerated(self) -> None:
        """上面那条闸门不能误伤真的在场的人：只漏一轮时，报警照常发。"""
        engine = self.engine()
        tracker = engine.tracker
        tracker.tracked_objects = tracked(INSIDE, track_id=1, confidence=None)

        frame = np.zeros((600, 700, 3), dtype=np.uint8)
        person = tracked(INSIDE, track_id=1, confidence=0.88)
        alarmed = []
        for step in range(6):
            # 第 1 轮（第 1 帧）检出；第 2 轮（第 5 帧）漏掉；其余是预测帧。
            tracker.update.return_value = person if step == 0 else empty()
            _, transitions = engine.process(frame, step * 0.25, "test.mp4")
            alarmed.extend(item for item in transitions if item.kind == "alarmed")

        # 漏一轮之后的第一帧预测（t=1.25）就发报警：驻留早已满 1 秒。
        self.assertEqual([item.event.alarm_at_seconds for item in alarmed], [1.25])

    def test_a_real_detection_still_alarms_after_the_prediction_gap(self) -> None:
        """闸门不能把真报警一起挡住：检测连着漏两轮、人再出现时报照顾旧发出。"""
        engine = self.engine()
        tracker = engine.tracker
        tracker.tracked_objects = tracked(INSIDE, track_id=1, confidence=None)

        frame = np.zeros((600, 700, 3), dtype=np.uint8)
        person = tracked(INSIDE, track_id=1, confidence=0.88)
        again = tracked(INSIDE, track_id=1, confidence=0.83)
        alarmed = []
        for step in range(13):
            # 第 1 轮检出 → 第 2、3 轮都漏（预测框撑着，但不许报警）→ 第 4 轮又检出。
            if step == 0:
                tracker.update.return_value = person
            elif step == 12:
                tracker.update.return_value = again
            else:
                tracker.update.return_value = empty()
            _, transitions = engine.process(frame, step * 0.1, "test.mp4")
            alarmed.extend(item for item in transitions if item.kind == "alarmed")

        self.assertEqual(len(alarmed), 1, "报警应当照旧发出")
        self.assertAlmostEqual(alarmed[0].event.alarm_at_seconds or 0.0, 1.2)
        # 记录下来的置信度来自**真正报警那一轮**的检出，不是之前那次。
        self.assertEqual(alarmed[0].event.alarm_confidence, 0.83)


class TrackerThresholdTests(unittest.TestCase):
    """第三道闸门：什么分数才配拿到一条新轨迹。"""

    def test_new_tracks_require_a_confident_detection(self) -> None:
        tracker = DetectionEngine._new_tracker()

        self.assertEqual(tracker.high_conf_det_threshold, DetectionEngine.NEW_TRACK_CONFIDENCE)
        self.assertEqual(tracker.minimum_iou_threshold, DetectionEngine.TRACK_IOU_THRESHOLD)

    def test_a_weak_repeated_detection_never_becomes_a_track(self) -> None:
        """0.5 的检出连续出现多少次都不该变成带 ID 的框 —— 这正是误检框的来源。

        用真的 ByteTrack（不 mock），因为这里要验的就是这个库的阈值语义：
        新建轨迹只认「没匹配上的高分检出」，而高分线由我们传进去。
        """
        tracker = ByteTrackTracker(
            track_activation_threshold=0.25,
            lost_track_buffer=60,
            frame_rate=30,
            minimum_consecutive_frames=1,
            minimum_iou_threshold=DetectionEngine.TRACK_IOU_THRESHOLD,
            high_conf_det_threshold=DetectionEngine.NEW_TRACK_CONFIDENCE,
        )
        box = np.asarray([[400.0, 300.0, 470.0, 520.0]], dtype=np.float32)
        weak = sv.Detections(
            xyxy=box,
            confidence=np.asarray([0.5], dtype=np.float32),
            class_id=np.zeros(1, dtype=int),
        )
        strong = sv.Detections(
            xyxy=box,
            confidence=np.asarray([0.85], dtype=np.float32),
            class_id=np.zeros(1, dtype=int),
        )

        ids = []
        for step in range(6):
            ids.append(tracker.update(weak, timestamp=step * 0.1).tracker_id.tolist())
        self.assertEqual(ids, [[-1]] * 6, "弱检出不该拿到 ID")
        self.assertEqual(len(tracker.tracked_objects), 0)

        # 同一个位置来一次够自信的检出，就得给它 ID（不能把真人也一起挡掉）。
        tracker.update(strong, timestamp=0.6)
        final = tracker.update(strong, timestamp=0.7).tracker_id.tolist()
        self.assertGreaterEqual(final[0], 0)


class BoxLabelTests(unittest.TestCase):
    """调试视图下的标签：打开「显示目标编号与置信度」之后，编号与分数一个都不能少。

    排查「为什么老误报」靠的就是它们 —— 编号能和报警记录对上号，分数说明模型当时
    有多确定。默认那版（只写「行人」「区域内 1.5秒」「闯入 2.0秒」）在
    tests/test_label_overlay.py 里钉着，这里只管开开关之后的那一版。
    """

    def test_label_shows_confidence(self) -> None:
        label = partial(DetectionEngine._box_label, show_details=True)

        self.assertEqual(label(9, 0.86, None), "目标9 0.86")
        self.assertEqual(label(9, 0.14, None), "目标9 0.14")
        self.assertEqual(label(9, 0.14, 1.5), "目标9 0.14 区内1.5秒")
        self.assertEqual(label(None, None, None), "人物")

    def test_predicted_frames_still_show_the_last_real_confidence(self) -> None:
        """低功耗模式下 3/4 的帧是预测帧，那些帧也得有分数。

        否则屏幕上大部分时间是一堆没有数字的框 —— 现场看到的效果就是"全都不识别了"
        （这是真事：把预测帧画成不带分数的虚线样式之后，用户就是这么反馈的）。
        """
        engine = low_power_engine()
        tracker = engine.tracker
        tracker.update.return_value = tracked(INSIDE, track_id=1, confidence=0.86)
        tracker.tracked_objects = tracked(INSIDE, track_id=1, confidence=None)
        frame = np.zeros((600, 700, 3), dtype=np.uint8)

        with patch.object(
            DetectionEngine, "_annotate", side_effect=lambda *a, **k: a[0]
        ) as annotate:
            engine.process(frame, 0.0, "test.mp4")  # 检出帧：分数来自这一帧的检出
            engine.process(frame, 0.1, "test.mp4")  # 预测帧：分数要沿用上一次
            engine.process(frame, 0.2, "test.mp4")

        detection_confidences = annotate.call_args_list[0].kwargs["confidences"]
        prediction_confidences = annotate.call_args_list[1].kwargs["confidences"]
        self.assertEqual(detection_confidences, [0.86])
        self.assertEqual(
            prediction_confidences,
            [0.86],
            "预测帧也要有分数，否则界面上会是一堆没有数字的框",
        )

    def test_boxes_follow_the_target_between_detections(self) -> None:
        """两次真实观测之间，预测框按估计速度插值 —— 这就是「框跟着人走」。

        低功耗模式下每 4 帧只有 1 帧真跑推理，如果中间几帧的框冻住，人走了框还留在
        原地，看上去就是「识别跟不上」。追踪器的滤波器必须（在跳过的帧上）被推进。
        """
        engine = low_power_engine()
        tracker = engine.tracker
        tracker.update.return_value = tracked([100.0, 100.0, 200.0, 300.0], track_id=1)
        frame = np.zeros((600, 700, 3), dtype=np.uint8)

        with patch.object(
            DetectionEngine, "_annotate", side_effect=lambda *a, **k: a[0]
        ) as annotate:
            engine.process(frame, 0.0, "test.mp4")  # 检测帧：真实观测
            tracker.tracked_objects = tracked([110.0, 100.0, 210.0, 300.0], track_id=1)
            engine.process(frame, 0.1, "test.mp4")  # 跳过帧：预测位置
            drawn = annotate.call_args_list[-1].args[1].xyxy

        # 刚被看见过 → 采用预测位置（不是钉在 100 上）。
        self.assertEqual([int(value) for value in drawn[0]], [110, 100, 210, 300])

    def test_a_lost_track_stops_moving_with_the_prediction(self) -> None:
        """跟丢之后不能再让预测框继续往前走 —— 那会变成「空地上有个框自己在走」。

        卡尔曼在没有任何观测时会按最后的速度一直外推（实测 225px/秒、最多滑出 375px），
        所以只对「最近一轮检测还匹配上」的轨迹采用预测位置，其余钉在最后一次真实观测处。
        """
        engine = low_power_engine()
        tracker = engine.tracker
        seen = [100.0, 100.0, 200.0, 300.0]
        drifted = [400.0, 100.0, 500.0, 300.0]
        tracker.update.return_value = tracked(seen, track_id=1)
        tracker.tracked_objects = tracked(seen, track_id=1)
        frame = np.zeros((600, 700, 3), dtype=np.uint8)

        with patch.object(
            DetectionEngine, "_annotate", side_effect=lambda *a, **k: a[0]
        ) as annotate:
            engine.process(frame, 0.0, "test.mp4")  # 第 1 轮检测：最后一次被看见
            engine.process(frame, 0.1, "test.mp4")  # 跳过帧
            tracker.update.return_value = empty()  # 这一轮检测没匹配上
            engine.process(frame, 0.2, "test.mp4")  # 跳过帧
            engine.process(frame, 0.3, "test.mp4")  # 跳过帧
            engine.process(frame, 0.4, "test.mp4")  # 第 2 轮检测：没匹配上 → 算是跟丢
            tracker.tracked_objects = tracked(drifted, track_id=1)
            engine.process(frame, 0.5, "test.mp4")  # 跳过帧：预测框已经飘到 400
            drawn = annotate.call_args_list[-1].args[1].xyxy

        self.assertEqual(
            [int(value) for value in drawn[0]],
            [int(value) for value in seen],
            "跟丢之后应当钉在最后一次被看见的位置，而不是跟着外推走",
        )

    def test_weaker_boxes_are_drawn_more_lightly(self) -> None:
        """弱检出画细线、暗色：分数低的框不该和确定的框一样抢眼。"""
        rng = np.random.default_rng(20260921)
        frame = rng.integers(40, 200, size=(600, 700, 3), dtype=np.uint8)
        box = [220.0, 220.0, 320.0, 420.0]
        engine = build_engine()

        def painted(confidence: float) -> int:
            rendered = engine._annotate(
                frame, tracked(box, confidence=confidence), {}, {}
            )
            return int((rendered != frame).any(axis=2).sum())

        strong = painted(0.9)
        weak = painted(0.2)

        self.assertGreater(strong, weak, "弱检出应当画得比确定的检出轻")
        self.assertGreater(weak, 0, "弱框仍要看得见 —— 它可能是真的")


class AlarmConfidenceRecordTests(unittest.TestCase):
    """报警记录带上那一刻的置信度：夜里一条误报，事后要能看出模型当时有多确定。"""

    def test_confidence_round_trips_through_the_event_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory))
            event = AlarmEvent(
                source="test.mp4",
                zone_name="区域 1",
                track_id="1",
                entered_at_seconds=1.0,
                alarm_at_seconds=2.0,
                wall_time="2026-09-21 10:00:00",
                operation_mode="video",
            )
            event.alarm_confidence = 0.86

            store.record(event, np.zeros((16, 16, 3), dtype=np.uint8))
            restored = store.load_recent()

            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0].alarm_confidence, 0.86)

    def test_a_hand_edited_confidence_is_read_as_not_recorded(self) -> None:
        """记录是纯文本。手改坏一个字段不该让整块报警记录读不出来。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "alarm_events.jsonl"
            log.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "action": "opened",
                        "event": {
                            "session_id": "s1",
                            "time": "2026-09-21 10:00:00",
                            "alarm_at_seconds": 2.0,
                            "entered_at_seconds": 1.0,
                            "alarm_confidence": "很高",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            events = EventStore(root).load_recent()

            self.assertEqual(len(events), 1)
            self.assertIsNone(events[0].alarm_confidence)

    def test_the_detail_dialog_shows_the_confidence_or_says_it_was_not_recorded(self) -> None:
        def row_value(event: AlarmEvent) -> str:
            rows = dict(_detail_rows(event))
            return rows["报警置信度"]

        recorded = AlarmEvent(
            source="test.mp4",
            zone_name="区域 1",
            track_id="1",
            entered_at_seconds=1.0,
            alarm_at_seconds=2.0,
            wall_time="2026-09-21 10:00:00",
        )
        recorded.alarm_confidence = 0.62

        self.assertEqual(row_value(recorded), "0.62")
        # 加字段之前写下的记录：如实说没记，不推算、也不拿它当「很确定」。
        self.assertIn("未记录", row_value(AlarmEvent(
            source="test.mp4",
            zone_name="区域 1",
            track_id="1",
            entered_at_seconds=1.0,
            alarm_at_seconds=2.0,
            wall_time="2026-09-21 10:00:00",
        )))

    def test_predicted_boxes_are_not_drawn_when_the_frame_is_not_rendered(self) -> None:
        """render=False（自检、基准测试走这条路）时不该白花画框的时间。"""
        engine = build_engine()
        engine.tracker.update.return_value = empty()
        frame = np.zeros((600, 700, 3), dtype=np.uint8)

        rendered, _ = engine.process(frame, 0.0, "test.mp4", render=False)

        self.assertIs(rendered, frame)


if __name__ == "__main__":
    unittest.main()
