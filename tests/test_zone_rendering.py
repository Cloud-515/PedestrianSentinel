"""警戒区的渲染：贴地效果与被行人遮挡。

渲染没法用断言描述"好不好看"，但它的**契约**可以：

* 行人占据的像素必须和原帧一模一样 —— 这正是"边界看起来在人后面"的实现方式，
  也是它唯一容易被后续改动破坏的地方（多画一层、换个顺序，效果就没了）；
* 没人靠近区域时，不该动任何额外像素；
* 未闭合的区域不能当成面来填。

这几条钉住了，剩下的浓淡宽窄都是参数，改坏了看得出来。
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import supervision as sv

from detection_engine import DetectionEngine
from models import ZoneDefinition


def build_engine(zones: list[ZoneDefinition]) -> DetectionEngine:
    """只用来测绘制的引擎：模型与追踪器都是假的。"""
    with (
        patch("detection_engine.YOLO", return_value=Mock()),
        patch.object(DetectionEngine, "_new_tracker", return_value=Mock()),
    ):
        return DetectionEngine("model.pt", zones, "cpu")


def detections(*boxes: tuple[int, int, int, int]) -> sv.Detections:
    return sv.Detections(
        # reshape(-1, 4)：一个框都不给时也要是 (0, 4)，supervision 不接受 (0,)。
        xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        confidence=np.full(len(boxes), 0.9, dtype=np.float32),
        class_id=np.zeros(len(boxes), dtype=int),
        tracker_id=np.arange(len(boxes)),
    )


class ZoneRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        # 有纹理的底图：纯色底图上看不出"贴回原像素"和"被区域盖住"的区别。
        rng = np.random.default_rng(20260921)
        self.frame = rng.integers(40, 200, size=(400, 600, 3), dtype=np.uint8)
        self.zone = ZoneDefinition(
            name="北侧通道",
            polygon=[[100.0, 120.0], [500.0, 120.0], [500.0, 360.0], [100.0, 360.0]],
            closed=True,
        )

    def render(
        self,
        zones: list[ZoneDefinition],
        boxes: list[tuple[int, int, int, int]],
        in_zone: dict[str, set[int]] | None = None,
    ) -> np.ndarray:
        engine = build_engine(zones)
        ids = in_zone if in_zone is not None else {zone.name: set() for zone in zones}
        elapsed = {index: 1.0 for index in range(len(boxes))}
        return engine._annotate(self.frame, detections(*boxes), ids, elapsed)

    def test_zone_is_painted_into_the_frame(self) -> None:
        rendered = self.render([self.zone], [])

        inside = rendered[150:330, 150:470]
        self.assertFalse(
            np.array_equal(inside, self.frame[150:330, 150:470]),
            "区域内部应当被画上（底色或边带）",
        )
        # 区域外不该被动过。
        np.testing.assert_array_equal(rendered[0:100, 0:100], self.frame[0:100, 0:100])

    def test_person_pixels_are_restored_over_the_zone(self) -> None:
        """被行人占据的像素与原帧完全一致 —— 这就是"被遮挡"的实现契约。

        比对的椭圆要往里收一点：检测框的线画在框的边缘上，而近似轮廓是内接的，
        两者在边缘处会重叠，那一圈本来就会被框线覆盖。
        """
        box = (220, 140, 320, 340)
        rendered = self.render([self.zone], [box])

        mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, (270, 240), (38, 88), 0, 0, 360, 255, -1)
        inside_person = mask > 0
        self.assertTrue(inside_person.sum() > 1000)
        np.testing.assert_array_equal(
            rendered[inside_person], self.frame[inside_person]
        )

    def test_zone_still_shows_beside_the_person(self) -> None:
        """遮挡是"按轮廓断开"，不是把整块区域擦掉。"""
        box = (220, 140, 320, 340)
        rendered = self.render([self.zone], [box])

        # 人框左右两侧、同一水平高度上的区域像素仍应被画过。
        for column in (200, 340):
            column_slice = slice(150, 330)
            self.assertFalse(
                np.array_equal(
                    rendered[column_slice, column], self.frame[column_slice, column]
                ),
                f"x={column} 处区域应当仍然可见",
            )

    def test_person_far_from_the_zone_only_changes_his_own_box(self) -> None:
        """没人靠近区域时不做遮挡 —— 那一步的代价只在真需要时才付。

        于是"远处的人"对这一帧的唯一影响就是他自己的检测框，区域部分逐像素不变。
        """
        without = self.render([self.zone], [])
        box = (20, 20, 80, 90)
        with_far_person = self.render([self.zone], [box])

        difference = np.any(without != with_far_person, axis=2)
        rows, columns = np.nonzero(difference)
        self.assertTrue(len(rows) > 0, "远处那个人的框应当被画出来")
        # 差异只允许出现在他的框附近（框线有 2px 宽，标签在框上方）。
        self.assertGreaterEqual(columns.min(), box[0] - 2)
        self.assertLessEqual(columns.max(), box[2] + 2)
        self.assertLessEqual(rows.min(), box[3] + 2)

    def test_alert_state_paints_the_zone_more_strongly(self) -> None:
        quiet = self.render([self.zone], [])
        alert = self.render([self.zone], [], in_zone={self.zone.name: {0}})

        quiet_band = quiet[118:126, 150:470].astype(int).sum()
        alert_band = alert[118:126, 150:470].astype(int).sum()
        self.assertGreater(alert_band, quiet_band)

    def test_unclosed_zone_is_a_line_not_a_filled_area(self) -> None:
        open_zone = ZoneDefinition(
            name="编辑中",
            polygon=[[100.0, 120.0], [500.0, 120.0], [500.0, 360.0]],
            closed=False,
        )
        rendered = self.render([open_zone], [])

        # 折线经过的地方变了，但三角面内部不该被填充。
        self.assertFalse(np.array_equal(rendered[118:126, 150:450], self.frame[118:126, 150:450]))
        np.testing.assert_array_equal(rendered[200:260, 200:300], self.frame[200:260, 200:300])

    def test_zone_partly_outside_the_frame_does_not_crash(self) -> None:
        """顶点可以落在画面外（换过不同分辨率的视频源就会这样）。"""
        outside = ZoneDefinition(
            name="越界",
            polygon=[[-50.0, 200.0], [300.0, 380.0], [700.0, 900.0], [-200.0, 500.0]],
            closed=True,
        )

        rendered = self.render([outside], [(200, 200, 300, 380)])

        self.assertEqual(rendered.shape, self.frame.shape)

    def test_zone_name_is_drawn_even_when_its_vertex_is_off_frame(self) -> None:
        outside = ZoneDefinition(
            name="越界",
            polygon=[[-400.0, -300.0], [300.0, 200.0], [500.0, 380.0]],
            closed=True,
        )
        rendered = self.render([outside], [])

        # 左上角（名字被夹回来的地方）应当有画过的痕迹。
        self.assertFalse(np.array_equal(rendered[0:30, 0:120], self.frame[0:30, 0:120]))

    def test_no_zones_means_only_boxes_are_drawn(self) -> None:
        rendered = self.render([], [(100, 100, 200, 300)])

        # 框边上的像素变了，其余区域没变。
        self.assertFalse(np.array_equal(rendered[100, 100:200], self.frame[100, 100:200]))
        np.testing.assert_array_equal(rendered[300:400, 300:400], self.frame[300:400, 300:400])

    def test_band_is_wider_near_the_bottom_of_the_frame(self) -> None:
        """透视：同一个区域的上下两条边，靠下（离相机近）的那条更宽。"""
        engine = build_engine([self.zone])

        top_width = engine._band_width(120.0)
        bottom_width = engine._band_width(360.0)

        self.assertLess(top_width, bottom_width)
        self.assertGreaterEqual(top_width, engine.BAND_MIN_WIDTH)
        self.assertLessEqual(engine._band_width(100_000.0), engine.BAND_MAX_WIDTH)


if __name__ == "__main__":
    unittest.main()
