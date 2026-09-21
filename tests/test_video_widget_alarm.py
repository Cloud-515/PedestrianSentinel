from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication
from unittest.mock import patch

from models import AlarmEvent, ZoneDefinition
from video_widget import VideoWidget


class VideoWidgetAlarmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.widget = VideoWidget()

    def _event(
        self,
        zone_name: str,
        track_id: str,
        wall_time: str = "2026-08-12 14:32:08",
    ) -> AlarmEvent:
        return AlarmEvent(
            source="0",
            zone_name=zone_name,
            track_id=track_id,
            entered_at_seconds=1.0,
            alarm_at_seconds=2.0,
            wall_time=wall_time,
            operation_mode="monitor",
        )

    def test_flushes_unique_events_in_first_seen_order(self) -> None:
        self.widget.show_alarm(self._event("北侧入口", "12"))
        self.widget.show_alarm(self._event("南侧通道", "7"))

        self.widget._flush_alarm_batch()

        self.assertEqual(
            [(event.zone_name, event.track_id) for event in self.widget._displayed_alarm_events],
            [("北侧入口", "12"), ("南侧通道", "7")],
        )
        self.assertFalse(self.widget._pending_alarm_events)

    def test_deduplicates_same_zone_and_target_with_latest_event(self) -> None:
        self.widget.show_alarm(self._event("北侧入口", "12", "2026-08-12 14:32:08"))
        self.widget.show_alarm(self._event("北侧入口", "12", "2026-08-12 14:32:09"))

        self.widget._flush_alarm_batch()

        self.assertEqual(len(self.widget._displayed_alarm_events), 1)
        self.assertEqual(
            self.widget._displayed_alarm_events[0].wall_time,
            "2026-08-12 14:32:09",
        )

    def test_retains_full_batch_when_more_than_four_events_arrive(self) -> None:
        for index in range(5):
            self.widget.show_alarm(self._event(f"区域 {index}", str(index)))

        self.widget._flush_alarm_batch()

        self.assertEqual(len(self.widget._displayed_alarm_events), 5)
        self.assertEqual(len(self.widget._displayed_alarm_events[:4]), 4)
        self.assertEqual(len(self.widget._displayed_alarm_events) - 4, 1)

    def test_only_first_event_starts_aggregation_window(self) -> None:
        self.widget.show_alarm(self._event("北侧入口", "12"))
        first_timer_id = self.widget._aggregation_timer.timerId()
        self.widget.show_alarm(self._event("南侧通道", "7"))

        self.assertTrue(self.widget._aggregation_timer.isActive())
        self.assertEqual(self.widget._aggregation_timer.timerId(), first_timer_id)

    def test_clear_alarm_stops_timers_and_clears_batches(self) -> None:
        self.widget.show_alarm(self._event("北侧入口", "12"))
        self.widget._flush_alarm_batch()

        self.widget.clear_alarm()

        self.assertFalse(self.widget._aggregation_timer.isActive())
        self.assertFalse(self.widget._alarm_timer.isActive())
        self.assertFalse(self.widget._pending_alarm_events)
        self.assertFalse(self.widget._displayed_alarm_events)


class ZoneOverlayTests(unittest.TestCase):
    """控件只画**编辑用的那一层**：激活区域的虚线轮廓与顶点手柄。

    区域本身（贴地效果、被行人遮挡）由引擎烧进画面里 —— 它必须画在人下面才能被遮挡，
    而控件只能画在图像之上。控件要是再把实线画一遍，就会出现两条错开的线（这正是
    原来"看着又粗又糊"的原因）。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.widget = VideoWidget()
        self.widget.resize(200, 150)
        self.widget.set_frame(np.full((150, 200, 3), 90, dtype=np.uint8))
        # _video_rect 平时是在 paintEvent 里算的；这里没走绘制流程，得自己算一次，
        # 否则它还是 (0,0,0,0)，所有顶点都会被换算到控件左上角去。
        self.widget._video_rect = self.widget._calculate_video_rect()

    @staticmethod
    def _zone(name: str) -> ZoneDefinition:
        return ZoneDefinition(
            name=name,
            polygon=[[40.0, 40.0], [160.0, 40.0], [160.0, 120.0], [40.0, 120.0]],
            closed=True,
        )

    def _draw(self) -> np.ndarray:
        """把控件层画到一张干净图上，返回像素（画布与控件同尺寸）。"""
        size = self.widget.size()
        image = QImage(size, QImage.Format.Format_RGB888)
        image.fill(0)
        painter = QPainter(image)
        self.widget._draw_zones(painter)
        painter.end()
        buffer = image.constBits()
        array = np.frombuffer(buffer, dtype=np.uint8).reshape(
            (size.height(), image.bytesPerLine())
        )
        return array[:, : size.width() * 3].reshape(
            (size.height(), size.width(), 3)
        ).copy()

    def test_draws_nothing_when_no_zone_is_active(self) -> None:
        self.widget.set_zones([self._zone("区域1")])
        self.widget.set_active_zone(None)

        self.assertEqual(self._draw().sum(), 0)

    def test_draws_the_active_zone_while_hovering(self) -> None:
        zone = self._zone("区域1")
        self.widget.set_zones([zone])
        self.widget.set_active_zone(zone)

        with patch.object(self.widget, "underMouse", return_value=True):
            painted = self._draw()

        self.assertGreater(painted.sum(), 0)
        # 区域中心是空的：这里只画轮廓，填充归引擎（它要画在人下面）。
        center = self.widget._to_display(QPointF(100.0, 80.0))
        self.assertEqual(painted[int(center.y()), int(center.x())].sum(), 0)

    def test_no_editing_overlay_while_monitoring(self) -> None:
        """鼠标不在画面上时不该有编辑痕迹 —— 虚线只能画在图像之上，会横穿在人身上。"""
        zone = self._zone("区域1")
        self.widget.set_zones([zone])
        self.widget.set_active_zone(zone)

        with patch.object(self.widget, "underMouse", return_value=False):
            painted = self._draw()

        self.assertEqual(painted.sum(), 0)

    def test_unclosed_zone_is_always_shown(self) -> None:
        """正在画的区域必须看得见，否则不知道下一个点会连到哪。"""
        zone = ZoneDefinition(
            name="编辑中",
            polygon=[[40.0, 40.0], [160.0, 40.0]],
            closed=False,
        )
        self.widget.set_zones([zone])
        self.widget.set_active_zone(zone)

        with patch.object(self.widget, "underMouse", return_value=False):
            painted = self._draw()

        self.assertGreater(painted.sum(), 0)

    def test_dragging_keeps_the_overlay_visible(self) -> None:
        zone = self._zone("区域1")
        self.widget.set_zones([zone])
        self.widget.set_active_zone(zone)
        self.widget._drag_index = 0

        with patch.object(self.widget, "underMouse", return_value=False):
            painted = self._draw()

        self.assertGreater(painted.sum(), 0)

    def test_outline_follows_true_positions_not_clamped_handles(self) -> None:
        """轮廓必须按顶点的真实位置画。

        顶点被外推到控件之外时，标记会被夹到边缘（为了还能抓住它），但**轮廓不能跟着
        夹** —— 引擎烧进帧里的贴地带用的是真实坐标，轮廓跟着夹就会出现两条对不上的
        边界，而错位的那条还横穿在人身上。
        """
        zone = ZoneDefinition(
            name="区域1",
            polygon=[[20.0, 20.0], [60.0, 20.0], [150.0, 200.0]],
            closed=False,
        )
        self.widget.set_zones([zone])
        self.widget.set_active_zone(zone)
        painted = self._draw()

        start = self.widget._to_display(QPointF(60.0, 20.0))
        end = self.widget._to_display(QPointF(150.0, 200.0))
        marker, clamped = self.widget._handle_position(QPointF(150.0, 200.0))
        self.assertTrue(clamped, "这个顶点应当落在控件之外（否则这条用例没意义）")
        self.assertNotEqual(marker.y(), end.y())

        def point_on(line_start: QPointF, line_end: QPointF, y: float) -> tuple[int, int]:
            ratio = (y - line_start.y()) / (line_end.y() - line_start.y())
            return (
                int(line_start.x() + (line_end.x() - line_start.x()) * ratio),
                int(y),
            )

        def has_white_near(x: int, y: int, radius: int = 5) -> bool:
            window = painted[
                max(0, y - radius) : y + radius, max(0, x - radius) : x + radius
            ]
            return bool(window.sum())

        # 虚线本来就有间隙，所以按命中比例判断：真实边上大多数采样点应当有像素，
        # 而被夹住的那条线上应当一个都没有（那里什么都没画）。
        samples = [(x, y) for y in range(150, 381, 20) for x, _ in [point_on(start, end, y)]]
        true_hits = sum(1 for x, y in samples if has_white_near(x, y))
        clamped_hits = sum(
            1 for _, y in samples for x, _ in [point_on(start, marker, y)] if has_white_near(x, y)
        )
        self.assertGreaterEqual(
            true_hits, len(samples) // 2, f"真实边上的采样命中太少: {true_hits}/{len(samples)}"
        )
        self.assertEqual(clamped_hits, 0, "被夹住的那条线上不该有任何像素")

    def test_antialiasing_is_requested(self) -> None:
        zone = self._zone("区域1")
        self.widget.set_zones([zone])
        self.widget.set_active_zone(zone)
        image = QImage(200, 150, QImage.Format.Format_RGB888)
        painter = QPainter(image)

        self.widget._draw_zones(painter)

        self.assertTrue(
            painter.renderHints() & QPainter.RenderHint.Antialiasing,
            "没有抗锯齿的折线全是锯齿，正是观感「突兀」的一大来源",
        )
        painter.end()


class FakeMouseEvent:
    """鼠标事件的最小替身：几个处理器只用到 position() 与 button()。"""

    def __init__(self, x: float, y: float, button: Qt.MouseButton = Qt.MouseButton.LeftButton):
        self._position = QPointF(x, y)
        self._button = button

    def position(self) -> QPointF:
        return self._position

    def button(self) -> Qt.MouseButton:
        return self._button


class ZoneEditingTests(unittest.TestCase):
    """区域的顶点编辑。

    这里钉住的是一个真发生过的 bug：顶点落在画面之外时拖不动。原因是 press 处理器
    **先**判断"点是否在画面内"，不在就 return，于是"抓顶点"那段根本没机会跑；而顶点
    落在画面外是允许的（换过不同分辨率的视频源、或者把顶点拖出去过都会这样），于是
    它就永远卡在画面外，既拖不动也删不掉。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        # 控件比画面高，于是上下各留一条黑边 —— 和用户截图里的情形一致。
        self.widget = VideoWidget()
        self.widget.resize(400, 400)
        self.widget.set_frame(np.full((200, 400, 3), 90, dtype=np.uint8))
        # 强制算一次 _video_rect（paintEvent 里才会更新）。
        self.widget._video_rect = self.widget._calculate_video_rect()

    def _zone(self, polygon: list[list[float]]) -> ZoneDefinition:
        zone = ZoneDefinition(name="区域1", polygon=polygon, closed=False)
        self.widget.set_zones([zone])
        self.widget.set_active_zone(zone)
        return zone

    def _display_of(self, x: float, y: float) -> QPointF:
        """顶点标记在控件里的落点 —— 用户实际点得到的位置。

        注意不是 `_to_display`：顶点被外推到控件之外时，标记会被夹到边缘（`_handle_position`），
        所以"标记画在哪"与"坐标本该映射到哪"不是一回事，测试要点的是前者。
        """
        return self.widget._handle_position(QPointF(x, y))[0]

    def test_vertex_outside_the_video_area_can_be_grabbed(self) -> None:
        """画面外的顶点必须能被抓住 —— 这是它唯一能被救回来的途径。"""
        self._zone([[50.0, 50.0], [350.0, 50.0], [200.0, 260.0]])
        outside = self._display_of(200.0, 260.0)
        self.assertFalse(
            self.widget._video_rect.contains(outside),
            "这个顶点应当落在画面之外（黑边里）",
        )

        self.widget.mousePressEvent(FakeMouseEvent(outside.x(), outside.y()))

        self.assertEqual(self.widget._drag_index, 2)

    def test_outside_vertex_can_be_dragged_back_into_the_frame(self) -> None:
        zone = self._zone([[50.0, 50.0], [350.0, 50.0], [200.0, 260.0]])
        outside = self._display_of(200.0, 260.0)
        self.widget.mousePressEvent(FakeMouseEvent(outside.x(), outside.y()))

        inside = self._display_of(200.0, 150.0)
        self.widget.mouseMoveEvent(FakeMouseEvent(inside.x(), inside.y()))

        self.assertEqual(zone.polygon[2], [200.0, 150.0])
        self.widget.mouseReleaseEvent(FakeMouseEvent(0, 0))
        self.assertIsNone(self.widget._drag_index)

    def test_dragging_a_vertex_out_of_the_frame_is_allowed(self) -> None:
        """反过来也要允许：警戒区贴着画面边缘、往外留一点余量是合理的。"""
        zone = self._zone([[50.0, 50.0], [350.0, 50.0], [200.0, 150.0]])
        grab = self._display_of(200.0, 150.0)
        self.widget.mousePressEvent(FakeMouseEvent(grab.x(), grab.y()))

        just_outside = self._display_of(200.0, 206.0)
        self.widget.mouseMoveEvent(FakeMouseEvent(just_outside.x(), just_outside.y()))

        self.assertEqual(zone.polygon[2], [200.0, 206.0])

    def test_dragging_far_outside_stays_within_a_sane_range(self) -> None:
        """拖太远会被夹住，配置里不会出现天文数字。"""
        zone = self._zone([[50.0, 50.0], [350.0, 50.0], [200.0, 150.0]])
        grab = self._display_of(200.0, 150.0)
        self.widget.mousePressEvent(FakeMouseEvent(grab.x(), grab.y()))

        self.widget.mouseMoveEvent(FakeMouseEvent(10_000.0, 10_000.0))

        x, y = zone.polygon[2]
        self.assertLessEqual(y, 200 * (1 + VideoWidget.OFF_FRAME_MARGIN) + 0.01)
        self.assertLessEqual(x, 400 * (1 + VideoWidget.OFF_FRAME_MARGIN) + 0.01)

    def test_vertex_far_outside_the_widget_is_still_grabbable(self) -> None:
        """顶点被外推到控件之外时，标记会被夹到边缘 —— 否则它永远抓不回来了。

        这是同一个 bug 的另一半：只限制拖动范围不够，屏幕位置是按比例外推的，窗口不大
        时画面外一点点的顶点在屏幕上就已经出界了。
        """
        self._zone([[50.0, 50.0], [350.0, 50.0], [900.0, 600.0]])
        handle, clamped = self.widget._handle_position(QPointF(900.0, 600.0))
        self.assertTrue(clamped)
        self.assertTrue(self.widget.rect().contains(handle.toPoint()))

        self.widget.mousePressEvent(FakeMouseEvent(handle.x(), handle.y()))

        self.assertEqual(self.widget._drag_index, 2)

    def test_dragging_a_clamped_vertex_back_inside_works(self) -> None:
        zone = self._zone([[50.0, 50.0], [350.0, 50.0], [900.0, 600.0]])
        handle, _ = self.widget._handle_position(QPointF(900.0, 600.0))
        self.widget.mousePressEvent(FakeMouseEvent(handle.x(), handle.y()))

        inside = self._display_of(200.0, 120.0)
        self.widget.mouseMoveEvent(FakeMouseEvent(inside.x(), inside.y()))

        self.assertEqual(zone.polygon[2], [200.0, 120.0])

    def test_clicking_the_letterbox_area_does_not_add_a_vertex(self) -> None:
        """点在黑边上不该凭空加一个点（那会造出一个看不见的顶点）。"""
        zone = self._zone([[50.0, 50.0], [350.0, 50.0]])
        # 画面下方那条黑边上的一个点，离任何顶点都很远。
        letterbox = QPointF(300.0, self.widget.height() - 10)
        self.assertFalse(self.widget._video_rect.contains(letterbox))

        self.widget.mousePressEvent(FakeMouseEvent(letterbox.x(), letterbox.y()))

        self.assertEqual(len(zone.polygon), 2)

    def test_clicking_inside_the_video_area_still_adds_a_vertex(self) -> None:
        zone = self._zone([[50.0, 50.0], [350.0, 50.0]])
        inside = self._display_of(200.0, 120.0)

        self.widget.mousePressEvent(FakeMouseEvent(inside.x(), inside.y()))

        self.assertEqual(zone.polygon[2], [200.0, 120.0])

    def test_right_click_removes_an_outside_vertex(self) -> None:
        zone = self._zone([[50.0, 50.0], [350.0, 50.0], [200.0, 260.0]])
        outside = self._display_of(200.0, 260.0)

        self.widget.mousePressEvent(
            FakeMouseEvent(outside.x(), outside.y(), Qt.MouseButton.RightButton)
        )

        self.assertEqual(len(zone.polygon), 2)

    def test_double_click_outside_closes_without_eating_a_vertex(self) -> None:
        """双击闭合时不该顺手删掉一个顶点：画面外的那一下并没有加过点。"""
        zone = self._zone([[50.0, 50.0], [350.0, 50.0], [200.0, 260.0]])
        outside = self._display_of(200.0, 260.0)

        self.widget.mouseDoubleClickEvent(FakeMouseEvent(outside.x(), outside.y()))

        self.assertEqual(len(zone.polygon), 3)
        self.assertTrue(zone.closed)

    def test_double_click_inside_still_closes_and_drops_the_duplicate(self) -> None:
        """双击的第一下会加点，闭合时要把它去掉 —— 但不能少于闭合所需的三个点。"""
        zone = self._zone([[50.0, 50.0], [350.0, 50.0], [200.0, 100.0]])
        point = self._display_of(200.0, 120.0)
        self.widget.mousePressEvent(FakeMouseEvent(point.x(), point.y()))
        self.assertEqual(len(zone.polygon), 4)

        self.widget.mouseDoubleClickEvent(FakeMouseEvent(point.x(), point.y()))

        self.assertEqual(len(zone.polygon), 3)
        self.assertTrue(zone.closed)


if __name__ == "__main__":
    unittest.main()
