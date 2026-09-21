from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

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

    @staticmethod
    def _zone(name: str) -> ZoneDefinition:
        return ZoneDefinition(
            name=name,
            polygon=[[40.0, 40.0], [160.0, 40.0], [160.0, 120.0], [40.0, 120.0]],
            closed=True,
        )

    def _draw(self) -> np.ndarray:
        """把控件层画到一张干净图上，返回像素。"""
        image = QImage(200, 150, QImage.Format.Format_RGB888)
        image.fill(0)
        painter = QPainter(image)
        self.widget._draw_zones(painter)
        painter.end()
        buffer = image.constBits()
        array = np.frombuffer(buffer, dtype=np.uint8).reshape((150, image.bytesPerLine()))
        return array[:, : 200 * 3].reshape((150, 200, 3)).copy()

    def test_draws_nothing_when_no_zone_is_active(self) -> None:
        self.widget.set_zones([self._zone("区域1")])
        self.widget.set_active_zone(None)

        self.assertEqual(self._draw().sum(), 0)

    def test_draws_the_active_zone(self) -> None:
        zone = self._zone("区域1")
        self.widget.set_zones([zone])
        self.widget.set_active_zone(zone)

        painted = self._draw()

        self.assertGreater(painted.sum(), 0)
        # 只在区域轮廓附近有像素：中心是空的（没有填充，填充归引擎画）。
        self.assertEqual(painted[80, 100].sum(), 0)

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


if __name__ == "__main__":
    unittest.main()
