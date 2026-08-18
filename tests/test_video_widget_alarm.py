from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from models import AlarmEvent
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


if __name__ == "__main__":
    unittest.main()
