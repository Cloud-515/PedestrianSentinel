from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np

from detection_engine import DetectionEngine
from inference_profiles import InferencePolicy
from models import AlarmEvent, ZoneDefinition


class EmptyDetections:
    tracker_id = None
    xyxy = np.empty((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return 0


class TrackedDetections:
    def __init__(self, box: list[float], track_id: int = 1) -> None:
        self.tracker_id = np.asarray([track_id])
        self.xyxy = np.asarray([box], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.xyxy)


class DetectionDeviceTests(unittest.TestCase):
    def test_monitor_event_times_use_local_computer_time(self) -> None:
        timestamp = 1786588135.995
        event = AlarmEvent(
            source="rtsp://camera",
            zone_name="区域1",
            track_id="0",
            entered_at_seconds=timestamp,
            alarm_at_seconds=timestamp + 12.041,
            wall_time="2026-08-13 10:29:08",
            operation_mode="monitor",
        )

        entered_at = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        alarm_at = datetime.fromtimestamp(timestamp + 12.041).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        self.assertEqual(event.format_event_time(event.entered_at_seconds, precision=3), entered_at)
        self.assertEqual(event.to_row()[4:7], [entered_at, alarm_at, "未记录"])

    def test_video_event_times_remain_playback_seconds(self) -> None:
        event = AlarmEvent(
            source="test.mp4",
            zone_name="区域1",
            track_id="0",
            entered_at_seconds=12.3456,
            alarm_at_seconds=24.5678,
            wall_time="2026-08-13 10:29:08",
            operation_mode="video",
        )

        self.assertEqual(event.format_event_time(event.entered_at_seconds, precision=3), "12.346s")
        self.assertEqual(event.to_row()[4:6], ["12.35s", "24.57s"])

    def test_engine_passes_device_only_to_inference(self) -> None:
        model = Mock()
        model.return_value = [object()]
        tracker = Mock()
        tracker.update.return_value = EmptyDetections()

        with (
            patch("detection_engine.YOLO", return_value=model) as yolo_factory,
            patch.object(DetectionEngine, "_new_tracker", return_value=tracker),
            patch(
                "detection_engine.sv.Detections.from_ultralytics",
                return_value=object(),
            ),
        ):
            engine = DetectionEngine("model.pt", [], "cuda:1")
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            engine.process(frame, 0.0, "test.mp4")

        yolo_factory.assert_called_once_with("model.pt")
        model.assert_called_once_with(
            frame,
            classes=[0],
            verbose=False,
            device="cuda:1",
        )

    def test_process_can_skip_rendering_for_benchmarking(self) -> None:
        model = Mock(return_value=[object()])
        tracker = Mock()
        tracker.update.return_value = EmptyDetections()
        frame = np.zeros((16, 16, 3), dtype=np.uint8)

        with (
            patch("detection_engine.YOLO", return_value=model),
            patch.object(DetectionEngine, "_new_tracker", return_value=tracker),
            patch("detection_engine.sv.Detections.from_ultralytics", return_value=object()),
            patch.object(DetectionEngine, "_annotate") as annotate,
        ):
            engine = DetectionEngine("model.pt", [], "cpu")
            output, events = engine.process(frame, 0.0, "test.mp4", render=False)

        self.assertIs(output, frame)
        self.assertEqual(events, [])
        annotate.assert_not_called()

    def test_low_power_mode_detects_every_fourth_frame_and_uses_predictions(self) -> None:
        model = Mock(return_value=[object()])
        tracker = Mock()
        detector_detections = EmptyDetections()
        predicted_detections = TrackedDetections([10, 10, 30, 50])
        tracker.update.return_value = detector_detections
        tracker.tracked_objects = predicted_detections
        policy = InferencePolicy(
            model_path="model_openvino",
            device="cpu",
            imgsz=512,
            detector_interval=4,
            label="CPU 低功耗模式",
        )
        frame = np.zeros((120, 120, 3), dtype=np.uint8)

        with (
            patch("detection_engine.YOLO", return_value=model),
            patch.object(DetectionEngine, "_new_tracker", return_value=tracker),
            patch(
                "detection_engine.sv.Detections.from_ultralytics",
                return_value=object(),
            ),
        ):
            engine = DetectionEngine("model.pt", [], "cpu", policy)
            for index in range(5):
                engine.process(frame, float(index), "test.mp4")

        self.assertEqual(model.call_count, 2)
        self.assertEqual(model.call_args.kwargs["imgsz"], 512)
        self.assertEqual(model.call_args.kwargs["device"], "cpu")
        self.assertEqual(tracker.update.call_count, 2)
        self.assertEqual(
            [call.kwargs["timestamp"] for call in tracker.update.call_args_list],
            [0.0, 4.0],
        )

    def test_low_power_mode_keeps_zone_entry_and_emits_dwell_alarm(self) -> None:
        model = Mock(return_value=[object()])
        tracker = Mock()
        tracked = TrackedDetections([10, 10, 30, 50], track_id=7)
        tracker.update.return_value = tracked
        tracker.tracked_objects = tracked
        policy = InferencePolicy("model_openvino", "cpu", imgsz=512, detector_interval=4)
        zone = ZoneDefinition(
            name="警戒区",
            polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
            closed=True,
            dwell_seconds=2.0,
            cooldown_seconds=30.0,
        )
        frame = np.zeros((120, 120, 3), dtype=np.uint8)

        with (
            patch("detection_engine.YOLO", return_value=model),
            patch.object(DetectionEngine, "_new_tracker", return_value=tracker),
            patch("detection_engine.sv.Detections.from_ultralytics", return_value=object()),
        ):
            engine = DetectionEngine("model.pt", [zone], "cpu", policy)
            events = []
            for timestamp in range(5):
                _, frame_events = engine.process(frame, float(timestamp), "test.mp4")
                events.extend(frame_events)

        self.assertEqual(tracker.update.call_count, 2)
        self.assertEqual(engine.entry_times, {("警戒区", 7): 0.0})
        self.assertEqual([transition.kind for transition in events], ["entered", "alarmed"])
        self.assertEqual(events[-1].event.track_id, "7")
        self.assertEqual(events[-1].event.entered_at_seconds, 0.0)
        self.assertEqual(events[-1].event.alarm_at_seconds, 2.0)

    def test_low_power_tracking_reset_restarts_detector_cadence(self) -> None:
        model = Mock(return_value=[object()])
        first_tracker = Mock()
        second_tracker = Mock()
        first_tracker.update.return_value = EmptyDetections()
        second_tracker.update.return_value = EmptyDetections()
        first_tracker.tracked_objects = EmptyDetections()
        second_tracker.tracked_objects = EmptyDetections()
        policy = InferencePolicy("model_openvino", "cpu", imgsz=512, detector_interval=4)
        frame = np.zeros((16, 16, 3), dtype=np.uint8)

        with (
            patch("detection_engine.YOLO", return_value=model),
            patch.object(
                DetectionEngine,
                "_new_tracker",
                side_effect=[first_tracker, second_tracker],
            ),
            patch(
                "detection_engine.sv.Detections.from_ultralytics",
                return_value=object(),
            ),
        ):
            engine = DetectionEngine("model.pt", [], "cpu", policy)
            engine.process(frame, 0.0, "test.mp4")
            engine.process(frame, 1.0, "test.mp4")
            engine.reset_tracking()
            engine.process(frame, 2.0, "test.mp4")

        self.assertEqual(model.call_count, 2)
        second_tracker.update.assert_called_once()

    def test_intrusion_session_closes_short_visit_and_creates_new_session_on_reentry(self) -> None:
        model = Mock(return_value=[object()])
        tracker = Mock()
        person = TrackedDetections([10, 10, 30, 50], track_id=4)
        tracker.update.side_effect = [person, EmptyDetections(), person, EmptyDetections()]
        zone = ZoneDefinition(
            name="警戒区",
            polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
            closed=True,
            dwell_seconds=5.0,
        )
        frame = np.zeros((120, 120, 3), dtype=np.uint8)

        with (
            patch("detection_engine.YOLO", return_value=model),
            patch.object(DetectionEngine, "_new_tracker", return_value=tracker),
            patch("detection_engine.sv.Detections.from_ultralytics", return_value=object()),
        ):
            engine = DetectionEngine("model.pt", [zone], "cpu")
            _, entered = engine.process(frame, 10.0, "test.mp4")
            _, exited = engine.process(frame, 12.0, "test.mp4")
            _, reentered = engine.process(frame, 20.0, "test.mp4")
            _, exited_again = engine.process(frame, 21.0, "test.mp4")

        self.assertEqual([transition.kind for transition in entered], ["entered"])
        self.assertEqual([transition.kind for transition in exited], ["exited"])
        self.assertEqual(exited[0].event.duration_seconds, 2.0)
        self.assertFalse(exited[0].event.alarmed)
        self.assertEqual([transition.kind for transition in reentered], ["entered"])
        self.assertNotEqual(entered[0].event.session_id, reentered[0].event.session_id)
        self.assertEqual([transition.kind for transition in exited_again], ["exited"])

    def test_intrusion_session_only_emits_one_alarm_while_target_remains(self) -> None:
        model = Mock(return_value=[object()])
        tracker = Mock()
        person = TrackedDetections([10, 10, 30, 50], track_id=5)
        tracker.update.side_effect = [person, person, person, EmptyDetections()]
        zone = ZoneDefinition(
            name="警戒区",
            polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
            closed=True,
            dwell_seconds=2.0,
        )
        frame = np.zeros((120, 120, 3), dtype=np.uint8)

        with (
            patch("detection_engine.YOLO", return_value=model),
            patch.object(DetectionEngine, "_new_tracker", return_value=tracker),
            patch("detection_engine.sv.Detections.from_ultralytics", return_value=object()),
        ):
            engine = DetectionEngine("model.pt", [zone], "cpu")
            transitions = []
            for timestamp in (0.0, 2.0, 5.0, 6.0):
                _, frame_transitions = engine.process(frame, timestamp, "test.mp4")
                transitions.extend(frame_transitions)

        self.assertEqual([transition.kind for transition in transitions], ["entered", "alarmed", "exited"])
        self.assertEqual(transitions[1].event.alarm_at_seconds, 2.0)
        self.assertEqual(transitions[-1].event.duration_seconds, 6.0)

    def test_format_elapsed_uses_seconds_and_milliseconds(self) -> None:
        self.assertEqual(DetectionEngine._format_elapsed(0), "0.000s")
        self.assertEqual(DetectionEngine._format_elapsed(5.237), "5.237s")
        self.assertEqual(DetectionEngine._format_elapsed(61.9), "61.900s")
        self.assertEqual(DetectionEngine._format_elapsed(-1), "0.000s")

    def test_process_displays_elapsed_time_and_restarts_after_exit(self) -> None:
        model = Mock(return_value=[object()])
        tracker = Mock()
        tracked = TrackedDetections([10, 10, 30, 50])
        tracker.update.side_effect = [tracked, EmptyDetections(), tracked]
        zone = ZoneDefinition(
            name="警戒区",
            polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
            closed=True,
            dwell_seconds=30,
        )
        frame = np.zeros((120, 120, 3), dtype=np.uint8)

        with (
            patch("detection_engine.YOLO", return_value=model),
            patch.object(DetectionEngine, "_new_tracker", return_value=tracker),
            patch(
                "detection_engine.sv.Detections.from_ultralytics",
                return_value=object(),
            ),
            patch("detection_engine.cv2.putText") as put_text,
        ):
            engine = DetectionEngine("model.pt", [zone], "cpu")
            engine.process(frame, 10.0, "test.mp4")
            engine.process(frame, 12.5, "test.mp4")
            self.assertEqual(engine.entry_times, {})
            engine.process(frame, 20.0, "test.mp4")

        labels = [call.args[1] for call in put_text.call_args_list if call.args[1].startswith("ID:")]
        self.assertEqual(labels, ["ID:1 IN ZONE 0.000s", "ID:1 IN ZONE 0.000s"])

    def test_process_uses_longest_elapsed_time_for_overlapping_zones(self) -> None:
        model = Mock(return_value=[object()])
        tracker = Mock()
        tracker.update.side_effect = [
            TrackedDetections([10, 10, 30, 50]),
            TrackedDetections([25, 10, 45, 50]),
        ]
        zones = [
            ZoneDefinition(
                name="区域A",
                polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
                closed=True,
                dwell_seconds=30,
            ),
            ZoneDefinition(
                name="区域B",
                polygon=[[30, 0], [100, 0], [100, 100], [30, 100]],
                closed=True,
                dwell_seconds=30,
            ),
        ]
        frame = np.zeros((120, 120, 3), dtype=np.uint8)

        with (
            patch("detection_engine.YOLO", return_value=model),
            patch.object(DetectionEngine, "_new_tracker", return_value=tracker),
            patch(
                "detection_engine.sv.Detections.from_ultralytics",
                return_value=object(),
            ),
            patch("detection_engine.cv2.putText") as put_text,
        ):
            engine = DetectionEngine("model.pt", zones, "cpu")
            engine.process(frame, 10.0, "test.mp4")
            engine.process(frame, 15.0, "test.mp4")

        self.assertEqual(set(engine.entry_times), {("区域A", 1), ("区域B", 1)})
        labels = [call.args[1] for call in put_text.call_args_list if call.args[1].startswith("ID:")]
        self.assertEqual(labels[-1], "ID:1 IN ZONE 5.000s")
        self.assertEqual(len(labels), 2)


class AlarmCooldownTests(unittest.TestCase):
    """冷却时间：同一目标同一区域在冷却期内不重复报警。

    这里的重点是「离开又立刻回来」。区域边界上的徘徊、或者 ByteTrack 丢一帧 ID
    又找回，都会关掉旧会话、开一个新会话并重新计时 —— 如果不看冷却，每一轮都会
    放告警音、写一条 JSONL、存一张截图。
    """

    @staticmethod
    def _zone(name: str = "警戒区", **overrides: float) -> ZoneDefinition:
        return ZoneDefinition(
            name=name,
            polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
            closed=True,
            **overrides,
        )

    @contextmanager
    def _engine(self, zone: ZoneDefinition, frames: list[object]):
        """构造一个由假 tracker 驱动的引擎：frames 里每一项就是那一帧的检测结果。"""
        model = Mock(return_value=[object()])
        tracker = Mock()
        tracker.update.side_effect = frames
        with (
            patch("detection_engine.YOLO", return_value=model),
            patch.object(DetectionEngine, "_new_tracker", return_value=tracker),
            patch("detection_engine.sv.Detections.from_ultralytics", return_value=object()),
        ):
            yield DetectionEngine("model.pt", [zone], "cpu")

    @staticmethod
    def _drive(engine: DetectionEngine, timestamps: list[float]) -> list:
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        transitions = []
        for timestamp in timestamps:
            _, frame_transitions = engine.process(frame, timestamp, "test.mp4")
            transitions.extend(frame_transitions)
        return transitions

    @staticmethod
    def _alarm_times(transitions: list) -> list[float]:
        return [
            transition.event.alarm_at_seconds
            for transition in transitions
            if transition.kind == "alarmed"
        ]

    def test_cooldown_suppresses_alarm_when_target_reenters_zone(self) -> None:
        zone = self._zone(dwell_seconds=2.0, cooldown_seconds=10.0)
        person = TrackedDetections([10, 10, 30, 50], track_id=8)
        frames = [person, person, EmptyDetections(), person, person, EmptyDetections()]

        with self._engine(zone, frames) as engine:
            transitions = self._drive(engine, [0.0, 2.0, 3.0, 4.0, 6.0, 7.0])

        self.assertEqual(
            [transition.kind for transition in transitions],
            ["entered", "alarmed", "exited", "entered", "exited"],
        )
        # 第二个会话被压住：没有 alarmed，事件也不能标成已报警，否则界面会显示成误报。
        self.assertFalse(transitions[-1].event.alarmed)
        self.assertIsNone(transitions[-1].event.alarm_at_seconds)

    def test_cooldown_lapse_allows_alarm_on_later_reentry(self) -> None:
        zone = self._zone(dwell_seconds=2.0, cooldown_seconds=10.0)
        person = TrackedDetections([10, 10, 30, 50], track_id=8)
        frames = [person, person, EmptyDetections(), person, person]

        with self._engine(zone, frames) as engine:
            transitions = self._drive(engine, [0.0, 2.0, 3.0, 20.0, 22.0])

        self.assertEqual(self._alarm_times(transitions), [2.0, 22.0])

    def test_suppressed_session_still_alarms_once_cooldown_lapses(self) -> None:
        """被压住的会话不会被永久吞掉：目标一直没走，冷却期一过就补报一次。"""
        zone = self._zone(dwell_seconds=2.0, cooldown_seconds=10.0)
        person = TrackedDetections([10, 10, 30, 50], track_id=8)
        frames = [person, person, EmptyDetections(), person, person, person]

        with self._engine(zone, frames) as engine:
            transitions = self._drive(engine, [0.0, 2.0, 3.0, 4.0, 6.0, 12.0])

        self.assertEqual(self._alarm_times(transitions), [2.0, 12.0])

    def test_cooldown_is_tracked_per_target(self) -> None:
        zone = self._zone(dwell_seconds=2.0, cooldown_seconds=10.0)
        first = TrackedDetections([10, 10, 30, 50], track_id=1)
        second = TrackedDetections([10, 10, 30, 50], track_id=2)
        frames = [first, first, second, second]

        with self._engine(zone, frames) as engine:
            transitions = self._drive(engine, [0.0, 2.0, 3.0, 5.0])

        # 换了目标就该报 —— 冷却只压同一个 (区域, 目标ID)。
        alarmed = [transition for transition in transitions if transition.kind == "alarmed"]
        self.assertEqual([transition.event.track_id for transition in alarmed], ["1", "2"])

    def test_zero_cooldown_alarms_on_every_reentry(self) -> None:
        zone = self._zone(dwell_seconds=2.0, cooldown_seconds=0.0)
        person = TrackedDetections([10, 10, 30, 50], track_id=8)
        frames = [person, person, EmptyDetections(), person, person]

        with self._engine(zone, frames) as engine:
            transitions = self._drive(engine, [0.0, 2.0, 3.0, 4.0, 6.0])

        self.assertEqual(self._alarm_times(transitions), [2.0, 6.0])

    def test_reset_tracking_clears_cooldown_history(self) -> None:
        """布撤防切换、换源、循环播放都会 reset —— 时间轴变了，旧的报警时刻不能留。"""
        zone = self._zone(dwell_seconds=2.0, cooldown_seconds=10.0)
        person = TrackedDetections([10, 10, 30, 50], track_id=8)
        frames = [person, person, person, person]

        with self._engine(zone, frames) as engine:
            transitions = self._drive(engine, [0.0, 2.0])
            engine.reset_tracking()
            transitions.extend(self._drive(engine, [3.0, 5.0]))

        self.assertEqual(self._alarm_times(transitions), [2.0, 5.0])

    def test_cooldown_history_is_pruned_once_it_stops_mattering(self) -> None:
        """ByteTrack 的 ID 只增不减，这个 dict 不清理的话会一直长。"""
        zone = self._zone(dwell_seconds=2.0, cooldown_seconds=10.0)
        person = TrackedDetections([10, 10, 30, 50], track_id=8)
        frames = [person, person, EmptyDetections(), EmptyDetections()]

        with self._engine(zone, frames) as engine:
            self._drive(engine, [0.0, 2.0])
            self.assertEqual(list(engine._last_alarm_at), [("警戒区", 8)])
            self._drive(engine, [3.0])
            self.assertEqual(list(engine._last_alarm_at), [("警戒区", 8)])
            self._drive(engine, [20.0])
            self.assertEqual(engine._last_alarm_at, {})

    def test_cooldown_history_drops_entries_for_removed_zones(self) -> None:
        zone = self._zone(dwell_seconds=2.0, cooldown_seconds=10.0)
        person = TrackedDetections([10, 10, 30, 50], track_id=8)
        frames = [person, person, EmptyDetections()]

        with self._engine(zone, frames) as engine:
            self._drive(engine, [0.0, 2.0])
            # 区域被改名或删掉之后，它名下的冷却记录永远不会再被查到，留着只是泄漏。
            engine.update_zones(
                [self._zone("改名后的区域", dwell_seconds=2.0, cooldown_seconds=10.0)]
            )
            self._drive(engine, [3.0])
            self.assertEqual(engine._last_alarm_at, {})


if __name__ == "__main__":
    unittest.main()
