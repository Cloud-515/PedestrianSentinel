from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from detection_engine import DetectionEngine
from models import ZoneDefinition


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


if __name__ == "__main__":
    unittest.main()
