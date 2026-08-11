from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from detection_engine import DetectionEngine


class EmptyDetections:
    tracker_id = None
    xyxy = np.empty((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return 0


class DetectionDeviceTests(unittest.TestCase):
    def test_engine_passes_device_only_to_inference(self) -> None:
        model = Mock()
        model.return_value = [object()]
        tracker = Mock()
        tracker.update_with_detections.return_value = EmptyDetections()

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


if __name__ == "__main__":
    unittest.main()
