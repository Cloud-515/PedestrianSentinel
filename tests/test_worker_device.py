from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alarm_service import EventStore
from detection_worker import DetectionWorker
from video_source import VideoSourceSpec


class FakeVideoSource:
    last_instance: "FakeVideoSource | None" = None

    def __init__(self, spec: VideoSourceSpec) -> None:
        self.spec = spec
        self.width = 640
        self.height = 480
        self.duration_seconds = 10.0
        self.closed = False
        FakeVideoSource.last_instance = self

    def open(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


class WorkerDeviceTests(unittest.TestCase):
    def test_engine_initialization_failure_keeps_error_status(self) -> None:
        statuses: list[str] = []
        event_store = EventStore(Path(tempfile.mkdtemp()) / "events")
        worker = DetectionWorker(
            spec=VideoSourceSpec("test.mp4", source_type="file"),
            model_path="model.pt",
            device="cuda:0",
            zones=[],
            event_store=event_store,
        )
        worker.status_changed.connect(statuses.append)

        with (
            patch("detection_worker.VideoSource", FakeVideoSource),
            patch(
                "detection_worker.DetectionEngine",
                side_effect=RuntimeError("CUDA unavailable"),
            ) as engine_factory,
        ):
            worker.run()

        engine_factory.assert_called_once_with("model.pt", [], "cuda:0")
        self.assertTrue(FakeVideoSource.last_instance.closed)
        self.assertTrue(statuses[-1].startswith("检测错误:"))
        self.assertNotIn("检测已停止", statuses[-1])


if __name__ == "__main__":
    unittest.main()
