from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alarm_service import EventStore
from detection_worker import DetectionWorker
from inference_profiles import InferencePolicy
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
            spec=VideoSourceSpec("test.mp4", operation_mode="video"),
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

    def test_worker_forwards_low_power_policy_to_engine(self) -> None:
        statuses: list[str] = []
        event_store = EventStore(Path(tempfile.mkdtemp()) / "events")
        policy = InferencePolicy(
            model_path="model_openvino",
            device="cpu",
            imgsz=512,
            detector_interval=4,
            label="CPU 低功耗模式",
        )
        worker = DetectionWorker(
            spec=VideoSourceSpec("test.mp4", operation_mode="video"),
            model_path=policy.model_path,
            device="cpu",
            zones=[],
            event_store=event_store,
            policy=policy,
        )
        worker.status_changed.connect(statuses.append)

        with (
            patch("detection_worker.VideoSource", FakeVideoSource),
            patch(
                "detection_worker.DetectionEngine",
                side_effect=RuntimeError("OpenVINO unavailable"),
            ) as engine_factory,
        ):
            worker.run()

        engine_factory.assert_called_once_with("model_openvino", [], "cpu", policy)
        self.assertTrue(statuses[-1].startswith("检测错误:"))


if __name__ == "__main__":
    unittest.main()
