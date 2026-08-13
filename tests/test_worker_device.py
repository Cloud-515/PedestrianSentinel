from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from alarm_service import EventStore
from detection_worker import DetectionWorker
from inference_profiles import InferencePolicy
from models import AlarmEvent
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
    def test_event_store_folds_intrusion_session_updates(self) -> None:
        root = Path(tempfile.mkdtemp()) / "events"
        store = EventStore(root)
        event = AlarmEvent(
            source="camera",
            zone_name="警戒区",
            track_id="3",
            entered_at_seconds=10.0,
            alarm_at_seconds=None,
            wall_time="2026-08-13 10:00:00",
            operation_mode="video",
        )
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        store.open_session(event, frame)
        event.alarm_at_seconds = 12.0
        event.status = "alarmed"
        store.mark_alarmed(event, frame)
        event.exited_at_seconds = 15.0
        event.duration_seconds = 5.0
        event.status = "completed"
        store.close_session(event)

        events = store.load_recent()
        self.assertEqual(len(events), 1)
        loaded = events[0]
        self.assertEqual(loaded.session_id, event.session_id)
        self.assertEqual(loaded.alarm_at_seconds, 12.0)
        self.assertEqual(loaded.exited_at_seconds, 15.0)
        self.assertEqual(loaded.duration_seconds, 5.0)
        self.assertTrue(Path(loaded.entry_screenshot_path).exists())
        self.assertTrue(Path(loaded.alarm_screenshot_path).exists())

    def test_event_store_loads_legacy_single_screenshot_record(self) -> None:
        root = Path(tempfile.mkdtemp()) / "events"
        root.mkdir()
        screenshot = root / "legacy.jpg"
        cv2.imwrite(str(screenshot), np.zeros((4, 4, 3), dtype=np.uint8))
        legacy = {
            "time": "2026-08-13 10:00:00",
            "video_source": "camera",
            "operation_mode": "monitor",
            "zone_name": "警戒区",
            "track_id": "1",
            "entered_at_seconds": 1786588000.0,
            "alarm_at_seconds": 1786588005.0,
            "screenshot_path": str(screenshot),
        }
        (root / "alarm_events.jsonl").write_text(json.dumps(legacy) + "\n", encoding="utf-8")

        loaded = EventStore(root).load_recent()[0]
        self.assertTrue(loaded.alarmed)
        self.assertEqual(loaded.alarm_screenshot_path, str(screenshot))
        self.assertEqual(loaded.screenshot_path, str(screenshot))

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
