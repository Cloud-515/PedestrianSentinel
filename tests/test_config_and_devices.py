from __future__ import annotations

import unittest
from unittest.mock import patch

import compute_devices
from models import AppConfig


class AppConfigDeviceTests(unittest.TestCase):
    def test_old_config_defaults_to_cpu(self) -> None:
        self.assertEqual(AppConfig.from_dict({}).inference_device, "cpu")

    def test_device_round_trips(self) -> None:
        config = AppConfig(inference_device="cuda:2")
        restored = AppConfig.from_dict(config.to_dict())
        self.assertEqual(restored.inference_device, "cuda:2")

    def test_config_migrates_legacy_file_settings_to_video_mode(self) -> None:
        config = AppConfig.from_dict({"source": "demo.mp4", "source_type": "file"})
        self.assertEqual(config.operation_mode, "video")
        self.assertEqual(config.video_source, "demo.mp4")
        self.assertEqual(config.monitor_source, "0")

    def test_config_preserves_separate_mode_sources(self) -> None:
        config = AppConfig(
            operation_mode="video",
            video_source="demo.mp4",
            monitor_source="rtsp://camera/live",
        )
        restored = AppConfig.from_dict(config.to_dict())
        self.assertEqual(restored.operation_mode, "video")
        self.assertEqual(restored.video_source, "demo.mp4")
        self.assertEqual(restored.monitor_source, "rtsp://camera/live")

    def test_config_rejects_unknown_operation_mode(self) -> None:
        config = AppConfig.from_dict({"operation_mode": "unknown"})
        self.assertEqual(config.operation_mode, "monitor")


class DeviceDiscoveryTests(unittest.TestCase):
    def test_cpu_only_runtime_lists_only_cpu(self) -> None:
        with patch.object(compute_devices.torch.cuda, "is_available", return_value=False):
            self.assertEqual(compute_devices.enumerate_inference_devices(), [("cpu", "CPU")])

    def test_cuda_runtime_lists_all_visible_devices(self) -> None:
        with (
            patch.object(compute_devices.torch.cuda, "is_available", return_value=True),
            patch.object(compute_devices.torch.cuda, "device_count", return_value=2),
            patch.object(
                compute_devices.torch.cuda,
                "get_device_name",
                side_effect=["First GPU", "Second GPU"],
            ),
        ):
            self.assertEqual(
                compute_devices.enumerate_inference_devices(),
                [
                    ("cpu", "CPU"),
                    ("cuda:0", "GPU 0: First GPU"),
                    ("cuda:1", "GPU 1: Second GPU"),
                ],
            )

    def test_cuda_discovery_failure_falls_back_to_cpu(self) -> None:
        with patch.object(
            compute_devices.torch.cuda,
            "is_available",
            side_effect=RuntimeError("driver error"),
        ):
            self.assertEqual(compute_devices.enumerate_inference_devices(), [("cpu", "CPU")])


if __name__ == "__main__":
    unittest.main()
