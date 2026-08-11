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
