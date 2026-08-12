from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import inference_profiles


class InferenceProfileTests(unittest.TestCase):
    def test_standard_policy_preserves_selected_model_and_device(self) -> None:
        result = inference_profiles.resolve_inference_policy("model.pt", "cuda:0", True)

        self.assertTrue(result.available)
        self.assertEqual(result.policy.model_path, "model.pt")
        self.assertEqual(result.policy.device, "cuda:0")
        self.assertEqual(result.policy.detector_interval, 1)
        self.assertIsNone(result.policy.imgsz)

    def test_low_power_policy_requires_openvino(self) -> None:
        with patch.dict("sys.modules", {"openvino": None}):
            result = inference_profiles.resolve_inference_policy("model.pt", "cpu", True)

        self.assertFalse(result.available)
        self.assertEqual(result.unavailable_reason, "未安装 OpenVINO 运行时")

    def test_low_power_policy_requires_complete_local_model(self) -> None:
        with (
            patch.dict("sys.modules", {"openvino": object()}),
            patch.object(inference_profiles, "LOW_POWER_MODEL_DIR", Path("missing-model")),
        ):
            result = inference_profiles.resolve_inference_policy("model.pt", "cpu", True)

        self.assertFalse(result.available)
        self.assertEqual(result.unavailable_reason, "未找到本地 YOLO11n OpenVINO INT8 模型清单")

    def test_low_power_policy_uses_local_openvino_int8_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            xml_path = model_dir / "yolo.xml"
            bin_path = model_dir / "yolo.bin"
            xml_path.write_bytes(b"xml")
            bin_path.write_bytes(b"bin")
            manifest = {
                "quantization": "INT8",
                "imgsz": 512,
                "detector_interval": 4,
                "files": {
                    xml_path.name: hashlib.sha256(xml_path.read_bytes()).hexdigest(),
                    bin_path.name: hashlib.sha256(bin_path.read_bytes()).hexdigest(),
                },
            }
            (model_dir / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with (
                patch.dict("sys.modules", {"openvino": object()}),
                patch.object(inference_profiles, "LOW_POWER_MODEL_DIR", model_dir),
            ):
                result = inference_profiles.resolve_inference_policy("model.pt", "cpu", True)

        self.assertTrue(result.available)
        self.assertEqual(result.policy.model_path, str(model_dir))
        self.assertEqual(result.policy.imgsz, 512)
        self.assertEqual(result.policy.detector_interval, 4)


if __name__ == "__main__":
    unittest.main()
