from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import app_paths

APP_DIR = app_paths.APP_DIR
LOW_POWER_MODEL_DIR = app_paths.resource("models/yolo11n_int8_openvino_model")


@dataclass(frozen=True)
class InferencePolicy:
    model_path: str
    device: str
    imgsz: int | None = None
    detector_interval: int = 1
    label: str = "标准模式"

    @property
    def uses_tracker_prediction(self) -> bool:
        return self.detector_interval > 1


@dataclass(frozen=True)
class PolicyResolution:
    policy: InferencePolicy | None = None
    unavailable_reason: str = ""

    @property
    def available(self) -> bool:
        return self.policy is not None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_low_power_model() -> str:
    manifest_path = LOW_POWER_MODEL_DIR / "model_manifest.json"
    if not manifest_path.is_file():
        return "未找到本地 YOLO11n OpenVINO INT8 模型清单"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "本地 OpenVINO 模型清单无法读取"
    if manifest.get("quantization") != "INT8":
        return "本地 OpenVINO 模型不是 INT8 量化版本"
    if manifest.get("imgsz") != 512 or manifest.get("detector_interval") != 4:
        return "本地 OpenVINO 模型配置与低功耗预设不匹配"
    files = manifest.get("files")
    if not isinstance(files, dict):
        return "本地 OpenVINO 模型清单缺少文件校验信息"
    for name, expected_hash in files.items():
        path = LOW_POWER_MODEL_DIR / str(name)
        if not path.is_file() or _sha256(path) != expected_hash:
            return "本地 OpenVINO 模型文件缺失或已损坏"
    if not any(path.with_suffix(".bin").is_file() for path in LOW_POWER_MODEL_DIR.glob("*.xml")):
        return "本地 OpenVINO XML/BIN 文件不完整"
    return ""


def resolve_inference_policy(
    model_path: str,
    device: str,
    cpu_low_power_preset: bool,
) -> PolicyResolution:
    if not cpu_low_power_preset or device != "cpu":
        return PolicyResolution(
            policy=InferencePolicy(model_path=model_path, device=device),
        )

    try:
        import openvino  # noqa: F401
    except ImportError:
        return PolicyResolution(unavailable_reason="未安装 OpenVINO 运行时")

    unavailable_reason = _validate_low_power_model()
    if unavailable_reason:
        return PolicyResolution(unavailable_reason=unavailable_reason)

    return PolicyResolution(
        policy=InferencePolicy(
            model_path=str(LOW_POWER_MODEL_DIR),
            device="cpu",
            imgsz=512,
            detector_interval=4,
            label="CPU 低功耗模式（OpenVINO INT8，512，每 4 帧检测）",
        )
    )
