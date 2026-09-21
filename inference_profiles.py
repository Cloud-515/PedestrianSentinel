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
    # 只检测「警戒区外扩这么多」的裁片；None = 检测整帧。
    #
    # 它**不是**省 CPU 的手段（实测过，见下），买的是**区域内**的识别率：网络的输入尺寸
    # 固定，裁片不会让每帧便宜，但它把同样的输入像素全花在要害处 —— 也就是区域内的
    # 有效分辨率更高。1080p / INT8 512 上的实测（15 帧、同一个引擎）：
    #
    #     整帧：每帧 32.2ms，15 帧共 262 个检出，**区域内**平均置信度 0.53
    #     裁片：每帧 39.7ms，15 帧共  93 个检出，**区域内**平均置信度 0.67
    #
    # 两个直接后果：①区外的误检（橱窗、栏杆、反光）根本不会产生，操作员不用再从
    # 一片绿框里分辨真假；②区域内的人置信度更高，更容易越过「新建轨迹 0.65」那道
    # 门槛 —— 低功耗模式下"真人检得到、却拿不到 ID"的症结就在这个刻度差上。
    # 代价是**区外的框不再产生**，所以低功耗模式只保证区域内的检测。
    #
    # （顺带记一笔：ultralytics 在 batch=1 时写死 OpenVINO 的 PERFORMANCE_HINT=LATENCY。
    # 实测 LATENCY 16.8ms、THROUGHPUT 21.7ms、多流 22.8~24.0ms —— 现在这个设置已经
    # 是最快的，不必去改后端配置。）
    detect_region_margin: float | None = None

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
            label="CPU 低功耗模式（OpenVINO INT8，512，每 4 帧检测，只检测警戒区）",
            detect_region_margin=0.3,
        )
    )
