from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import nncf
import openvino
import ultralytics
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_model(weights: Path, data: Path, output_dir: Path) -> None:
    if not weights.is_file() or not data.is_file():
        raise SystemExit("权重或校准 YAML 不存在")
    if output_dir.exists():
        raise SystemExit(f"目标模型目录已存在: {output_dir}")

    model = YOLO(str(weights))
    exported = Path(
        model.export(
            format="openvino",
            quantize=8,
            imgsz=512,
            data=str(data),
            fraction=1.0,
            device="cpu",
        )
    )
    xml_files = list(exported.glob("*.xml"))
    if len(xml_files) != 1 or not xml_files[0].with_suffix(".bin").is_file():
        raise RuntimeError("导出的 OpenVINO XML/BIN 文件不完整")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), str(output_dir))
    xml_path = next(output_dir.glob("*.xml"))
    bin_path = xml_path.with_suffix(".bin")
    smoke = YOLO(str(output_dir))
    result = smoke(np.zeros((512, 512, 3), dtype=np.uint8), device="cpu", imgsz=512, verbose=False)
    if not result:
        raise RuntimeError("OpenVINO 模型加载验证失败")

    calibration_manifest = data.parent / "calibration_manifest.json"
    manifest = {
        "quantization": "INT8",
        "source_weights": str(weights.resolve()),
        "source_weights_sha256": sha256(weights),
        "calibration_manifest_sha256": sha256(calibration_manifest) if calibration_manifest.is_file() else "",
        "imgsz": 512,
        "detector_interval": 4,
        "files": {xml_path.name: sha256(xml_path), bin_path.name: sha256(bin_path)},
        "toolchain": {
            "python": sys.version,
            "platform": platform.platform(),
            "ultralytics": ultralytics.__version__,
            "openvino": openvino.__version__,
            "nncf": nncf.__version__,
        },
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="离线导出 YOLO11n OpenVINO INT8 模型")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_model(args.weights, args.data, args.output)


if __name__ == "__main__":
    main()
