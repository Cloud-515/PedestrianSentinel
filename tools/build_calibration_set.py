from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_calibration_set(video_path: Path, output_dir: Path, count: int) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0 or fps <= 0:
        capture.release()
        raise RuntimeError("无法读取视频帧数或帧率")

    image_dir = output_dir / "images" / "val"
    label_dir = output_dir / "labels" / "val"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    indices = sorted({round(index * (frame_count - 1) / (count - 1)) for index in range(count)})
    records = []
    for ordinal, frame_index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"无法读取校准帧 {frame_index}")
        image_path = image_dir / f"frame_{ordinal:04d}.jpg"
        if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            capture.release()
            raise RuntimeError(f"无法写入校准图像 {image_path}")
        (label_dir / f"frame_{ordinal:04d}.txt").touch()
        records.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": frame_index / fps,
                "image": image_path.relative_to(output_dir).as_posix(),
                "sha256": sha256(image_path),
            }
        )
    capture.release()

    data_yaml = output_dir / "calibration.yaml"
    data_yaml.write_text(
        f"path: {output_dir.resolve().as_posix()}\ntrain: images/val\nval: images/val\nnames:\n  0: person\n",
        encoding="utf-8",
    )
    manifest = {
        "purpose": "INT8 activation calibration only; this dataset is not labeled for accuracy evaluation.",
        "source_video": str(video_path.resolve()),
        "source_video_sha256": sha256(video_path),
        "frame_count": frame_count,
        "fps": fps,
        "selected_frames": records,
    }
    manifest_path = output_dir / "calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="从视频生成 OpenVINO INT8 校准图像集")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()
    if args.count < 2:
        raise SystemExit("--count 必须至少为 2")
    print(build_calibration_set(args.video, args.output, args.count))


if __name__ == "__main__":
    main()
