from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import psutil

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from detection_engine import DetectionEngine
from inference_profiles import InferencePolicy, resolve_inference_policy
from models import ZoneDefinition


def profile(name: str) -> InferencePolicy:
    if name == "cpu_pt_default":
        return InferencePolicy(str(ROOT / "yolo11n.pt"), "cpu", label="普通 CPU 模式")
    resolution = resolve_inference_policy(str(ROOT / "yolo11n.pt"), "cpu", True)
    if not resolution.available:
        raise RuntimeError(f"低功耗模型不可用：{resolution.unavailable_reason}")
    return resolution.policy


def run_trial(
    video_path: Path,
    policy: InferencePolicy,
    *,
    render: bool = False,
    zones: list | None = None,
) -> dict[str, float | int | str]:
    """跑一遍完整视频，返回这一轮的耗时与占用。

    ``render`` 打开时走**带标注**的那条路径（线上实际跑的就是它）：画检测框、画警戒区、
    把行人像素贴回区域之上。默认关闭是为了让不同预设之间的对比只反映推理与跟踪的差异；
    要看渲染值多少成本，就用同一个预设跑两次、只切这个开关。
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    process = psutil.Process()
    process.cpu_percent(None)
    system_cpu_samples: list[float] = []
    latencies: list[float] = []
    detector_calls = 0
    model_started = time.perf_counter()
    engine = DetectionEngine(policy.model_path, zones or [], policy.device, policy)
    model_load_seconds = time.perf_counter() - model_started
    cpu_before = process.cpu_times()
    rss_before = process.memory_info().rss
    started = time.perf_counter()
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_started = time.perf_counter()
        engine.process(frame, frames / 30.0, str(video_path), render=render)
        latencies.append((time.perf_counter() - frame_started) * 1000)
        frames += 1
        if (engine._processed_frames - 1) % policy.detector_interval == 0:
            detector_calls += 1
        if frames % 20 == 0:
            system_cpu_samples.append(psutil.cpu_percent(interval=None))
    elapsed = time.perf_counter() - started
    cpu_after = process.cpu_times()
    capture.release()
    return {
        "profile": policy.label,
        "render": render,
        "zones": len(zones or []),
        "frames": frames,
        "detector_calls": detector_calls,
        "wall_seconds": round(elapsed, 4),
        "fps": round(frames / elapsed, 3),
        "detector_fps": round(detector_calls / elapsed, 3),
        "model_load_seconds": round(model_load_seconds, 4),
        "mean_frame_ms": round(statistics.fmean(latencies), 3),
        "p95_frame_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 3),
        "process_cpu_seconds": round(
            (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system), 4
        ),
        "process_cpu_percent": round(
            ((cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system))
            / elapsed
            / (os.cpu_count() or 1)
            * 100,
            2,
        ),
        "system_cpu_percent_mean": round(statistics.fmean(system_cpu_samples), 2) if system_cpu_samples else 0.0,
        "rss_delta_mib": round((process.memory_info().rss - rss_before) / 1024 / 1024, 2),
        "rss_mib": round(process.memory_info().rss / 1024 / 1024, 2),
        "threads": process.num_threads(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="比较普通 CPU 与 OpenVINO INT8 低功耗预设")
    parser.add_argument("--video", type=Path, default=ROOT / "8月11日-1.mp4")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--render",
        action="store_true",
        help="带上标注渲染（线上实际跑的就是这条路径）；不给则只测解码+检测+跟踪",
    )
    parser.add_argument(
        "--zones",
        default="",
        help="渲染时用的警戒区顶点：x1,y1,x2,y2,…（不给则不画区域，只画检测框）",
    )
    args = parser.parse_args()
    if args.trials < 1:
        raise SystemExit("--trials 必须至少为 1")

    zones: list[ZoneDefinition] = []
    if args.zones:
        numbers = [float(item) for item in args.zones.split(",")]
        if len(numbers) % 2 != 0 or len(numbers) < 6:
            raise SystemExit("--zones 需要成对的坐标，至少 3 个点")
        zones = [
            ZoneDefinition(
                name="基准区域",
                polygon=[
                    [numbers[index], numbers[index + 1]]
                    for index in range(0, len(numbers), 2)
                ],
                closed=True,
            )
        ]

    output_dir = ROOT / "benchmarks" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True)
    results = []
    profiles = [profile("cpu_pt_default"), profile("cpu_openvino_int8_low_power")]
    for trial in range(args.trials + 1):
        order = profiles if trial % 2 == 0 else list(reversed(profiles))
        for current in order:
            result = run_trial(
                args.video, current, render=args.render, zones=zones
            )
            result["trial"] = trial
            result["warmup"] = trial == 0
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))

    with (output_dir / "trials.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(output_dir)


if __name__ == "__main__":
    main()
