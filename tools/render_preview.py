"""把警戒区的渲染效果离线导出成 PNG，用来对着图调视觉效果。

为什么要有它：区域的观感（贴地感、透明度、被行人遮挡）只能靠眼睛判断，而为了看一眼
就去开一次 GUI、开一次视频、等模型加载，改一轮要几分钟。这个脚本直接拿一段视频的
某一帧跑一遍真实检测与标注，把「引擎烧进帧里的图」与「控件上最终显示的样子」各存
一张 PNG —— 改一行跑一次，两秒钟看到结果。

用法::

    python tools\\render_preview.py --frame 12 --out preview
    python tools\\render_preview.py --video other.mp4 --frame 0 --zones 100,100,900,100,900,700,100,700

默认读 config.json 里当前生效的警戒区域；`--zones` 可以直接给一个矩形顶点列表，
用来构造「人正好在区域边界上」这种需要凑的画面。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from detection_engine import DetectionEngine  # noqa: E402
from models import AppConfig, ZoneDefinition  # noqa: E402

# 渲染前先喂几帧给追踪器，否则首帧的框全是"未确认"，看不出真实效果。
WARMUP_FRAMES = 6


def load_zones(config_path: Path, override: str | None) -> list[ZoneDefinition]:
    if override:
        numbers = [float(item) for item in override.split(",")]
        if len(numbers) % 2 != 0 or len(numbers) < 6:
            raise SystemExit("--zones 需要成对的坐标，至少 3 个点：x1,y1,x2,y2,x3,y3")
        polygon = [
            [numbers[index], numbers[index + 1]] for index in range(0, len(numbers), 2)
        ]
        return [ZoneDefinition(name="预览区域", polygon=polygon, closed=True, dwell_seconds=1.0)]

    config = AppConfig.from_dict(
        __import__("json").loads(config_path.read_text(encoding="utf-8"))
    )
    if not config.zones:
        raise SystemExit(f"{config_path} 里没有警戒区域，用 --zones 指定一个")
    return config.zones


def annotate_frame(
    engine: DetectionEngine,
    frame: np.ndarray,
    zones: list[ZoneDefinition],
    *,
    warmup_frames: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, dict]:
    """让引擎按真实路径处理一帧（含检测与标注）。

    ``warmup_frames`` 会先喂给引擎但丢弃结果：ByteTrack 在全新追踪器的第一帧会把所有
    目标标成未确认（track_id = -1），那一帧既不会有目标被判进区域，框也全是绿色。
    直接把首帧当成效果图会让人误以为"人在区域里却没被判进去"。
    """
    engine.update_zones(zones)
    for warmup in warmup_frames or []:
        engine.process(warmup, 0.0, "preview", "monitor")
    annotated, _ = engine.process(frame, 0.0, "preview", "monitor")
    return annotated, engine.active_sessions


def render_widget(annotated: np.ndarray, zones: list[ZoneDefinition]) -> np.ndarray | None:
    """再走一遍控件的显示层（区域在这里也会被画一次），得到用户最终看到的样子。"""
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication

        from video_widget import VideoWidget
    except ImportError:
        return None

    app = QApplication.instance() or QApplication([])
    widget = VideoWidget()
    widget.resize(annotated.shape[1], annotated.shape[0])
    widget.set_zones(zones)
    widget.set_active_zone(zones[0] if zones else None)
    widget.set_frame(annotated)
    pixmap = widget.grab()
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    width, height = image.width(), image.height()
    buffer = image.constBits()
    array = np.frombuffer(buffer, dtype=np.uint8).reshape((height, image.bytesPerLine()))
    array = array[:, : width * 3].reshape((height, width, 3))
    del app
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def main() -> int:
    parser = argparse.ArgumentParser(description="导出警戒区渲染效果预览")
    parser.add_argument("--video", default="8月11日-1.mp4", help="视频素材")
    parser.add_argument("--frame", type=int, default=12, help="取第几帧（从 0 开始）")
    parser.add_argument("--config", default="config.json", help="读哪个配置里的区域")
    parser.add_argument("--zones", default=None, help="直接给顶点：x1,y1,x2,y2,…")
    parser.add_argument("--weights", default="yolo11n.pt", help="检测权重")
    parser.add_argument("--device", default="cpu", help="推理设备")
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "preview" / "preview"),
        help="输出文件名前缀（默认落在已被忽略的 artifacts\\preview\\ 下）",
    )
    arguments = parser.parse_args()

    video = Path(arguments.video)
    if not video.is_file():
        raise SystemExit(f"找不到视频 {video}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"打不开视频 {video}")
    warmup: list[np.ndarray] = []
    try:
        # 目标帧之前先读几帧当预热（见 annotate_frame 的说明）。
        for _ in range(max(0, arguments.frame - WARMUP_FRAMES)):
            capture.read()
        for _ in range(min(WARMUP_FRAMES, arguments.frame)):
            ok, warmup_frame = capture.read()
            if not ok:
                break
            warmup.append(warmup_frame)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise SystemExit(f"读不到第 {arguments.frame} 帧")

    zones = load_zones(Path(arguments.config), arguments.zones)
    engine = DetectionEngine(arguments.weights, zones, arguments.device)
    annotated, sessions = annotate_frame(engine, frame, zones, warmup_frames=warmup)

    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    engine_out = Path(f"{arguments.out}_engine.png")
    # cv2.imwrite 失败只返回 False、不抛异常；调视觉时静默失败最费时间，所以要报出来。
    if not cv2.imwrite(str(engine_out), annotated):
        raise SystemExit(f"写不出 {engine_out}（路径里有非 ASCII 字符时 cv2 可能失败）")
    print(f"引擎标注帧: {engine_out}  ({annotated.shape[1]}x{annotated.shape[0]})")
    print(f"区域: {[zone.name for zone in zones]}")
    print(f"检出并进入区域的目标数: {len(sessions)}")

    widget_image = render_widget(annotated, zones)
    if widget_image is None:
        print("未渲染控件层（没有可用的 Qt 环境）")
        return 0
    widget_out = Path(f"{arguments.out}_widget.png")
    if not cv2.imwrite(str(widget_out), widget_image):
        raise SystemExit(f"写不出 {widget_out}")
    print(f"控件合成帧: {widget_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
