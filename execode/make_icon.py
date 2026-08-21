"""生成 execode/app.ico —— exe 的程序图标。

为什么是脚本而不是直接扔一个 .ico 进仓库：图标是构建产物，源头应该可读、可改。
想换配色或改造型，改下面的常量重跑一次就行，不用去找作图软件。

    .venv-build\\Scripts\\python.exe execode\\make_icon.py

**不接进 build.ps1**：app.ico 是签入的资产，不该每次构建都重新生成 ——
那样图标会随本脚本的任何改动悄悄变样，而 Windows 的图标缓存又出了名地黏。

造型：深色圆角底 + 琥珀色透视警戒区四边形 + 站在区内的白色行人。
16×16 下细节必然糊成一团，所以几何全用归一化坐标按目标尺寸重新绘制（而不是把
一张大图缩下来），描边宽度因此始终 >= 1 px。同理，凡是「大尺寸下才看得见」的
细节一律不画 —— 试过在腿之间留一道缝，结果每档都读成裂纹而非两条腿。

PIL 的 ICO 写入器对**所有**尺寸都用 PNG 压缩（16×16/32bpp 约 739 B，未压缩的
DIB 要 1128 B）。这是有意接受的：ICO 里的 PNG 条目从 Vista 起被 Windows 全尺寸
支持，PyInstaller 也只是把图像块整段搬进 RT_ICON、不重新编码；本项目的下限是
Win10（PySide6 6.11）。真要退回「小尺寸用 BMP、256 用 PNG」的老式布局，得自己
拼 ICO 容器 —— PIL 没有按档指定格式的接口。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUTPUT = Path(__file__).resolve().parent / "app.ico"

# Windows 会按场景挑用这些尺寸：16 托盘/标题栏，32 桌面小图标，48 资源管理器
# 「中等图标」，256 「超大图标」与属性面板。缺哪个 Windows 就自己缩，效果更差。
SIZES = (16, 24, 32, 48, 64, 128, 256)

BACKGROUND = (22, 32, 46, 255)      # 深板岩蓝：满幅铺底，任务栏上是个实心块，不虚
ZONE_LINE = (255, 176, 32, 255)     # 琥珀色：警戒区边框
FIGURE = (245, 249, 255, 255)       # 近白：行人，和琥珀区形成最大反差
_ZONE_TINT = 0.45                   # 区域填充 = 琥珀按此比例压在底色上


def _blend(fg: tuple[int, ...], bg: tuple[int, ...], alpha: float) -> tuple[int, int, int, int]:
    """预先把半透明色算成实色。

    **不要**改回给 fill 传带 alpha 的颜色：PIL 的 draw.* 是直接**覆写**像素、
    不做 alpha 合成，那样画出来的是一块 alpha=115 的**半透明洞** —— 浅色背景下
    看着像淡琥珀（所以很容易误判为「成了」），深色任务栏上却会整块发暗。
    """
    return (*(round(f * alpha + b * (1 - alpha)) for f, b in zip(fg[:3], bg[:3])), 255)


ZONE_FILL = _blend(ZONE_LINE, BACKGROUND, _ZONE_TINT)

# 归一化几何（0..1）。警戒区是个上窄下宽的梯形，读起来就是「地面上的一片区域」。
ZONE = ((0.11, 0.830), (0.89, 0.830), (0.69, 0.495), (0.31, 0.495))
HEAD_CENTER, HEAD_RADIUS = (0.5, 0.250), 0.082
# 身体比头窄：梯形里就能在人两侧各留出一段琥珀，16px 下人和区域才不糊成一坨。
BODY = (0.436, 0.352, 0.564, 0.720)
BODY_RADIUS = 0.064
# 腿缝不画。任何尺寸下它都读成「裂纹」而不是「两条腿」，纯减分。


def _render(size: int) -> Image.Image:
    """按目标尺寸的 4 倍作画再缩回去 —— 纯粹为了抗锯齿，几何比例不受影响。"""
    scale = 4
    canvas = size * scale
    image = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    def pt(x: float, y: float) -> tuple[float, float]:
        return (x * canvas, y * canvas)

    # 圆角底。半径给到 0.20 是 Windows 11 图标的观感，不至于像个方补丁。
    draw.rounded_rectangle(
        [0, 0, canvas - 1, canvas - 1],
        radius=0.20 * canvas,
        fill=BACKGROUND,
    )

    # 警戒区。用 polygon 的 outline+width 而不是 line 逐段连 —— 后者在闭合处
    # 首尾两个线帽对不上，256px 下右下角会多出一个小疙瘩。
    # 描边宽度跟着尺寸走，16px 时约 1 px，不会消失也不会糊住内部。
    draw.polygon(
        [pt(x, y) for x, y in ZONE],
        fill=ZONE_FILL,
        outline=ZONE_LINE,
        width=max(scale, round(0.052 * canvas)),
    )

    # 行人：头 + 一整条身体。分成两块画而不是拼火柴人，小尺寸下才立得住。
    hx, hy = HEAD_CENTER
    r = HEAD_RADIUS
    draw.rounded_rectangle(
        [pt(BODY[0], BODY[1]), pt(BODY[2], BODY[3])],
        radius=BODY_RADIUS * canvas,
        fill=FIGURE,
    )
    draw.ellipse([pt(hx - r, hy - r), pt(hx + r, hy + r)], fill=FIGURE)

    return image.resize((size, size), Image.LANCZOS)


def main() -> None:
    frames = [_render(size) for size in SIZES]
    # PIL 的 ICO 写入器只从 append_images 里挑尺寸匹配的帧，所以每一档都得给全。
    frames[-1].save(
        OUTPUT,
        format="ICO",
        sizes=[(s, s) for s in SIZES],
        append_images=frames[:-1],
    )
    print(f"已写出 {OUTPUT}（{OUTPUT.stat().st_size / 1024:.1f} KB，尺寸 {SIZES}）")


if __name__ == "__main__":
    main()
