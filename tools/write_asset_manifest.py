"""生成发布资源的哈希清单 ``asset_manifest.json``（源码树里用）。

为什么要它：低功耗的 OpenVINO 模型早就有 ``model_manifest.json`` 逐文件校验 sha256，
而 ``yolo11n.pt`` 和 ``warming_converted.wav`` 什么都没有 —— 它们被换掉、拷坏、或者
解压时截断了，程序照样加载，只是「检测不出人」或者「报警没声音」。这两种故障都不会
报错，只会让人以为程序坏了。

什么时候要重新跑：**替换了发布资源之后**。这两个文件在说明里都是允许用户替换的
（换自己的 YOLO 权重、换自己的告警音），替换完请重新生成清单，否则 ``--selftest``
会以「资源与清单不一致」判失败 —— 那正是它的用途：让「我换了个权重」和「权重被
悄悄改坏了」这两件事都必须被看见。

生成逻辑本身在 ``asset_manifest.py`` 里，与打包版的
``行人警戒区域监控.exe --write-asset-manifest`` 共用同一份 —— 现场没有 Python 环境，
``tools/`` 也不随发布包一起发，所以那件事必须由 exe 自己能做。

用法::

    python tools\\write_asset_manifest.py
    python tools\\write_asset_manifest.py --check    # 只校验，不写
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import asset_manifest  # noqa: E402 - 必须先把它所在的项目根加进 sys.path

# 源码树里清单就放在项目根（打包版跟着资源放在 assets\ 下）。
MANIFEST_PATH = ROOT / "asset_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成发布资源哈希清单")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验现有清单是否与资源一致，不写入（用于构建前的快速检查）",
    )
    arguments = parser.parse_args()

    try:
        if arguments.check:
            expected = asset_manifest.manifest_text()
            if not MANIFEST_PATH.is_file():
                print(f"缺少清单 {MANIFEST_PATH}，请先运行本脚本生成", file=sys.stderr)
                return 1
            if MANIFEST_PATH.read_text(encoding="utf-8") != expected:
                print("清单与资源不一致，请重新生成（资源被替换或损坏）", file=sys.stderr)
                return 1
            print("清单与资源一致")
            return 0

        target, manifest = asset_manifest.write_manifest(MANIFEST_PATH)
    except FileNotFoundError as error:
        print(f"生成清单失败：{error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"写入清单失败：{error}", file=sys.stderr)
        return 1

    for line in asset_manifest.describe(manifest):
        print(line)
    print(f"已写入 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
