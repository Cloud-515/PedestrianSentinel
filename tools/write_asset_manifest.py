"""生成发布资源的哈希清单 ``asset_manifest.json``。

为什么要它：低功耗的 OpenVINO 模型早就有 ``model_manifest.json`` 逐文件校验 sha256，
而 ``yolo11n.pt`` 和 ``warming_converted.wav`` 什么都没有 —— 它们被换掉、拷坏、或者
解压时截断了，程序照样加载，只是「检测不出人」或者「报警没声音」。这两种故障都不会
报错，只会让人以为程序坏了。

什么时候要重新跑：**替换了发布资源之后**。这两个文件在说明里都是允许用户替换的
（换自己的 YOLO 权重、换自己的告警音），替换完请重新生成清单，否则 ``--selftest``
会以「资源与清单不一致」判失败 —— 那正是它的用途：让「我换了个权重」和「权重被
悄悄改坏了」这两件事都必须被看见。

用法::

    python tools\\write_asset_manifest.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "asset_manifest.json"

# 清单里的每一项：逻辑名、源码树里的位置、以及在发布目录里的候选相对路径
# （分组式 models\ / assets\ 优先，扁平式直接放根部作为兼容回退 —— 与运行期
# app_paths.resource() 的查找顺序一致）。
ASSETS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("yolo11n.pt", "yolo11n.pt", ("models/yolo11n.pt", "yolo11n.pt")),
    (
        "warming_converted.wav",
        "warming_converted.wav",
        ("assets/warming_converted.wav", "warming_converted.wav"),
    ),
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest() -> dict:
    entries = []
    for name, source, candidates in ASSETS:
        path = ROOT / source
        if not path.is_file():
            raise SystemExit(f"缺少发布资源 {path}，无法生成清单")
        entries.append(
            {
                "name": name,
                "candidates": list(candidates),
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
            }
        )
    return {"version": 1, "assets": entries}


def main() -> int:
    parser = argparse.ArgumentParser(description="生成发布资源哈希清单")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验现有清单是否与资源一致，不写入（用于构建前的快速检查）",
    )
    arguments = parser.parse_args()

    manifest = build_manifest()
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"

    if arguments.check:
        if not MANIFEST_PATH.is_file():
            print(f"缺少清单 {MANIFEST_PATH}，请先运行本脚本生成", file=sys.stderr)
            return 1
        if MANIFEST_PATH.read_text(encoding="utf-8") != text:
            print("清单与资源不一致，请重新生成（资源被替换或损坏）", file=sys.stderr)
            return 1
        print("清单与资源一致")
        return 0

    MANIFEST_PATH.write_text(text, encoding="utf-8")
    for entry in manifest["assets"]:
        print(f"{entry['name']:<24} {entry['bytes']:>10} 字节  {entry['sha256'][:16]}…")
    print(f"已写入 {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
