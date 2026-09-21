"""发布资源的哈希清单：生成与校验。

``yolo11n.pt`` 与 ``warming_converted.wav`` 被换掉、拷坏、解压截断时，程序都照样加载，
只是「检测不出人」或「报警没声音」—— 两种故障都不报错，只会让人以为程序坏了。低功耗
用的 OpenVINO 模型一直有 sha256 校验，这两个文件以前没有，所以补了这份清单。

清单必须能被**现场**重新生成：说明书里写明这两个文件可以替换，而替换完就该更新清单
（否则自检会以「资源与清单不一致」判失败）。所以生成逻辑放在这里，源码树的
``tools/write_asset_manifest.py`` 与打包版的 ``--write-asset-manifest`` 共用同一份 ——
两边各写一套迟早会漂移成两份不同的清单。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import app_paths

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1

# 逻辑名 + 在发布目录里的候选相对路径。分组式（models\ / assets\）优先，扁平式
# （直接放在 exe 同级）作为兼容回退 —— 与运行期 app_paths.resource() 的查找顺序一致，
# 所以源码模式（文件在项目根）与打包模式都能解析到同一个文件。
ASSETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("yolo11n.pt", ("models/yolo11n.pt", "yolo11n.pt")),
    (
        "warming_converted.wav",
        ("assets/warming_converted.wav", "warming_converted.wav"),
    ),
)


@dataclass(frozen=True)
class AssetCheck:
    name: str
    path: Path | None
    ok: bool
    detail: str
    problem: str = ""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path() -> Path:
    """清单当前所在的位置（也是 ``--write-asset-manifest`` 会写回去的位置）。

    打包版里清单跟着资源放在 ``assets\\``，所以现场重新生成时要覆盖**那一份** ——
    写到别处会留下旧清单，查找顺序仍然会读到旧的。
    """
    return app_paths.resource("assets/asset_manifest.json", "asset_manifest.json")


def build_manifest() -> dict[str, object]:
    """按当前资源算出清单内容。缺资源时抛 FileNotFoundError。"""
    entries = []
    for name, candidates in ASSETS:
        path = app_paths.resource(*candidates)
        if not path.is_file():
            raise FileNotFoundError(f"缺少发布资源 {name}（期望位置 {path}）")
        entries.append(
            {
                "name": name,
                "candidates": list(candidates),
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
            }
        )
    return {"version": MANIFEST_VERSION, "assets": entries}


def manifest_text() -> str:
    return json.dumps(build_manifest(), ensure_ascii=False, indent=2) + "\n"


def write_manifest(
    target: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    """把清单写到 ``target``（默认写到当前正在用的那个位置），返回 (路径, 清单内容)。"""
    destination = target or manifest_path()
    manifest = build_manifest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("已写入资源清单 %s", destination)
    return destination, manifest


def describe(manifest: dict[str, object]) -> list[str]:
    """把清单内容摊成人能看的一行行，供命令行输出。"""
    lines = []
    for entry in manifest.get("assets", []):
        if not isinstance(entry, dict):
            continue
        lines.append(
            f"{str(entry.get('name', '?')):<24} "
            f"{int(entry.get('bytes', 0)):>10} 字节  "
            f"{str(entry.get('sha256', ''))[:16]}…"
        )
    return lines


def verify_manifest() -> tuple[list[AssetCheck], list[str]]:
    """按清单逐项校验，返回 (每一项的结果, 失败说明)。

    只读，不写；自检与构建脚本都调它。
    """
    path = manifest_path()
    if not path.is_file():
        return [], [f"缺少资源清单: {path}（用 --write-asset-manifest 重新生成）"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        entries = manifest["assets"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [], [f"资源清单无法解析（{path}）: {error!r}"]

    checks: list[AssetCheck] = []
    failures: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "?"))
        candidates = [str(item) for item in entry.get("candidates", [name])]
        asset = app_paths.resource(*candidates)
        if not asset.is_file():
            problem = f"发布资源缺失: {name}（期望位置 {asset}）"
            checks.append(AssetCheck(name, asset, False, f"FAIL 不存在 {asset}", problem))
            failures.append(problem)
            continue
        digest = sha256_of(asset)
        expected = str(entry.get("sha256", ""))
        size = asset.stat().st_size
        if digest != expected:
            problem = (
                f"发布资源与清单不一致: {name}（{asset}）"
                "。替换过资源的话，运行「行人警戒区域监控.exe --write-asset-manifest」"
                "重新生成清单；没替换过则说明文件已损坏"
            )
            checks.append(
                AssetCheck(name, asset, False, f"FAIL 哈希不符 {digest[:16]}…", problem)
            )
            failures.append(problem)
            continue
        checks.append(
            AssetCheck(
                name,
                asset,
                True,
                f"ok | {size} 字节 | sha256 {digest[:16]}…",
            )
        )
    return checks, failures
