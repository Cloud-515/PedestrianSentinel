"""统一的路径解析：源码运行与 PyInstaller 冻结运行共用同一套逻辑。

单目录（onedir）发布版把只读资源与可写数据都放在 exe 同级，而不是 ``_internal/``
里面 —— 这样模型和告警音频可以直接替换。冻结后模块的 ``__file__`` 指向
``_internal/``，所以任何 ``Path(__file__).parent`` 都算不出正确的程序根目录，
统一改走这里的 :data:`APP_DIR`。

- :data:`APP_DIR` 程序根目录。冻结后是 exe 所在目录，源码模式是项目根目录。
- :func:`resource` 只读资源查找，按候选相对路径依次尝试，因此分组式布局
  （``models/`` + ``assets/``）与扁平式布局（资源直接放在根部）都能识别。
- :func:`data` 可写数据路径。程序目录可写时就地写（绿色便携），只读时回落到
  ``%LOCALAPPDATA%``。
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

APP_NAME = "行人警戒区域监控"

IS_FROZEN = bool(getattr(sys, "frozen", False))


def _resolve_app_dir() -> Path:
    if IS_FROZEN:
        # 单目录版：sys.executable 在发布目录根部，_internal/ 是它的子目录。
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_DIR = _resolve_app_dir()


def _is_writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".write-probe-", suffix=".tmp"):
            pass
    except OSError:
        return False
    return True


def _fallback_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    return (Path(base) if base else Path.home()) / APP_NAME


@lru_cache(maxsize=1)
def data_dir() -> Path:
    """返回可写数据根目录（config.json / events / profiles / logs 的落点）。"""
    if _is_writable(APP_DIR):
        return APP_DIR
    fallback = _fallback_data_dir()
    # 装到 C:\Program Files 之类只读位置时，让「配置存哪了」有迹可循。
    logger.warning("程序目录不可写(%s)，配置与取证数据改写入 %s", APP_DIR, fallback)
    return fallback


def data(*parts: str) -> Path:
    """拼出一个可写数据路径。目录创建由各调用方自行 mkdir 负责。"""
    return data_dir().joinpath(*parts)


def resource(*candidates: str) -> Path:
    """在 :data:`APP_DIR` 下按候选相对路径依次查找，返回第一个存在的。

    全都不存在时返回**第一个**候选的绝对路径，让调用方的错误信息里出现推荐位置
    而不是最后的兜底位置。
    """
    if not candidates:
        raise ValueError("resource() 至少需要一个候选路径")
    for candidate in candidates:
        path = APP_DIR.joinpath(candidate)
        if path.exists():
            return path
    return APP_DIR.joinpath(candidates[0])
