"""取证截图的留存策略。

只删截图，不删报警记录。原因是两者体量差了三个数量级：本地这台机器上 220 张截图是
102 MB，而对应的 alarm_events.jsonl 只有 700 KB。记录是纯文本索引（表格、详情、事后
追溯都靠它），截图才是会把磁盘吃干净的那部分。所以更早的记录仍然留在表格里，只是
详情对话框里的图会显示「不可用」—— 这比悄悄删掉一整段审计记录要诚实得多。

两条规则各自独立触发，都从最旧的开始删：

* ``days`` —— 早于这个天数的截图。
* ``max_bytes`` —— 目录总体积超过这个上限时，继续从最旧的删到不超为止。

任一为 0 表示那条规则不生效；两条都为 0 表示完全不自动清理。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400.0

# 只碰这些后缀。目录里要是混进了别的文件（说明、脚本、别处拷来的东西），
# 一律不动 —— 自动清理最忌讳的就是删掉自己没生成过的东西。
SCREENSHOT_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class RetentionPolicy:
    days: int = 0
    max_bytes: int = 0

    def __post_init__(self) -> None:
        # 兜一层：负数无从解释，一律当「不设这条规则」。
        object.__setattr__(self, "days", max(0, int(self.days)))
        object.__setattr__(self, "max_bytes", max(0, int(self.max_bytes)))

    @property
    def enabled(self) -> bool:
        return self.days > 0 or self.max_bytes > 0


@dataclass(frozen=True)
class RetentionResult:
    removed: int = 0
    removed_bytes: int = 0
    kept: int = 0
    kept_bytes: int = 0
    failed: int = 0

    @property
    def changed(self) -> bool:
        return self.removed > 0 or self.failed > 0

    def describe(self) -> str:
        if not self.changed:
            return f"无需清理（保留 {self.kept} 张，{format_size(self.kept_bytes)}）"
        text = f"已清理 {self.removed} 张截图，释放 {format_size(self.removed_bytes)}"
        if self.failed:
            text += f"，{self.failed} 张删除失败（可能被占用）"
        text += f"，保留 {self.kept} 张（{format_size(self.kept_bytes)}）"
        return text


def format_size(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size} B"


def _screenshots(directory: Path) -> list[tuple[float, int, Path]]:
    """列出目录里的截图，按修改时间从旧到新。

    修改时间就是报警发生的时刻，所以「最旧」直接对应「最早的那次闯入」。
    """
    entries: list[tuple[float, int, Path]] = []
    if not directory.is_dir():
        # 还没报过警（或者刚清空过）时目录根本不存在，那是正常状态，不是错误。
        return entries
    try:
        candidates = list(directory.iterdir())
    except OSError as error:
        logger.warning("无法列出截图目录 %s: %s", directory, error)
        return entries
    for path in candidates:
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix.lower() not in SCREENSHOT_SUFFIXES:
            continue
        try:
            stat = path.stat()
        except OSError as error:
            logger.warning("无法读取截图信息 %s: %s", path, error)
            continue
        entries.append((stat.st_mtime, stat.st_size, path))
    entries.sort(key=lambda entry: entry[0])
    return entries


def prune_screenshots(
    directory: Path,
    policy: RetentionPolicy,
    *,
    now: float | None = None,
) -> RetentionResult:
    """按策略清理目录里的截图，返回这次清掉了多少。

    ``now`` 只为测试可注入；正常调用走系统时间。
    """
    if not policy.enabled:
        return RetentionResult()
    if not directory.is_dir():
        return RetentionResult()

    entries = _screenshots(directory)
    if not entries:
        return RetentionResult()

    current_time = time.time() if now is None else now
    cutoff = (
        current_time - policy.days * SECONDS_PER_DAY if policy.days > 0 else None
    )

    survivors: list[tuple[float, int, Path]] = []
    doomed: list[tuple[float, int, Path]] = []
    for entry in entries:
        if cutoff is not None and entry[0] < cutoff:
            doomed.append(entry)
        else:
            survivors.append(entry)

    if policy.max_bytes > 0:
        total = sum(size for _, size, _ in survivors)
        index = 0
        # survivors 仍按时间升序，所以这里也是「从最旧的开始删」。
        while total > policy.max_bytes and index < len(survivors):
            size = survivors[index][1]
            doomed.append(survivors[index])
            total -= size
            index += 1
        survivors = survivors[index:]

    removed = 0
    removed_bytes = 0
    failed = 0
    for _, size, path in doomed:
        try:
            path.unlink()
        except OSError as error:
            # 文件被别的程序占用（看图工具、杀软扫描）是常态，不该中断这一轮清理，
            # 更不该让调用方炸掉。
            failed += 1
            logger.warning("无法删除过期截图 %s: %s", path, error)
            continue
        removed += 1
        removed_bytes += size

    result = RetentionResult(
        removed=removed,
        removed_bytes=removed_bytes,
        kept=len(survivors),
        kept_bytes=sum(size for _, size, _ in survivors),
        failed=failed,
    )
    if result.changed:
        logger.info("取证留存清理: %s", result.describe())
    return result


def directory_usage(directory: Path) -> tuple[int, int]:
    """返回 (张数, 字节数)，供界面显示当前占用。"""
    entries = _screenshots(directory)
    return len(entries), sum(size for _, size, _ in entries)
