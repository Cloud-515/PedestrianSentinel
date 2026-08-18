from __future__ import annotations

import os
from pathlib import Path

MAX_HISTORY_ENTRIES = 10


def normalize_camera_source(value: str) -> str:
    return value.strip()


def normalize_file_source(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    return str(Path(value).expanduser().resolve(strict=False))


def add_history_entry(history: list[str], value: str, *, file_source: bool = False) -> list[str]:
    normalized = normalize_file_source(value) if file_source else normalize_camera_source(value)
    if not normalized:
        return [entry for entry in history if isinstance(entry, str) and entry.strip()]
    # 文件路径在 Windows 上不区分大小写，按规范键去重。
    key = os.path.normcase(normalized) if file_source else normalized
    result = [normalized]
    seen_keys = {key}
    for entry in history:
        if not isinstance(entry, str):
            continue
        entry = entry.strip()
        if not entry:
            continue
        candidate = normalize_file_source(entry) if file_source else entry
        candidate_key = os.path.normcase(candidate) if file_source else candidate
        if candidate_key not in seen_keys:
            result.append(candidate)
            seen_keys.add(candidate_key)
    return result[:MAX_HISTORY_ENTRIES]


def load_history(value: object, *, file_source: bool = False) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            continue
        normalized = normalize_file_source(entry) if file_source else normalize_camera_source(entry)
        if normalized:
            key = os.path.normcase(normalized) if file_source else normalized
            if all((os.path.normcase(item) if file_source else item) != key for item in result):
                result.append(normalized)
        if len(result) >= MAX_HISTORY_ENTRIES:
            break
    return result
