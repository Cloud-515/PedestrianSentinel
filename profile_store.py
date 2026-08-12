from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from models import ZoneProfile


@dataclass(frozen=True)
class ProfileSummary:
    name: str
    zone_count: int
    updated_at: str


class ProfileStore:
    _invalid_name = re.compile(r'[<>:"/\\|?*]')

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> list[ProfileSummary]:
        profiles: list[ProfileSummary] = []
        for path in sorted(self.directory.glob("*.json")):
            profile = self.load(path.stem)
            profiles.append(ProfileSummary(profile.name, len(profile.zones), profile.updated_at))
        return profiles

    def load(self, name: str) -> ZoneProfile:
        path = self._path_for(name)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取配置组: {name}") from exc
        return ZoneProfile.from_dict(data)

    def save(self, profile: ZoneProfile) -> None:
        self._validate_name(profile.name)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        created_at = profile.created_at or now
        updated = ZoneProfile(
            name=profile.name,
            zones=profile.zones,
            description=profile.description,
            created_at=created_at,
            updated_at=now,
            version=profile.version,
        )
        target = self._path_for(updated.name)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(updated.to_dict(), stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temp_name, target)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    def delete(self, name: str) -> None:
        path = self._path_for(name)
        try:
            path.unlink()
        except FileNotFoundError as exc:
            raise ValueError(f"配置组不存在: {name}") from exc

    def exists(self, name: str) -> bool:
        return self._path_for(name).exists()

    def _path_for(self, name: str) -> Path:
        self._validate_name(name)
        return self.directory / f"{name}.json"

    def _validate_name(self, name: str) -> None:
        if not name or name in {".", ".."} or "/" in name or "\\" in name or self._invalid_name.search(name):
            raise ValueError("配置组名称包含非法字符")
