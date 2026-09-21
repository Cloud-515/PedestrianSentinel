from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from models import ZoneProfile

logger = logging.getLogger(__name__)


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
        """列出目录里读得出来的配置组。

        一个读不出来的文件只跳过它自己。「列配置组」是启动路径上的一环
        （``MainWindow._load_controls`` 会刷新列表），让一个坏文件把整个程序挡在门外
        是不可接受的 —— 配置文件是给人手改的文本，现场改坏一处完全可能。坏文件仍留在
        磁盘上（不改名、不删），日志里有它的名字。
        """
        profiles: list[ProfileSummary] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                profile = self.load(path.stem)
            except ValueError as error:
                logger.warning("跳过读不出来的配置组 %s：%s", path.name, error)
                continue
            profiles.append(ProfileSummary(profile.name, len(profile.zones), profile.updated_at))
        return profiles

    def load(self, name: str) -> ZoneProfile:
        path = self._path_for(name)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            # 路径与原因都带上：文件被占用、权限不足、目录整个不见了，修法各不相同。
            raise ValueError(f"无法读取配置组 {name}：{exc}") from exc
        except UnicodeDecodeError as exc:
            # 不是 UTF-8 文本（改存成别的编码、或被同步工具写坏了）。它本身也是 ValueError，
            # 但单独包一层，让报错里出现配置组的名字而不是一串 codec 细节。
            raise ValueError(f"无法读取配置组 {name}：文件不是 UTF-8 文本") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"无法读取配置组 {name}：{exc}") from exc
        return ZoneProfile.from_dict(data)

    def quarantine(self, name: str) -> Path | None:
        """把读不出来的配置组改名留档，返回它的新路径。

        与 ``ConfigStore.quarantine`` 同一套做法（后缀 ``.bad-<时间戳>``，``*.json`` 的
        遍历不会捡起它）：坏文件不能原地留着，否则每次启动都在同一个地方失败；也不该
        直接删掉 —— 用户很可能只是改错了一个字符，区域坐标还得能捞回来。
        """
        path = self._path_for(name)
        if not path.exists():
            return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = path.with_name(f"{path.name}.bad-{stamp}")
        try:
            os.replace(path, target)
        except OSError:
            logger.exception("无法隔离损坏的配置组 %s", path)
            return None
        logger.warning("损坏的配置组已备份为 %s", target)
        return target

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
        """这个配置组能不能读。名字非法时按「不存在」处理，而不是抛异常。

        调用方（新建、另存为、启动时找回当前配置组）都拿它当一次纯粹的询问用；名字
        是 config.json 里的一个字符串，也能从输入框里来，带个斜杠或冒号完全不奇怪。
        抛异常的话，问一句「有没有」就能把窗口建不起来（``_load_controls`` 里那一问
        跑在 ``__init__`` 内）。真正写盘时 ``save`` 仍会照常拒绝非法名字。
        """
        if not self.is_valid_name(name):
            return False
        return self._path_for(name).exists()

    def is_valid_name(self, name: str) -> bool:
        """这个名字能不能当作配置组文件名。

        与 ``exists`` 的区别在「文件不在」和「名字根本不可能对应文件」：前者可以靠
        重新保存长回来，后者永远不会 —— 启动时读 config.json 拿到一个带斜杠的名字，
        摘掉指针比留着一条点不动的条目好。
        """
        try:
            self._validate_name(name)
        except ValueError:
            return False
        return True

    def _path_for(self, name: str) -> Path:
        self._validate_name(name)
        return self.directory / f"{name}.json"

    def _validate_name(self, name: str) -> None:
        if not name or name in {".", ".."} or "/" in name or "\\" in name or self._invalid_name.search(name):
            raise ValueError("配置组名称包含非法字符")
