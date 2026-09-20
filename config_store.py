from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

from models import AppConfig

logger = logging.getLogger(__name__)


class ConfigStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> AppConfig:
        if not self.path.exists():
            return AppConfig()
        with self.path.open("r", encoding="utf-8") as config_file:
            data = json.load(config_file)
        if not isinstance(data, dict):
            # 根节点不是对象时 AppConfig.from_dict 里的每个 data.get 都会变成
            # AttributeError —— 那既不是 ValueError 也不是 TypeError，会从调用方的
            # except 子句底下溜过去，把启动整个带崩。在这里挡成 ValueError。
            raise ValueError(
                f"配置文件根节点应为对象，实际是 {type(data).__name__}"
            )
        return AppConfig.from_dict(data)

    def quarantine(self) -> Path | None:
        """把读不了的配置文件改名留档，返回它的新路径。

        坏文件不能原地留着：程序每次启动都读它、每次都以失败告终。也不该直接删掉 ——
        里面很可能就是用户想要的区域坐标，只是某一处改坏了。改名既能让程序起来，
        又给人留了手工找回的余地。
        """
        if not self.path.exists():
            return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = self.path.with_name(f"{self.path.name}.bad-{stamp}")
        try:
            os.replace(self.path, target)
        except OSError:
            logger.exception("无法隔离损坏的配置文件 %s", self.path)
            return None
        logger.warning("损坏的配置文件已备份为 %s", target)
        return target

    def save(self, config: AppConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as config_file:
                temporary_path = Path(config_file.name)
                json.dump(config.to_dict(), config_file, ensure_ascii=False, indent=2)
                config_file.flush()
                os.fsync(config_file.fileno())
            # 同目录替换可避免写入中断时损坏原配置。
            os.replace(temporary_path, self.path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
