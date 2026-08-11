from __future__ import annotations

import json
from pathlib import Path

from models import AppConfig


class ConfigStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> AppConfig:
        if not self.path.exists():
            return AppConfig()
        with self.path.open("r", encoding="utf-8") as config_file:
            return AppConfig.from_dict(json.load(config_file))

    def save(self, config: AppConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as config_file:
            json.dump(config.to_dict(), config_file, ensure_ascii=False, indent=2)
