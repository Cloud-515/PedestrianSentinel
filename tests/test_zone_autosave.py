"""区域改动要落进「当前配置组」，不能只写 config.json。

现场反馈是「新增区域和改动区域后没自动保存」。查下来：config.json 那条路是自动保存
的，但**启动时配置组优先**（_restore_active_profile 用它覆盖 config.json 里的区域）——
所以画完区域直接关窗，改动只躺在 config.json 里，下次启动被配置组盖掉，等于白画。

这里钉住的行为：区域一改，防抖保存那一步既写 config.json、也写当前配置组，并把状态栏
那句「未保存」变回「已保存」；没有当前配置组时（从没命名过）就只写 config.json ——
下次启动会回落到它，不必凭空造一个配置组出来。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

import main_window  # noqa: E402
from main_window import MainWindow  # noqa: E402
from models import ZoneProfile  # noqa: E402
from profile_store import ProfileStore  # noqa: E402


class ZoneAutoSaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.config_path = self.base / "config.json"
        self.profiles_dir = self.base / "profiles"
        for name, path in (
            ("CONFIG_PATH", self.config_path),
            ("EVENTS_DIR", self.base / "events"),
            ("PROFILES_DIR", self.profiles_dir),
        ):
            patcher = patch.object(main_window, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        for method in ("warning", "information"):
            patcher = patch.object(main_window.QMessageBox, method, lambda *a, **k: None)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _write_config(self, payload: dict) -> None:
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _saved_config(self) -> dict:
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def test_a_new_zone_is_written_into_the_active_profile(self) -> None:
        self._write_config({"version": 1, "active_profile": "夜间值守", "zones": []})
        ProfileStore(self.profiles_dir).save(ZoneProfile(name="夜间值守", zones=[]))
        window = MainWindow()

        window._add_zone()
        self.assertTrue(window.profile_dirty, "加了区域就该标成未保存")
        window._save_config()  # 防抖到点后真正执行的那一步

        saved = ProfileStore(self.profiles_dir).load("夜间值守")
        self.assertEqual([zone.name for zone in saved.zones], ["区域 1"])
        self.assertEqual(
            [zone["name"] for zone in self._saved_config()["zones"]], ["区域 1"]
        )
        self.assertFalse(window.profile_dirty, "存过之后状态栏该回到「已保存」")

    def test_zone_edits_survive_a_restart(self) -> None:
        """改完区域直接关窗（不走「保存配置组」按钮），下次启动仍在。"""
        self._write_config({"version": 1, "active_profile": "夜间值守", "zones": []})
        ProfileStore(self.profiles_dir).save(ZoneProfile(name="夜间值守", zones=[]))
        window = MainWindow()
        window._add_zone()
        window.closeEvent(type("E", (), {"accept": lambda self: None, "ignore": lambda self: None})())

        restarted = MainWindow()

        self.assertEqual([zone.name for zone in restarted.zones], ["区域 1"])

    def test_without_a_profile_only_the_config_is_written(self) -> None:
        """从没命名过配置组时不该凭空造一个文件出来 —— config.json 就够下次启动用。"""
        self._write_config({"version": 1, "zones": []})
        window = MainWindow()

        window._add_zone()
        window._save_config()

        self.assertEqual(list(self.profiles_dir.glob("*.json")), [])
        self.assertEqual(
            [zone["name"] for zone in self._saved_config()["zones"]], ["区域 1"]
        )

    def test_an_unrelated_setting_does_not_rewrite_the_profile(self) -> None:
        """只在区域真的变了时才写配置组：改个通知地址不该刷新配置组的时间戳。"""
        self._write_config({"version": 1, "active_profile": "夜间值守", "zones": []})
        store = ProfileStore(self.profiles_dir)
        store.save(ZoneProfile(name="夜间值守", zones=[]))
        before = (self.profiles_dir / "夜间值守.json").read_text(encoding="utf-8")
        window = MainWindow()

        window.settings_panel.notification_url_edit.setText("https://example.com/hook")
        window._save_config()

        self.assertEqual(
            (self.profiles_dir / "夜间值守.json").read_text(encoding="utf-8"), before
        )


if __name__ == "__main__":
    unittest.main()
