"""启动时读不出来的文件：配置组与 config.json 各自该怎么收场。

这两条路上原来各有一个把用户逼到死角的坑：

* **配置组读不出来**时异常会一路冒出 ``MainWindow.__init__``（列配置组、找回当前
  配置组都跑在启动路径上），窗口根本建不起来 —— 每次启动只看到一个「程序出错」，
  用户只能自己猜到去 profiles 目录里删文件；
* **config.json 读不出来**时程序回落默认值，这没错，但改名留档也可能失败（文件被
  同步盘/杀软/别的程序占着）。那时原文件还在原处又读不动，而退出时的保存照样会写：
  一份本来完好的配置就被默认值盖掉了，连备份都没有。

这里锁住的是收场方式：能起、能自愈、且绝不覆盖一份读不懂的原文件。
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
from models import ZoneDefinition, ZoneProfile  # noqa: E402
from profile_store import ProfileStore  # noqa: E402

GOOD_ZONE = {
    "name": "卸货区",
    "polygon": [[10, 20], [30, 40], [50, 60]],
    "closed": True,
    "dwell_seconds": 3.0,
    "cooldown_seconds": 30.0,
    "color": "#1E88E5",
}


class StartupResilienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.config_path = self.base / "config.json"
        self.profiles_dir = self.base / "profiles"
        # 不能让测试碰到开发机上真实的 config.json / events / profiles。
        for name, path in (
            ("CONFIG_PATH", self.config_path),
            ("EVENTS_DIR", self.base / "events"),
            ("PROFILES_DIR", self.profiles_dir),
        ):
            patcher = patch.object(main_window, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        # 启动失败要弹模态窗，offscreen 下它会一直等一个不会来的点击。
        for method in ("warning", "information"):
            patcher = patch.object(main_window.QMessageBox, method, lambda *a, **k: None)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _write_config(self, payload: object) -> None:
        self.config_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def _write_profile(self, name: str, text: str) -> Path:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        path = self.profiles_dir / f"{name}.json"
        path.write_text(text, encoding="utf-8")
        return path

    def _saved_config(self) -> dict:
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def _listed_profiles(self, window: MainWindow) -> list[str]:
        listing = window.zone_panel.profile_list
        return [
            listing.itemWidget(listing.item(index)).name
            for index in range(listing.count())
        ]

    # --- 配置组 ---------------------------------------------------------

    def test_a_broken_profile_does_not_block_startup(self) -> None:
        self._write_config({"version": 1, "zones": []})
        self._write_profile("好的", json.dumps({"name": "好的", "zones": []}))
        self._write_profile("坏的", "{不是 json")

        window = MainWindow()

        self.assertEqual(self._listed_profiles(window), ["好的"])

    def test_a_broken_active_profile_falls_back_to_the_config_zones(self) -> None:
        self._write_config(
            {"version": 1, "active_profile": "夜间值守", "zones": [GOOD_ZONE]}
        )
        broken = self._write_profile("夜间值守", "{坏了")

        window = MainWindow()

        # 区域来自 config.json —— 上次退出时写下的那一份，正是当时生效的区域。
        self.assertEqual([zone.name for zone in window.zones], ["卸货区"])
        # 指向它的指针要摘掉并落盘，否则下次启动还要为同一件事再报一次。
        self.assertEqual(window.config.active_profile, "")
        window._save_config()
        self.assertEqual(self._saved_config()["active_profile"], "")
        # 坏文件改名留档，内容一个字不少：用户可能只是改错了一个字符。
        self.assertFalse(broken.exists())
        backups = list(self.profiles_dir.glob("夜间值守.json.bad-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), "{坏了")
        # 列表里不再挂着那个名字。
        self.assertEqual(self._listed_profiles(window), [])

    def test_an_impossible_profile_name_is_cleared_instead_of_raising(self) -> None:
        """config.json 能手改出带斜杠的 active_profile：它永远对应不上文件。"""
        self._write_config(
            {"version": 1, "active_profile": "夜间/值守", "zones": [GOOD_ZONE]}
        )

        window = MainWindow()

        self.assertEqual(window.config.active_profile, "")
        self.assertEqual([zone.name for zone in window.zones], ["卸货区"])
        self.assertEqual(self._listed_profiles(window), [])

    def test_a_readable_profile_still_wins_over_the_config_zones(self) -> None:
        """上面几条都在讲回落；这里确认好配置组照旧生效。"""
        self._write_config(
            {"version": 1, "active_profile": "夜间值守", "zones": [GOOD_ZONE]}
        )
        ProfileStore(self.profiles_dir).save(
            ZoneProfile(name="夜间值守", zones=[ZoneDefinition(name="入口")])
        )

        window = MainWindow()

        self.assertEqual([zone.name for zone in window.zones], ["入口"])
        self.assertEqual(window.config.active_profile, "夜间值守")

    # --- config.json ----------------------------------------------------

    def test_an_unreadable_config_that_cannot_be_moved_is_left_alone(self) -> None:
        self._write_config([1, 2, 3])
        original = self.config_path.read_text(encoding="utf-8")

        with patch.object(main_window.ConfigStore, "quarantine", return_value=None):
            window = MainWindow()
            self.assertFalse(window._config_trusted)
            self.assertFalse(window._config_writable)
            window._save_config()

        # 这是本条的重点：读不懂又挪不走的原文件必须保持原样，否则用户关窗那一刻
        # 那份默认配置就把它盖掉了，连备份都没有。
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), original)
        self.assertIn("未写入", window.source_panel.status_label.text())

    def test_a_quarantined_config_is_writable_again(self) -> None:
        """隔离成功时本会话该照常落盘 —— 文件已经挪走，改动是唯一的一份。"""
        self._write_config([1, 2, 3])

        window = MainWindow()

        self.assertFalse(window._config_trusted)
        self.assertTrue(window._config_writable)
        self.assertEqual(len(list(self.base.glob("config.json.bad-*"))), 1)

        window._save_config()

        saved = self._saved_config()
        self.assertEqual(saved["version"], 1)
        self.assertEqual(saved["zones"], [])

    # --- 新建配置组 -----------------------------------------------------

    def test_a_new_profile_exists_on_disk_right_away(self) -> None:
        """新建即建文件：否则列表里挂着一条没有文件的条目，切走后再点「应用」只会报错。"""
        self._write_config({"version": 1, "zones": []})
        window = MainWindow()

        with patch.object(
            main_window.QInputDialog, "getText", return_value=("新配置组", True)
        ):
            window._new_profile()

        self.assertEqual(window.config.active_profile, "新配置组")
        self.assertTrue(window.profile_store.exists("新配置组"))
        self.assertEqual(window.profile_store.load("新配置组").zones, [])
        self.assertEqual(self._listed_profiles(window), ["新配置组"])
        # 指针要安排落盘，不然重启就丢了。
        window._save_config()
        self.assertEqual(self._saved_config()["active_profile"], "新配置组")

    def test_a_new_profile_with_an_illegal_name_is_refused_not_raised(self) -> None:
        self._write_config({"version": 1, "zones": []})
        window = MainWindow()

        with patch.object(main_window.QInputDialog, "getText", return_value=("a/b", True)):
            window._new_profile()

        self.assertEqual(window.config.active_profile, "")
        self.assertEqual(self._listed_profiles(window), [])


if __name__ == "__main__":
    unittest.main()
