from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from models import ZoneDefinition, ZoneProfile
from profile_store import ProfileStore


class ProfileStoreTests(unittest.TestCase):
    def test_saves_and_loads_multiple_zones(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            profile = ZoneProfile(
                name="仓库夜间监控",
                description="入口和装卸区",
                zones=[
                    ZoneDefinition(name="入口", polygon=[[1, 2], [3, 4]], dwell_seconds=2),
                    ZoneDefinition(name="装卸区", polygon=[[5, 6], [7, 8]], dwell_seconds=5),
                ],
            )
            store.save(profile)
            loaded = store.load(profile.name)
            self.assertEqual(loaded.name, profile.name)
            self.assertEqual(len(loaded.zones), 2)
            self.assertEqual(loaded.zones[1].dwell_seconds, 5)
            self.assertEqual(store.list_profiles()[0].zone_count, 2)

    def test_rejects_path_like_profile_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            with self.assertRaises(ValueError):
                store.save(ZoneProfile(name="../outside"))

    def test_delete_removes_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            store.save(ZoneProfile(name="临时配置"))
            store.delete("临时配置")
            self.assertFalse(store.exists("临时配置"))


class ProfileStoreResilienceTests(unittest.TestCase):
    """一个读不出来的配置组不该连累别的配置组，更不该把程序挡在门外。

    「列配置组」是启动路径上的一环（``MainWindow._load_controls`` 会刷新列表），
    以前它逐个 ``load`` 且一个有异常就往外抛，于是 profiles 目录里任何一个坏文件
    都让窗口建不起来 —— 用户每次启动只看到一个「程序出错」。配置组和 config.json
    一样是给人手改的文本，现场改坏一处完全可能。
    """

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.store = ProfileStore(self.directory)

    def _write(self, name: str, text: str) -> Path:
        path = self.directory / f"{name}.json"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_broken_file_does_not_hide_the_healthy_ones(self) -> None:
        self.store.save(ZoneProfile(name="夜间值守", zones=[ZoneDefinition(name="入口")]))
        broken = self._write("坏的", "{不是 json")

        with self.assertLogs("profile_store", level=logging.WARNING) as captured:
            summaries = self.store.list_profiles()

        self.assertEqual([summary.name for summary in summaries], ["夜间值守"])
        # 不改名、不删：坏文件留在原处，日志里有它的名字，用户能自己去处理。
        self.assertTrue(broken.exists())
        self.assertTrue(
            any("坏的.json" in message for message in captured.output), captured.output
        )

    def test_load_names_the_profile_for_every_kind_of_damage(self) -> None:
        """报错里必须有配置组的名字，否则用户只看到一个 codec 细节。"""
        cases = {
            "不是 json": "{不是 json",
            "不是 utf-8": None,
        }
        for label, text in cases.items():
            with self.subTest(kind=label):
                name = f"坏-{label}"
                if text is None:
                    (self.directory / f"{name}.json").write_bytes(b"\xff\xfe\x00\x01")
                else:
                    self._write(name, text)

                with self.assertRaises(ValueError) as caught:
                    self.store.load(name)

                self.assertIn(name, str(caught.exception))

    def test_quarantine_moves_the_file_and_keeps_its_content(self) -> None:
        path = self._write("夜间值守", "{坏掉的内容")

        backup = self.store.quarantine("夜间值守")

        self.assertIsNotNone(backup)
        self.assertFalse(path.exists())
        self.assertEqual(backup.read_text(encoding="utf-8"), "{坏掉的内容")
        # 后缀不是 .json，所以这份留档不会被再次当成一个配置组列出来。
        self.assertFalse(backup.name.endswith(".json"))
        self.assertEqual(self.store.list_profiles(), [])

    def test_quarantine_without_a_file_is_a_no_op(self) -> None:
        self.assertIsNone(self.store.quarantine("不存在"))

    def test_exists_answers_instead_of_raising_for_an_impossible_name(self) -> None:
        """config.json 里的 active_profile 能手改出带斜杠的名字。

        问一句「有没有」就抛异常的话，这一问会发生在 ``__init__`` 里，窗口直接建不起来。
        """
        self.assertFalse(self.store.exists("a/b"))
        self.assertFalse(self.store.exists(""))
        self.assertFalse(self.store.is_valid_name("../outside"))
        self.assertTrue(self.store.is_valid_name("夜间值守"))
        self.assertFalse(self.store.exists("夜间值守"))


if __name__ == "__main__":
    unittest.main()
