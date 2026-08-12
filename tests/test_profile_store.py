from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
