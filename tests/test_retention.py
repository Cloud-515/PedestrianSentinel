"""取证截图的留存策略。

这是本项目第二处会删数据的代码，而且是**自动**删（启动时 + 每小时一次），所以测试
要盯住的不只是「会不会删该删的」，还有「会不会删不该删的」：目录里的非截图文件、
不是自己生成的东西、以及两条规则都没设时绝不能动任何东西。
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from retention import (
    RetentionPolicy,
    RetentionResult,
    directory_usage,
    format_size,
    prune_screenshots,
)


class ScreenshotDirectory:
    """临时截图目录，可指定每个文件的年龄与大小。"""

    def __init__(self, test: unittest.TestCase) -> None:
        temporary = tempfile.TemporaryDirectory()
        test.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.now = time.time()

    def add(self, name: str, age_days: float, size: int = 1024, suffix: str = ".jpg") -> Path:
        file = self.path / f"{name}{suffix}"
        file.write_bytes(b"x" * size)
        stamp = self.now - age_days * 86_400
        import os

        os.utime(file, (stamp, stamp))
        return file

    def add_foreign(self, name: str, age_days: float) -> Path:
        """不是截图的东西：策略不该碰它。"""
        file = self.path / name
        file.write_text("留着", encoding="utf-8")
        stamp = self.now - age_days * 86_400
        import os

        os.utime(file, (stamp, stamp))
        return file

    def names(self) -> set[str]:
        return {item.name for item in self.path.iterdir() if item.is_file()}


class AgeRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = ScreenshotDirectory(self)

    def test_removes_only_screenshots_older_than_the_limit(self) -> None:
        old = self.directory.add("old", age_days=16)
        edge = self.directory.add("edge", age_days=15)
        fresh = self.directory.add("fresh", age_days=2)

        result = prune_screenshots(
            self.directory.path, RetentionPolicy(days=15), now=self.directory.now
        )

        self.assertFalse(old.exists())
        # 刚好卡在边界上的不算过期（cutoff 用的是「早于」，不是「不晚于」）。
        self.assertTrue(edge.exists())
        self.assertTrue(fresh.exists())
        self.assertEqual(result.removed, 1)
        self.assertEqual(result.kept, 2)
        self.assertEqual(result.failed, 0)

    def test_default_of_fifteen_days_keeps_the_usual_forensic_window(self) -> None:
        recent = self.directory.add("recent", age_days=14.5)
        ancient = self.directory.add("ancient", age_days=400)

        prune_screenshots(
            self.directory.path, RetentionPolicy(days=15), now=self.directory.now
        )

        self.assertTrue(recent.exists())
        self.assertFalse(ancient.exists())

    def test_zero_days_disables_the_age_rule(self) -> None:
        ancient = self.directory.add("ancient", age_days=4000)

        prune_screenshots(
            self.directory.path, RetentionPolicy(days=0), now=self.directory.now
        )

        self.assertTrue(ancient.exists())

    def test_never_touches_files_that_are_not_screenshots(self) -> None:
        keep = self.directory.add_foreign("说明.txt", age_days=900)
        self.directory.add_foreign("alarm_events.jsonl", age_days=900)
        old_shot = self.directory.add("old", age_days=900)

        prune_screenshots(
            self.directory.path, RetentionPolicy(days=15), now=self.directory.now
        )

        self.assertTrue(keep.exists())
        self.assertTrue((self.directory.path / "alarm_events.jsonl").exists())
        self.assertFalse(old_shot.exists())

    def test_ignores_directories_even_if_they_look_like_screenshots(self) -> None:
        nested = self.directory.path / "伪装.jpg"
        nested.mkdir()
        (nested / "inner.jpg").write_bytes(b"x")

        result = prune_screenshots(
            self.directory.path, RetentionPolicy(days=1), now=self.directory.now
        )

        self.assertTrue(nested.is_dir())
        self.assertTrue((nested / "inner.jpg").exists())
        self.assertEqual(result.removed, 0)

    def test_accepts_the_other_screenshot_suffixes(self) -> None:
        jpeg = self.directory.add("a", age_days=30, suffix=".jpeg")
        png = self.directory.add("b", age_days=30, suffix=".png")
        self.directory.add("c", age_days=30, suffix=".mp4")

        prune_screenshots(
            self.directory.path, RetentionPolicy(days=15), now=self.directory.now
        )

        self.assertFalse(jpeg.exists())
        self.assertFalse(png.exists())
        self.assertTrue((self.directory.path / "c.mp4").exists())


class SizeRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = ScreenshotDirectory(self)

    def test_deletes_oldest_first_until_under_the_cap(self) -> None:
        oldest = self.directory.add("oldest", age_days=3, size=1000)
        middle = self.directory.add("middle", age_days=2, size=1000)
        newest = self.directory.add("newest", age_days=1, size=1000)

        result = prune_screenshots(
            self.directory.path,
            RetentionPolicy(max_bytes=2000),
            now=self.directory.now,
        )

        self.assertFalse(oldest.exists())
        self.assertTrue(middle.exists())
        self.assertTrue(newest.exists())
        self.assertEqual(result.removed, 1)
        self.assertEqual(result.removed_bytes, 1000)
        self.assertEqual(result.kept_bytes, 2000)

    def test_keeps_going_until_the_total_fits(self) -> None:
        for index in range(5):
            self.directory.add(f"shot{index}", age_days=5 - index, size=1000)

        result = prune_screenshots(
            self.directory.path, RetentionPolicy(max_bytes=1500), now=self.directory.now
        )

        self.assertEqual(result.kept, 1)
        self.assertEqual(result.kept_bytes, 1000)
        self.assertEqual(result.removed, 4)

    def test_leaves_everything_alone_when_under_the_cap(self) -> None:
        self.directory.add("a", age_days=1, size=100)
        self.directory.add("b", age_days=2, size=100)

        result = prune_screenshots(
            self.directory.path,
            RetentionPolicy(max_bytes=10 * 1024 * 1024),
            now=self.directory.now,
        )

        self.assertEqual(result.removed, 0)
        self.assertEqual(result.kept, 2)

    def test_zero_cap_disables_the_size_rule(self) -> None:
        self.directory.add("a", age_days=1, size=100)

        result = prune_screenshots(
            self.directory.path, RetentionPolicy(max_bytes=0), now=self.directory.now
        )

        self.assertEqual(result.removed, 0)

    def test_both_rules_apply_together(self) -> None:
        expired = self.directory.add("expired", age_days=40, size=1000)
        self.directory.add("old", age_days=3, size=1000)
        self.directory.add("new", age_days=1, size=1000)

        # 天数先删掉过期的那张，剩下的 2000 字节仍超过 1500 的上限，于是再删最旧的一张。
        result = prune_screenshots(
            self.directory.path,
            RetentionPolicy(days=15, max_bytes=1500),
            now=self.directory.now,
        )

        self.assertFalse(expired.exists())
        self.assertEqual(self.directory.names(), {"new.jpg"})
        self.assertEqual(result.removed, 2)


class SafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = ScreenshotDirectory(self)

    def test_disabled_policy_touches_nothing(self) -> None:
        ancient = self.directory.add("ancient", age_days=9000, size=10**6)

        result = prune_screenshots(
            self.directory.path, RetentionPolicy(days=0, max_bytes=0)
        )

        self.assertTrue(ancient.exists())
        self.assertEqual(result, RetentionResult())
        self.assertTrue(RetentionPolicy(days=15).enabled)
        self.assertFalse(RetentionPolicy().enabled)

    def test_missing_directory_is_not_an_error(self) -> None:
        missing = self.directory.path / "还没有人报过警"

        result = prune_screenshots(missing, RetentionPolicy(days=1))

        self.assertEqual(result.removed, 0)
        self.assertEqual(directory_usage(missing), (0, 0))

    def test_empty_directory_is_a_no_op(self) -> None:
        result = prune_screenshots(self.directory.path, RetentionPolicy(days=1))

        self.assertEqual(result, RetentionResult())

    def test_negative_values_are_treated_as_disabled(self) -> None:
        policy = RetentionPolicy(days=-5, max_bytes=-1)
        ancient = self.directory.add("ancient", age_days=9000)

        prune_screenshots(self.directory.path, policy, now=self.directory.now)

        self.assertEqual((policy.days, policy.max_bytes), (0, 0))
        self.assertTrue(ancient.exists())

    def test_kept_files_are_reported_for_the_interface(self) -> None:
        self.directory.add("a", age_days=1, size=2048)
        self.directory.add("b", age_days=2, size=1024)

        result = prune_screenshots(
            self.directory.path, RetentionPolicy(days=15), now=self.directory.now
        )

        self.assertEqual((result.kept, result.kept_bytes), (2, 3072))
        self.assertFalse(result.changed)

    def test_usage_counts_match_the_directory(self) -> None:
        self.directory.add("a", age_days=1, size=100)
        self.directory.add("b", age_days=2, size=200)
        self.directory.add_foreign("readme.txt", age_days=1)

        self.assertEqual(directory_usage(self.directory.path), (2, 300))


class ReportingTests(unittest.TestCase):
    def test_describes_what_it_did(self) -> None:
        removed = RetentionResult(removed=3, removed_bytes=3 * 1024**2, kept=7, kept_bytes=1024)
        self.assertIn("已清理 3 张", removed.describe())
        self.assertIn("3.0 MB", removed.describe())
        self.assertIn("保留 7 张", removed.describe())

        untouched = RetentionResult(kept=7, kept_bytes=1024)
        self.assertIn("无需清理", untouched.describe())

        failed = RetentionResult(removed=1, failed=2, kept=1, kept_bytes=10)
        self.assertIn("2 张删除失败", failed.describe())

    def test_formats_sizes_for_humans(self) -> None:
        self.assertEqual(format_size(512), "512 B")
        self.assertEqual(format_size(2048), "2 KB")
        self.assertEqual(format_size(5 * 1024**2), "5.0 MB")
        self.assertEqual(format_size(3 * 1024**3), "3.00 GB")


if __name__ == "__main__":
    unittest.main()
