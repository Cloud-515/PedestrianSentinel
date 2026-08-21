from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app_paths


class ResourceLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        patcher = patch.object(app_paths, "APP_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_prefers_earlier_candidate_when_both_exist(self) -> None:
        (self.root / "assets").mkdir()
        (self.root / "assets" / "warming_converted.wav").write_bytes(b"grouped")
        (self.root / "warming_converted.wav").write_bytes(b"flat")

        resolved = app_paths.resource(
            "assets/warming_converted.wav", "warming_converted.wav"
        )

        self.assertEqual(resolved, self.root / "assets" / "warming_converted.wav")

    def test_falls_back_to_flat_layout_when_grouped_missing(self) -> None:
        (self.root / "warming_converted.wav").write_bytes(b"flat")

        resolved = app_paths.resource(
            "assets/warming_converted.wav", "warming_converted.wav"
        )

        self.assertEqual(resolved, self.root / "warming_converted.wav")

    def test_returns_first_candidate_when_nothing_exists(self) -> None:
        # 让调用方的报错信息指向推荐位置，而不是最后的兜底位置。
        resolved = app_paths.resource(
            "assets/warming_converted.wav", "warming_converted.wav"
        )

        self.assertEqual(resolved, self.root / "assets" / "warming_converted.wav")
        self.assertFalse(resolved.exists())

    def test_finds_directory_candidate(self) -> None:
        model_dir = self.root / "models" / "yolo11n_int8_openvino_model"
        model_dir.mkdir(parents=True)

        self.assertEqual(
            app_paths.resource("models/yolo11n_int8_openvino_model"), model_dir
        )

    def test_rejects_empty_candidate_list(self) -> None:
        with self.assertRaises(ValueError):
            app_paths.resource()


class DataDirTests(unittest.TestCase):
    def setUp(self) -> None:
        app_paths.data_dir.cache_clear()
        # 缓存是模块级的，用完必须清掉，否则污染同批次的其他测试。
        self.addCleanup(app_paths.data_dir.cache_clear)

    def test_uses_app_dir_when_writable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(app_paths, "APP_DIR", root):
                self.assertEqual(app_paths.data_dir(), root)
                self.assertEqual(app_paths.data("logs", "app.log"), root / "logs" / "app.log")

    def test_falls_back_to_localappdata_when_app_dir_read_only(self) -> None:
        with (
            patch.object(app_paths, "_is_writable", return_value=False),
            patch.dict("os.environ", {"LOCALAPPDATA": r"C:\Users\tester\AppData\Local"}),
        ):
            resolved = app_paths.data_dir()

        self.assertEqual(
            resolved, Path(r"C:\Users\tester\AppData\Local") / app_paths.APP_NAME
        )

    def test_is_writable_reports_false_for_unusable_location(self) -> None:
        self.assertFalse(app_paths._is_writable(Path("Z:/no-such-drive/pp-human")))

    def test_is_writable_reports_true_for_temporary_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(app_paths._is_writable(Path(directory)))


class AppDirResolutionTests(unittest.TestCase):
    def test_source_mode_uses_module_directory(self) -> None:
        with patch.object(app_paths, "IS_FROZEN", False):
            self.assertEqual(
                app_paths._resolve_app_dir(), Path(app_paths.__file__).resolve().parent
            )

    def test_frozen_mode_uses_executable_directory(self) -> None:
        # 单目录版的资源在 exe 同级，而不是 _internal/ 里面。
        fake_exe = Path(tempfile.gettempdir()) / "行人警戒区域监控.exe"
        with (
            patch.object(app_paths, "IS_FROZEN", True),
            patch.object(sys, "executable", str(fake_exe)),
        ):
            self.assertEqual(app_paths._resolve_app_dir(), fake_exe.resolve().parent)


if __name__ == "__main__":
    unittest.main()
