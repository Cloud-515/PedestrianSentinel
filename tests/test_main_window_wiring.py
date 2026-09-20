"""界面与检测线程之间的帧接线。

这里测的是 ``MainWindow._connect_worker`` 里最容易出错的那一条：frame_ready 需要
先交给画面，再回执给**发出这一帧的那个** worker，让它可以继续投递下一帧。

把回执写成 ``self.worker`` 是个很自然的写法，但它是错的：点「停止」再点「打开」
之后，旧 worker 留在队列里的帧会被新 worker 的界面处理，于是清掉的是新 worker 的
待显示标志 —— 这种偏差在真机上看不出来，却正好破坏了丢帧节流要守住的东西
（队列里最多一帧）。
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

import main_window
from main_window import MainWindow


class FakeWorker(QObject):
    """只保留 MainWindow 会接的那几个信号，不碰真实的推理与视频源。"""

    frame_ready = Signal(object)
    event_ready = Signal(object)
    event_updated = Signal(object)
    status_changed = Signal(str)
    source_opened = Signal(int, int, float)
    progress_changed = Signal(float)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.consumed = 0

    def frame_consumed(self) -> None:
        self.consumed += 1


class MainWindowFrameWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        # 不能让测试碰到开发机上真实的 config.json / events / profiles。
        for name, path in (
            ("CONFIG_PATH", base / "config.json"),
            ("EVENTS_DIR", base / "events"),
            ("PROFILES_DIR", base / "profiles"),
        ):
            patcher = patch.object(main_window, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.window = MainWindow()

    def test_frame_reaches_the_video_widget_and_is_acknowledged(self) -> None:
        worker = FakeWorker()
        self.window._connect_worker(worker)
        frame = np.zeros((8, 12, 3), dtype=np.uint8)

        worker.frame_ready.emit(frame)

        self.assertIsNotNone(self.window.video_widget._image)
        self.assertEqual(self.window.video_widget._frame_size, (12, 8))
        self.assertEqual(worker.consumed, 1)

    def test_acknowledgement_goes_to_the_emitting_worker_not_the_current_one(self) -> None:
        stale = FakeWorker()
        current = FakeWorker()
        self.window._connect_worker(stale)
        self.window._connect_worker(current)
        self.window.worker = current

        stale.frame_ready.emit(np.zeros((4, 4, 3), dtype=np.uint8))

        # 旧 worker 的帧由旧 worker 自己回执；新 worker 的标志不能被它动。
        self.assertEqual(stale.consumed, 1)
        self.assertEqual(current.consumed, 0)

    def test_every_frame_is_drawn_when_the_interface_keeps_up(self) -> None:
        worker = FakeWorker()
        self.window._connect_worker(worker)

        for index in range(5):
            worker.frame_ready.emit(np.full((4, 4, 3), index, dtype=np.uint8))

        self.assertEqual(worker.consumed, 5)


class MainWindowRetentionTests(unittest.TestCase):
    """设置页里的取证留存设置要真的落到 config.json，并真的按它清理。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        for name, path in (
            ("CONFIG_PATH", self.base / "config.json"),
            ("EVENTS_DIR", self.base / "events"),
            ("PROFILES_DIR", self.base / "profiles"),
        ):
            patcher = patch.object(main_window, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.window = MainWindow()

    def _screenshot(self, name: str, age_days: float, size: int = 1024) -> Path:
        directory = self.window.event_store.screenshot_dir
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / f"{name}.jpg"
        file.write_bytes(b"x" * size)
        stamp = time.time() - age_days * 86_400
        os.utime(file, (stamp, stamp))
        return file

    def test_settings_panel_shows_the_defaults_from_the_config(self) -> None:
        days, megabytes = self.window.settings_panel.retention()

        self.assertEqual(days, 15)
        self.assertEqual(megabytes, 2048)

    def test_edited_retention_is_written_into_the_config_file(self) -> None:
        self.window.settings_panel.retention_days_spin.setValue(30)
        self.window.settings_panel.retention_mb_spin.setValue(512)

        self.window._save_config()

        saved = json.loads((self.base / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["screenshot_retention_days"], 30)
        self.assertEqual(saved["screenshot_retention_mb"], 512)
        # 界面上改一次就要生效，不能等下次启动。
        self.assertEqual(self.window.config.screenshot_retention_days, 30)

    def test_retention_survives_a_restart(self) -> None:
        self.window.settings_panel.retention_days_spin.setValue(7)
        self.window._save_config()

        reopened = MainWindow()

        self.assertEqual(reopened.settings_panel.retention_days_spin.value(), 7)

    def test_automatic_prune_removes_expired_screenshots(self) -> None:
        expired = self._screenshot("expired", age_days=40)
        fresh = self._screenshot("fresh", age_days=1)

        self.window._run_retention()

        self.assertFalse(expired.exists())
        self.assertTrue(fresh.exists())

    def test_manual_prune_reports_what_it_did(self) -> None:
        self._screenshot("expired", age_days=40)

        self.window._prune_now()

        self.assertIn("已清理 1 张", self.window.source_panel.status_label.text())

    def test_manual_prune_is_available_even_when_the_config_could_not_be_read(self) -> None:
        """配置读坏时退回默认值，自动清理会停手 —— 但用户看着界面按的那一下照做。"""
        expired = self._screenshot("expired", age_days=40)
        self.window._config_trusted = False

        self.window._run_retention()
        self.assertTrue(expired.exists(), "配置不可信时不该自动删取证材料")

        self.window._prune_now()
        self.assertFalse(expired.exists())

    def test_disabled_retention_touches_nothing(self) -> None:
        expired = self._screenshot("expired", age_days=4000)
        self.window.settings_panel.retention_days_spin.setValue(0)
        self.window.settings_panel.retention_mb_spin.setValue(0)

        self.window._run_retention()

        self.assertTrue(expired.exists())

    def test_storage_label_counts_the_screenshots(self) -> None:
        self._screenshot("a", age_days=1, size=2048)
        self._screenshot("b", age_days=2, size=1024)

        self.window._refresh_storage_usage()

        text = self.window.settings_panel.storage_label.text()
        self.assertIn("已存 2 张", text)
        self.assertIn("3 KB", text)


if __name__ == "__main__":
    unittest.main()
