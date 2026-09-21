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
from models import AlarmEvent


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

    def _enable_retention(self, days: int = 15, megabytes: int = 2048) -> None:
        panel = self.window.settings_panel
        panel.retention_days_spin.setValue(days)
        panel.retention_mb_spin.setValue(megabytes)
        panel.retention_enabled_cb.setChecked(True)

    def test_automatic_cleanup_is_off_on_first_launch(self) -> None:
        """默认必须是关的：程序不该在用户还没看过设置的时候就动他的取证材料。"""
        enabled, days, megabytes = self.window.settings_panel.retention()

        self.assertFalse(enabled)
        # 数值仍然预填成建议值，方便用户一键打开，但开关没开就不生效。
        self.assertEqual(days, 15)
        self.assertEqual(megabytes, 2048)
        self.assertFalse(self.window.config.screenshot_retention_enabled)
        self.assertFalse(self.window._retention_policy().enabled)

    def test_default_off_keeps_expired_screenshots_even_when_pruned(self) -> None:
        expired = self._screenshot("expired", age_days=4000)

        self.window._run_retention()
        self.window._prune_now()

        self.assertTrue(expired.exists(), "开关没打开就不该删任何东西")
        self.assertIn("未开启", self.window.source_panel.status_label.text())

    def test_numbers_are_locked_until_the_switch_is_on(self) -> None:
        panel = self.window.settings_panel

        self.assertFalse(panel.retention_days_spin.isEnabled())
        self.assertFalse(panel.retention_mb_spin.isEnabled())
        self.assertFalse(panel.prune_btn.isEnabled())

        panel.retention_enabled_cb.setChecked(True)

        self.assertTrue(panel.retention_days_spin.isEnabled())
        self.assertTrue(panel.retention_mb_spin.isEnabled())
        self.assertTrue(panel.prune_btn.isEnabled())

    def test_edited_retention_is_written_into_the_config_file(self) -> None:
        self._enable_retention(days=30, megabytes=512)

        self.window._save_config()

        saved = json.loads((self.base / "config.json").read_text(encoding="utf-8"))
        self.assertTrue(saved["screenshot_retention_enabled"])
        self.assertEqual(saved["screenshot_retention_days"], 30)
        self.assertEqual(saved["screenshot_retention_mb"], 512)
        # 界面上改一次就要生效，不能等下次启动。
        self.assertEqual(self.window.config.screenshot_retention_days, 30)

    def test_retention_survives_a_restart(self) -> None:
        self._enable_retention(days=7)
        self.window._save_config()

        reopened = MainWindow()

        panel = reopened.settings_panel
        self.assertTrue(panel.retention_enabled_cb.isChecked())
        self.assertEqual(panel.retention_days_spin.value(), 7)

    def test_switching_off_again_is_remembered(self) -> None:
        self._enable_retention()
        self.window._save_config()
        self.window.settings_panel.retention_enabled_cb.setChecked(False)
        self.window._save_config()

        reopened = MainWindow()

        self.assertFalse(reopened.settings_panel.retention_enabled_cb.isChecked())
        self.assertFalse(reopened._retention_policy().enabled)

    def test_automatic_prune_removes_expired_screenshots_once_enabled(self) -> None:
        expired = self._screenshot("expired", age_days=40)
        fresh = self._screenshot("fresh", age_days=1)
        self._enable_retention(days=15)

        self.window._run_retention()

        self.assertFalse(expired.exists())
        self.assertTrue(fresh.exists())

    def test_manual_prune_reports_what_it_did(self) -> None:
        self._screenshot("expired", age_days=40)
        self._enable_retention(days=15)

        self.window._prune_now()

        self.assertIn("已清理 1 张", self.window.source_panel.status_label.text())

    def test_automatic_prune_stops_when_the_config_could_not_be_read(self) -> None:
        """配置读坏时退回的默认值是「关闭」，拿一份不代表用户意愿的策略去删材料不可接受。"""
        expired = self._screenshot("expired", age_days=4000)
        self._enable_retention(days=15)
        self.window._config_trusted = False

        self.window._run_retention()

        self.assertTrue(expired.exists())

    def test_zero_rules_still_mean_never_clean(self) -> None:
        expired = self._screenshot("expired", age_days=4000)
        self._enable_retention(days=0, megabytes=0)

        self.window._run_retention()

        self.assertTrue(expired.exists())

    def test_storage_label_counts_the_screenshots(self) -> None:
        self._screenshot("a", age_days=1, size=2048)
        self._screenshot("b", age_days=2, size=1024)

        self.window._refresh_storage_usage()

        text = self.window.settings_panel.storage_label.text()
        self.assertIn("已存 2 张", text)
        self.assertIn("3 KB", text)
        self.assertIn("自动清理已关闭", text)

    def test_storage_label_follows_the_switch(self) -> None:
        """占用提示是「现在生效的是什么」的指示，拨了开关就该跟着变。"""
        label = self.window.settings_panel.storage_label

        self.assertIn("自动清理已关闭", label.text())

        self._enable_retention(days=7, megabytes=512)

        self.assertIn("保留 7 天", label.text())
        self.assertIn("512 MB", label.text())


class RecordingDispatcher:
    """替掉真正的发送器：只记录被交出去的作业，不碰网络。"""

    def __init__(self) -> None:
        self.jobs: list[object] = []
        self.callbacks: list[object] = []
        self.closed = False

    def notify(self, job: object, on_result: object = None) -> bool:
        self.jobs.append(job)
        self.callbacks.append(on_result)
        return True

    def close(self, timeout: float = 2.0) -> None:
        self.closed = True


class MainWindowNotificationTests(unittest.TestCase):
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
        self.dispatcher = RecordingDispatcher()
        self.window.notification_dispatcher = self.dispatcher

    def _enable(self, url: str = "https://example.com/hook", **extra: object) -> None:
        panel = self.window.settings_panel
        panel.notification_url_edit.setText(url)
        for name, value in extra.items():
            getattr(panel, name).setChecked(value)
        panel.notification_enabled_cb.setChecked(True)

    def _alarm(self) -> AlarmEvent:
        return AlarmEvent(
            source="rtsp://camera/live",
            zone_name="北侧入口",
            track_id="12",
            entered_at_seconds=1.0,
            alarm_at_seconds=3.0,
            wall_time="2026-09-21 10:00:00",
            operation_mode="monitor",
        )

    def test_settings_round_trip_through_the_config_file(self) -> None:
        self._enable(url="https://example.com/hook", notification_screenshot_cb=True)
        self.window.settings_panel.notification_format_combo.setCurrentIndex(
            self.window.settings_panel.notification_format_combo.findData("text_bot")
        )

        self.window._save_config()

        saved = json.loads((self.base / "config.json").read_text(encoding="utf-8"))
        self.assertTrue(saved["notification_enabled"])
        self.assertEqual(saved["notification_url"], "https://example.com/hook")
        self.assertEqual(saved["notification_format"], "text_bot")
        self.assertTrue(saved["notification_include_screenshot"])

        reopened = MainWindow()
        panel = reopened.settings_panel
        self.assertTrue(panel.notification_enabled_cb.isChecked())
        self.assertEqual(panel.notification_url_edit.text(), "https://example.com/hook")
        self.assertEqual(panel.notification_format_combo.currentData(), "text_bot")
        self.assertTrue(panel.notification_screenshot_cb.isChecked())

    def test_alarm_is_dispatched_when_enabled(self) -> None:
        self._enable()

        self.window._on_alarm_event(self._alarm())

        self.assertEqual(len(self.dispatcher.jobs), 1)
        job = self.dispatcher.jobs[0]
        self.assertEqual(job.url, "https://example.com/hook")
        self.assertIn("北侧入口", job.body.decode("utf-8"))

    def test_no_notification_when_disabled(self) -> None:
        self.window._on_alarm_event(self._alarm())

        self.assertEqual(self.dispatcher.jobs, [])

    def test_enabled_but_url_missing_does_not_dispatch(self) -> None:
        self.window.settings_panel.notification_enabled_cb.setChecked(True)

        self.window._on_alarm_event(self._alarm())

        self.assertEqual(self.dispatcher.jobs, [])
        self.assertIn(
            "地址为空", self.window.settings_panel.notification_status_label.text()
        )

    def test_test_button_dispatches_a_marked_test_notification(self) -> None:
        self._enable()

        self.window._send_test_notification()

        self.assertEqual(len(self.dispatcher.jobs), 1)
        body = json.loads(self.dispatcher.jobs[0].body)
        self.assertIn("测试", body["text"])
        # 测试通知不能写进报警记录，也不该在表格里冒出来。
        self.assertEqual(self.window.event_panel.table.rowCount(), 0)

    def test_test_button_without_a_usable_url_explains_instead_of_sending(self) -> None:
        self.window._send_test_notification()

        self.assertEqual(self.dispatcher.jobs, [])
        self.assertIn("请先启用", self.window.source_panel.status_label.text())

    def test_result_is_shown_in_the_settings_panel(self) -> None:
        self.window._on_notification_result(False, "通知发送失败：连接失败")

        label = self.window.settings_panel.notification_status_label
        self.assertIn("连接失败", label.text())
        self.assertIn("连接失败", self.window.source_panel.status_label.text())

    def test_closing_the_window_closes_the_dispatcher(self) -> None:
        self.window.close()

        self.assertTrue(self.dispatcher.closed)


if __name__ == "__main__":
    unittest.main()
