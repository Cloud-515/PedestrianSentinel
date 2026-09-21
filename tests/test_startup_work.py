"""启动时该做多少事：只做「窗口能用」必须的那些。

现场反馈「启动慢」。实测下来 15 秒里有 9 秒是 import torch/ultralytics/supervision 那条
推理链（只有真的开始推理才需要），5.6 秒是 MainWindow() —— 其中 3.4 秒是把 200 条报警
记录**填了两遍**表，而 11 列开着 ResizeToContents 时每写一个单元格都会重算列宽。

这里钉住推迟之后的行为：

* 报警记录只载一次（以前 _load_controls 里的 _apply_operation_mode 载一遍、__init__
  又载一遍）；
* 占用统计与设备探测都不在构造期间做 —— 前者要遍历截图目录，后者要 import torch；
* 设备探测结果回来之前「打开」是禁用的（否则会带着没填好的设备列表开始检测），
  结果回来之后下拉框、状态栏、按钮一起就位。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

import main_window  # noqa: E402
from main_window import MainWindow  # noqa: E402


class StartupWorkTests(unittest.TestCase):
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
        for method in ("warning", "information"):
            patcher = patch.object(main_window.QMessageBox, method, lambda *a, **k: None)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_the_alarm_history_is_loaded_exactly_once(self) -> None:
        store = Mock()
        store.load_recent.return_value = []
        store.resolve_screenshot = Mock(return_value=None)

        with patch.object(main_window, "EventStore", return_value=store):
            MainWindow()

        self.assertEqual(
            store.load_recent.call_count, 1, "启动时不该把同一批记录载两遍"
        )

    def test_the_storage_scan_is_not_done_during_construction(self) -> None:
        window = MainWindow()

        self.assertEqual(window.settings_panel.storage_label.text(), "正在统计占用…")

        window._refresh_storage_usage()

        self.assertIn("已存", window.settings_panel.storage_label.text())

    def test_the_device_probe_is_not_done_during_construction(self) -> None:
        with patch.object(
            main_window, "DeviceProbe", side_effect=AssertionError("构造期不该探测设备")
        ):
            window = MainWindow()

        self.assertEqual(window.source_panel.device_combo.count(), 0)
        self.assertFalse(window.source_panel.open_btn.isEnabled())
        self.assertIn("正在检测推理设备", window.source_panel.status_label.text())

    def test_the_probe_result_fills_the_controls_and_unlocks_open(self) -> None:
        window = MainWindow()

        window._apply_device_options([("cpu", "CPU"), ("cuda:0", "GPU 0")])

        self.assertEqual(window.source_panel.device_combo.count(), 2)
        self.assertTrue(window.source_panel.open_btn.isEnabled())
        self.assertIn("1 个可用 GPU", window.source_panel.status_label.text())

    def test_an_unavailable_saved_device_falls_back_and_is_persisted(self) -> None:
        """上次选的 GPU 这次不在（换机器、驱动掉了）：切回 CPU 并写回配置。"""
        (self.base / "config.json").write_text(
            '{"version": 1, "inference_device": "cuda:0", "zones": []}', encoding="utf-8"
        )
        window = MainWindow()

        window._apply_device_options([("cpu", "CPU")])

        self.assertEqual(window.config.inference_device, "cpu")
        self.assertEqual(window.source_panel.selected_device(), "cpu")
        self.assertIn("已切换到 CPU", window.source_panel.status_label.text())


if __name__ == "__main__":
    unittest.main()
