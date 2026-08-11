from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from main_window import SourcePanel


class SourcePanelDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_restores_available_device_by_value(self) -> None:
        panel = SourcePanel()
        found = panel.set_device_options(
            [("cpu", "CPU"), ("cuda:0", "GPU 0: Example")],
            "cuda:0",
        )
        self.assertTrue(found)
        self.assertEqual(panel.selected_device(), "cuda:0")

    def test_unavailable_device_falls_back_to_cpu(self) -> None:
        panel = SourcePanel()
        found = panel.set_device_options([("cpu", "CPU")], "cuda:9")
        self.assertFalse(found)
        self.assertEqual(panel.selected_device(), "cpu")

    def test_selector_is_locked_while_running(self) -> None:
        panel = SourcePanel()
        panel.set_device_options([("cpu", "CPU")], "cpu")
        panel.set_running(True)
        self.assertFalse(panel.device_combo.isEnabled())
        panel.set_running(False)
        self.assertTrue(panel.device_combo.isEnabled())


if __name__ == "__main__":
    unittest.main()
