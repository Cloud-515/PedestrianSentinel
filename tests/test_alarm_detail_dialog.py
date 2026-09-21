"""报警详情弹窗：截图预览与"双击打开原图"。

这里钉住的契约：双击交给系统默认程序时，交出去的必须是**原图文件的真实路径**
（而不是预览、不是相对路径），而且只有在图片真的能打开时才给这个入口 —— 图片不可用时
双击不该有任何反应。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel

import main_window
from main_window import AlarmDetailDialog, PreviewImageLabel
from models import AlarmEvent


class PreviewImageLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_double_click_emits_the_path(self) -> None:
        label = PreviewImageLabel("D:/events/screenshots/x_alarm.jpg")
        received: list[str] = []
        label.activated.connect(received.append)

        label.mouseDoubleClickEvent(None)

        self.assertEqual(received, ["D:/events/screenshots/x_alarm.jpg"])

    def test_label_looks_clickable(self) -> None:
        label = PreviewImageLabel("D:/x.jpg")

        self.assertIn("双击", label.toolTip())
        self.assertEqual(label.cursor().shape(), Qt.CursorShape.PointingHandCursor)


class AlarmDetailDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.screenshot_dir = self.root / "screenshots"
        self.screenshot_dir.mkdir(parents=True)
        self.entry = self.screenshot_dir / "s1_entry.jpg"
        self.alarm = self.screenshot_dir / "s1_alarm.jpg"
        for path, level in ((self.entry, 90), (self.alarm, 120)):
            self.assertTrue(
                cv2.imwrite(str(path), np.full((40, 60, 3), level, dtype=np.uint8))
            )

    def event(self, *, entry_path: str = "", alarm_path: str) -> AlarmEvent:
        return AlarmEvent(
            source="rtsp://camera/live",
            zone_name="北侧入口",
            track_id="12",
            entered_at_seconds=1.0,
            alarm_at_seconds=3.0,
            wall_time="2026-09-21 10:00:00",
            operation_mode="monitor",
            entry_screenshot_path=entry_path,
            alarm_screenshot_path=alarm_path,
        )

    def previews(self, dialog: AlarmDetailDialog) -> list[PreviewImageLabel]:
        return dialog.findChildren(PreviewImageLabel)

    def test_preview_is_double_clickable_and_opens_the_original(self) -> None:
        """双击交出去的必须是原图路径 —— 不是预览图，也不是相对路径。"""
        dialog = AlarmDetailDialog(
            self.event(
                entry_path="screenshots/s1_entry.jpg",
                alarm_path="screenshots/s1_alarm.jpg",
            ),
            # 记录里存的是相对路径，解析器负责还原成实际文件。
            resolver=lambda path: self.root / path if path else None,
        )

        previews = self.previews(dialog)
        self.assertEqual(len(previews), 2, "进入与报警两张取证图都该可双击")

        with patch.object(main_window.QDesktopServices, "openUrl", return_value=True) as open_url:
            for preview in previews:
                preview.mouseDoubleClickEvent(None)

        opened = {Path(call.args[0].toLocalFile()) for call in open_url.call_args_list}
        self.assertEqual(opened, {self.entry, self.alarm})
        self.assertTrue(all(call.args[0].isLocalFile() for call in open_url.call_args_list))

    def test_unavailable_screenshot_has_no_double_click_entry(self) -> None:
        """图片显示不出来时不该有可双击的预览 —— 点了没反应比没有入口更让人困惑。"""
        dialog = AlarmDetailDialog(
            self.event(alarm_path="screenshots/已经没了.jpg"),
            resolver=lambda path: None,
        )

        self.assertEqual(self.previews(dialog), [])
        texts = [label.text() for label in dialog.findChildren(QLabel)]
        self.assertTrue(any("不可用" in text for text in texts), texts)

    def test_open_failure_is_reported_instead_of_silent(self) -> None:
        """系统里没有能打开它的程序时要说一声，否则双击没反应像是程序坏了。"""
        dialog = AlarmDetailDialog(
            self.event(alarm_path=str(self.alarm)), resolver=None
        )

        with (
            patch.object(main_window.QDesktopServices, "openUrl", return_value=False),
            patch.object(main_window.QMessageBox, "warning") as warning,
        ):
            self.previews(dialog)[0].mouseDoubleClickEvent(None)

        warning.assert_called_once()

    def test_without_a_resolver_the_direct_path_still_works(self) -> None:
        dialog = AlarmDetailDialog(
            self.event(alarm_path=str(self.alarm)), resolver=None
        )

        self.assertTrue(self.previews(dialog))


if __name__ == "__main__":
    unittest.main()
