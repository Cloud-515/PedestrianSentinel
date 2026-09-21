"""右侧面板的宽度：一条长状态不能把整个面板撑宽。

现场问题：点"打开"之后状态栏变成"检测运行中：CPU 低功耗模式（OpenVINO INT8，512，
每 4 帧检测），推理设备: cpu"，右侧面板所有行的右端就被切在同一条线上 —— 删除按钮、
配置组状态、颜色值都缺一截。

原因是 QLabel 不换行时，它的最小宽度就是整行文字的宽度，而面板的最小宽度取各行里最大
的那个。所以这里钉住的契约是：**设一条很长的状态之后，面板的最小宽度不该变大**。
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from main_window import SettingsPanel, SourcePanel

LONG_STATUS = (
    "检测运行中：CPU 低功耗模式（OpenVINO INT8，512，每 4 帧检测），推理设备: cpu"
)


class SourcePanelWidthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_a_long_status_does_not_widen_the_panel(self) -> None:
        panel = SourcePanel()
        before = panel.minimumSizeHint().width()

        panel.set_status(LONG_STATUS)

        self.assertLessEqual(
            panel.minimumSizeHint().width(),
            before + 4,
            "长状态把面板撑宽了 —— 面板变宽后超出视口的部分会被裁掉",
        )

    def test_status_wraps_and_keeps_the_full_text_in_a_tooltip(self) -> None:
        panel = SourcePanel()

        panel.set_status(LONG_STATUS)

        self.assertTrue(panel.status_label.wordWrap())
        # 换行之后可能折成几行、也可能被截断，完整内容要能在悬停时读到。
        self.assertEqual(panel.status_label.toolTip(), LONG_STATUS)
        self.assertEqual(panel.status_label.text(), LONG_STATUS)

    def test_status_label_does_not_demand_width(self) -> None:
        """宽度上告诉布局"我不影响你"，面板窄了就多折几行，而不是把面板撑宽。"""
        panel = SourcePanel()

        self.assertEqual(
            panel.status_label.sizePolicy().horizontalPolicy(),
            panel.status_label.sizePolicy().Policy.Ignored,
        )


class SettingsPanelWidthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_long_notification_and_storage_text_do_not_widen_the_panel(self) -> None:
        panel = SettingsPanel()
        before = panel.minimumSizeHint().width()

        panel.set_notification_status(
            "已启用 → https://example.com/very/long/webhook/path/for/alerts（附带截图）",
            ok=True,
        )
        panel.set_storage_usage(
            "已存 12345 张 · 1.23 GB（容量上限 2048 MB，保留 15 天）"
        )

        self.assertLessEqual(panel.minimumSizeHint().width(), before + 4)


if __name__ == "__main__":
    unittest.main()
