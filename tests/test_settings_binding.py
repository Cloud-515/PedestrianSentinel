"""声明式绑定表：字段名必须真的是 AppConfig 的字段。

加一个设置项现在只动两处（字段本身 + 绑定表里的一行），代价是绑定表里的名字写错
不会有任何反馈：``store`` 用的是裸 ``setattr``，会把值写到一个谁也不认识的新属性上，
``to_dict`` 里根本没有它 —— 表现正是这张表本来要消灭的「界面存了但重启就丢」。
所以这里把名字钉住，写错在 CI 上就红。
"""

from __future__ import annotations

import dataclasses
import logging
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QSpinBox,
)

from main_window import SettingsPanel  # noqa: E402
from models import AppConfig  # noqa: E402
from settings_binding import (  # noqa: E402
    SettingsBinder,
    checkbox_field,
    combo_field,
    line_edit_field,
    spinbox_field,
)


class BindingNameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_every_binding_name_is_a_real_config_field(self) -> None:
        known = {field.name for field in dataclasses.fields(AppConfig)}

        for binding in SettingsPanel().field_bindings():
            with self.subTest(field=binding.name):
                self.assertIn(
                    binding.name,
                    known,
                    f"绑定表里的 {binding.name!r} 不是 AppConfig 的字段："
                    "写错名字不会报错，只会静默地存不进配置",
                )


class BindingRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _binder(self) -> tuple[SettingsBinder, QCheckBox, QSpinBox, QLineEdit, QComboBox]:
        checkbox = QCheckBox()
        spinbox = QSpinBox()
        spinbox.setRange(0, 3650)
        edit = QLineEdit()
        combo = QComboBox()
        combo.addItem("通用 JSON", "generic")
        combo.addItem("企业微信 / 钉钉文本", "text_bot")
        binder = SettingsBinder(
            [
                checkbox_field("screenshot_retention_enabled", checkbox),
                spinbox_field("screenshot_retention_days", spinbox),
                line_edit_field("notification_url", edit),
                combo_field("notification_format", combo),
            ]
        )
        return binder, checkbox, spinbox, edit, combo

    def test_config_values_reach_the_controls_and_come_back(self) -> None:
        binder, checkbox, spinbox, edit, combo = self._binder()
        source = AppConfig(
            screenshot_retention_enabled=True,
            screenshot_retention_days=7,
            notification_url="https://example.com/hook",
            notification_format="text_bot",
        )

        binder.load(source)

        self.assertTrue(checkbox.isChecked())
        self.assertEqual(spinbox.value(), 7)
        self.assertEqual(edit.text(), "https://example.com/hook")
        self.assertEqual(combo.currentData(), "text_bot")

        target = AppConfig()
        binder.store(target)

        # 回写到的必须是真字段：这也正是名字写错时那条断言要挡住的东西。
        self.assertEqual(target.screenshot_retention_days, 7)
        self.assertEqual(target.notification_format, "text_bot")
        self.assertTrue(target.screenshot_retention_enabled)
        self.assertEqual(target.to_dict()["notification_url"], "https://example.com/hook")

    def test_loading_into_controls_does_not_look_like_a_user_edit(self) -> None:
        """载入时改控件不能发信号：否则每启动一次都会触发一轮「设置变了」的连锁反应。"""
        checkbox = QCheckBox()
        edits: list[bool] = []
        checkbox.toggled.connect(edits.append)

        checkbox_field("screenshot_retention_enabled", checkbox).write(True)

        self.assertTrue(checkbox.isChecked())
        self.assertEqual(edits, [])

    def test_a_value_outside_the_list_falls_back_and_says_so(self) -> None:
        """配置里的值不在列表里时控件只能落到第一项，而回写会把它写回配置。

        这种改写必须留痕：枚举字段在 models 那边本该用 _coerce_choice 挡住列表外的值，
        真溜进来了也要能从日志里看出来「值是什么时候被换掉的」。
        """
        _, _, _, _, combo = self._binder()

        with self.assertLogs("settings_binding", level=logging.WARNING) as captured:
            combo_field("notification_format", combo).write("不认识的格式")

        self.assertEqual(combo.currentData(), "generic")
        self.assertTrue(
            any("notification_format" in message for message in captured.output),
            captured.output,
        )


if __name__ == "__main__":
    unittest.main()
