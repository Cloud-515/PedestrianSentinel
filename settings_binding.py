"""把配置字段和控件绑成一张表。

加一个设置项原本要改五处：``AppConfig`` 的字段、``from_dict``、``to_dict``、面板
的载入、面板的保存。设置项长到第三组（留存、通知）时，这种改法开始明显地碍事 ——
漏掉任何一处都是「界面存了但重启就丢」这类不好查的问题。

这里把「字段名 ↔ 控件」声明成一个列表，载入与保存各走一遍。代价是多了一层间接，
换来的是加一个设置只动两处：字段本身，和绑定表里的一行。

只用于**简单的值绑定**（勾选框、输入框、下拉框）。带副作用的控件不进来 —— 比如
推理设备下拉框要处理「上次选的设备这次不可用」，播放控件要通知检测线程，那些逻辑
属于它们各自的代码，硬塞进绑定表只会把副作用藏进声明里。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QSpinBox, QWidget


def _silently(widget: QWidget, action: Callable[[], None]) -> None:
    """改控件但不发信号。

    载入配置时用：值是我们自己写进去的，不该被当成「用户改了这一项」，否则每启动
    一次都会触发一轮「设置变了」的连锁反应（包括一次多余的配置落盘）。
    """
    previous = widget.blockSignals(True)
    try:
        action()
    finally:
        widget.blockSignals(previous)


@dataclass(frozen=True)
class FieldBinding:
    """一个配置字段与一个控件的双向绑定。

    ``read`` 从控件取值，``write`` 把值写回控件（要静默）。
    """

    name: str
    read: Callable[[], object]
    write: Callable[[object], None]


def checkbox_field(name: str, widget: QCheckBox) -> FieldBinding:
    return FieldBinding(
        name=name,
        read=widget.isChecked,
        write=lambda value: _silently(widget, lambda: widget.setChecked(bool(value))),
    )


def spinbox_field(name: str, widget: QSpinBox) -> FieldBinding:
    return FieldBinding(
        name=name,
        read=widget.value,
        write=lambda value: _silently(widget, lambda: widget.setValue(int(value))),
    )


def line_edit_field(name: str, widget: QLineEdit) -> FieldBinding:
    return FieldBinding(
        name=name,
        read=lambda: widget.text().strip(),
        write=lambda value: _silently(widget, lambda: widget.setText(str(value))),
    )


def combo_field(name: str, widget: QComboBox) -> FieldBinding:
    """下拉框绑的是 ``itemData`` 而不是显示文本 —— 界面文案可以随时改，配置值不行。"""

    def read() -> object:
        data = widget.currentData()
        return data if data is not None else ""

    def write(value: object) -> None:
        index = widget.findData(value)
        _silently(widget, lambda: widget.setCurrentIndex(max(0, index)))

    return FieldBinding(name=name, read=read, write=write)


class SettingsBinder:
    def __init__(self, bindings: Sequence[FieldBinding]) -> None:
        self._bindings = list(bindings)

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(binding.name for binding in self._bindings)

    def load(self, source: object) -> None:
        """把配置里的值写进控件。"""
        for binding in self._bindings:
            bound = getattr(source, binding.name, None)
            if bound is None:
                # 字段缺失（或值为 None）时不动控件：控件上本来就有合理的默认值，
                # 把 None 写进去只会让界面出现空白或 0。
                continue
            binding.write(bound)

    def store(self, target: object) -> None:
        """把控件的值写回配置。"""
        for binding in self._bindings:
            setattr(target, binding.name, binding.read())
