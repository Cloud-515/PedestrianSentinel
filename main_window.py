from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QInputDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import app_paths
from alarm_service import EventStore
from config_store import ConfigStore
from compute_devices import enumerate_inference_devices
from detection_worker import DetectionWorker
from inference_profiles import resolve_inference_policy
from models import (
    DEFAULT_RETENTION_DAYS,
    DEFAULT_RETENTION_MB,
    MAX_RETENTION_DAYS,
    MAX_RETENTION_MB,
    NOTIFICATION_FORMAT_GENERIC,
    NOTIFICATION_FORMAT_TEXT_BOT,
    AlarmEvent,
    AppConfig,
    ZoneDefinition,
    ZoneProfile,
)
from notifications import (
    NotificationDispatcher,
    NotificationResult,
    NotificationSettings,
    build_job,
    synthetic_event,
)
from profile_store import ProfileStore
from retention import RetentionPolicy, directory_usage, format_size, prune_screenshots
from settings_binding import (
    FieldBinding,
    SettingsBinder,
    checkbox_field,
    combo_field,
    line_edit_field,
    spinbox_field,
)
from source_history import add_history_entry
from video_source import VideoSourceSpec
from video_widget import VideoWidget

logger = logging.getLogger(__name__)

APP_DIR = app_paths.APP_DIR
CONFIG_PATH = app_paths.data("config.json")
EVENTS_DIR = app_paths.data("events")
PROFILES_DIR = app_paths.data("profiles")


# ---------------------------------------------------------------------------
# 右侧面板：视频源区
# ---------------------------------------------------------------------------

class EditableSourceComboBox(QComboBox):
    def text(self) -> str:
        return self.currentText()

    def setText(self, value: str) -> None:
        self.setEditText(value)

    def placeholderText(self) -> str:
        return self.lineEdit().placeholderText() if self.lineEdit() else ""

    def setPlaceholderText(self, value: str) -> None:
        if self.lineEdit() is not None:
            self.lineEdit().setPlaceholderText(value)


class SourcePanel(QGroupBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("视频源", parent)
        layout = QVBoxLayout(self)

        # 来源输入行
        src_row = QHBoxLayout()
        src_row.setContentsMargins(0, 0, 0, 0)
        src_row.setSpacing(6)
        self.source_label = QLabel("监控来源:")
        self.source_label.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Preferred,
        )
        self.source_edit = EditableSourceComboBox()
        self.source_edit.setEditable(True)
        self.source_edit.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.source_edit.setEditText("0")
        self.source_edit.setPlaceholderText("摄像头索引或实时流地址")
        self.browse_btn = QPushButton("…")
        self.browse_btn.setFixedWidth(28)
        self.browse_btn.setToolTip("选择视频文件")
        self.browse_btn.clicked.connect(self._browse_file)
        src_row.addWidget(self.source_label, 0, Qt.AlignmentFlag.AlignVCenter)
        src_row.addWidget(self.source_edit, 1)
        src_row.addWidget(self.browse_btn, 0)
        layout.addLayout(src_row)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("类型:"))
        self.camera_radio = QRadioButton("摄像头")
        self.file_radio = QRadioButton("文件")
        self.camera_radio.setChecked(True)
        self.source_type_group = QButtonGroup(self)
        self.source_type_group.addButton(self.camera_radio)
        self.source_type_group.addButton(self.file_radio)
        type_row.addWidget(self.camera_radio)
        type_row.addWidget(self.file_radio)
        type_row.addStretch()
        layout.addLayout(type_row)

        # 模式由设置页控制，来源类别会随模式自动切换。
        self.test_mode_cb = QCheckBox("视频模式")
        self.test_mode_cb.setVisible(False)
        layout.addWidget(self.test_mode_cb)

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("推理设备:"))
        self.device_combo = QComboBox()
        self.device_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        device_row.addWidget(self.device_combo, 1)
        layout.addLayout(device_row)

        self.armed_cb = QCheckBox("警戒检测与报警")
        self.armed_cb.setChecked(True)
        self.armed_cb.setEnabled(False)
        layout.addWidget(self.armed_cb)

        # 打开 / 停止
        btn_row = QHBoxLayout()
        self.open_btn = QPushButton("▶ 打开")
        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #AAB4BE; font-size: 11px;")
        layout.addWidget(self.status_label)

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有文件 (*)"
        )
        if path:
            self.source_edit.setText(path)

    def get_source(self) -> str:
        return self.source_edit.text().strip()

    def refresh_source_history(self, history: list[str]) -> None:
        # 刷新候选项时保留用户正在编辑的输入。
        current_text = self.source_edit.currentText()
        self.source_edit.blockSignals(True)
        self.source_edit.clear()
        self.source_edit.addItems(history)
        self.source_edit.setEditText(current_text)
        self.source_edit.blockSignals(False)

    def set_operation_mode(self, operation_mode: str) -> None:
        is_video = operation_mode == "video"
        self.file_radio.setChecked(is_video)
        self.camera_radio.setChecked(not is_video)
        self.camera_radio.setVisible(not is_video)
        self.file_radio.setVisible(is_video)
        self.test_mode_cb.setChecked(is_video)
        self.source_label.setText("视频源:" if is_video else "监控来源:")
        self.source_edit.setPlaceholderText(
            "选择本地视频文件" if is_video else "摄像头索引或实时流地址"
        )
        self.browse_btn.setVisible(is_video)

    def set_device_options(
        self,
        options: list[tuple[str, str]],
        selected: str,
    ) -> bool:
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for value, label in options:
            self.device_combo.addItem(label, value)
        index = self.device_combo.findData(selected)
        found = index >= 0
        self.device_combo.setCurrentIndex(index if found else 0)
        self.device_combo.blockSignals(False)
        return found

    def selected_device(self) -> str:
        value = self.device_combo.currentData()
        return str(value) if value else "cpu"

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_running(self, running: bool) -> None:
        self.open_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.source_edit.setEnabled(not running)
        self.browse_btn.setEnabled(not running)
        self.camera_radio.setEnabled(not running)
        self.file_radio.setEnabled(not running)
        self.test_mode_cb.setEnabled(not running)
        self.device_combo.setEnabled(not running)
        if running:
            self.armed_cb.setEnabled(True)
        else:
            self.armed_cb.blockSignals(True)
            self.armed_cb.setChecked(True)
            self.armed_cb.blockSignals(False)
            self.armed_cb.setEnabled(False)


# ---------------------------------------------------------------------------
# 右侧面板：区域管理区
# ---------------------------------------------------------------------------

class SettingsPanel(QGroupBox):
    mode_changed = Signal(str)
    cpu_low_power_changed = Signal(bool)
    retention_changed = Signal(int, int)
    prune_requested = Signal()
    notification_changed = Signal()
    notification_test_requested = Signal()

    def __init__(self) -> None:
        super().__init__("设置")
        layout = QVBoxLayout(self)
        self.video_radio = QRadioButton("视频模式")
        self.monitor_radio = QRadioButton("监控模式")
        self.monitor_radio.setChecked(True)
        self.cpu_low_power_cb = QCheckBox("CPU 低功耗模式")
        self.cpu_low_power_cb.setToolTip(
            "仅在 CPU 推理时使用 OpenVINO INT8、512 输入和每 4 帧检测。"
        )
        layout.addWidget(self.video_radio)
        layout.addWidget(self.monitor_radio)
        layout.addWidget(self.cpu_low_power_cb)
        layout.addWidget(self._build_retention_box())
        layout.addWidget(self._build_notification_box())
        layout.addStretch()
        self.back_btn = QPushButton("返回主页")
        layout.addWidget(self.back_btn)
        self.video_radio.toggled.connect(self._emit_mode)
        self.cpu_low_power_cb.toggled.connect(self.cpu_low_power_changed)

    def field_bindings(self) -> list[FieldBinding]:
        """本面板里那些「纯粹的设置项」的绑定表。

        仅限没有副作用的控件；模式单选、推理设备、播放控件各有自己的逻辑，不在此列
        （理由见 settings_binding 模块的说明）。
        """
        return [
            checkbox_field("cpu_low_power_preset", self.cpu_low_power_cb),
            checkbox_field("screenshot_retention_enabled", self.retention_enabled_cb),
            spinbox_field("screenshot_retention_days", self.retention_days_spin),
            spinbox_field("screenshot_retention_mb", self.retention_mb_spin),
            checkbox_field("notification_enabled", self.notification_enabled_cb),
            line_edit_field("notification_url", self.notification_url_edit),
            combo_field("notification_format", self.notification_format_combo),
            checkbox_field(
                "notification_include_screenshot", self.notification_screenshot_cb
            ),
            checkbox_field("notification_in_video_mode", self.notification_video_cb),
        ]

    def _build_notification_box(self) -> QGroupBox:
        box = QGroupBox("远程通知")
        box.setToolTip(
            "报警时向下面的地址 POST 一条 JSON。默认关闭。\n"
            "发出的内容包括区域名称、目标 ID、时间与视频源；勾选「附带取证截图」后\n"
            "还会把那一帧的画面（base64）一并发出，请注意接收方能看到什么。"
        )
        layout = QVBoxLayout(box)

        self.notification_enabled_cb = QCheckBox("报警时发送远程通知")
        layout.addWidget(self.notification_enabled_cb)

        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("地址:"))
        self.notification_url_edit = QLineEdit()
        self.notification_url_edit.setPlaceholderText("https://…（群机器人地址或你自己的接口）")
        url_row.addWidget(self.notification_url_edit, 1)
        layout.addLayout(url_row)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("格式:"))
        self.notification_format_combo = QComboBox()
        for value, label in (
            (NOTIFICATION_FORMAT_GENERIC, "通用 JSON"),
            (NOTIFICATION_FORMAT_TEXT_BOT, "企业微信 / 钉钉文本"),
        ):
            self.notification_format_combo.addItem(label, value)
        self.notification_format_combo.setToolTip(
            "通用 JSON：本程序自己的字段（text/event/screenshot_base64）。\n"
            "企业微信 / 钉钉文本：{\"msgtype\":\"text\",\"text\":{\"content\":…}}，"
            "这种机器人只收文本，附不了图。"
        )
        format_row.addWidget(self.notification_format_combo, 1)
        layout.addLayout(format_row)

        self.notification_screenshot_cb = QCheckBox("附带取证截图（base64，≤1 MB）")
        layout.addWidget(self.notification_screenshot_cb)

        self.notification_video_cb = QCheckBox("视频模式（测试播放）也发送")
        self.notification_video_cb.setToolTip(
            "默认不勾选：拿一段视频试跑时，里面的报警不该往群里刷屏。\n"
            "勾选后视频模式里的报警也会发出，适合现场演示完整链路。"
        )
        layout.addWidget(self.notification_video_cb)

        self.notification_status_label = QLabel("未配置")
        self.notification_status_label.setStyleSheet("color: #7A8A99; font-size: 11px;")
        self.notification_status_label.setWordWrap(True)
        layout.addWidget(self.notification_status_label)

        self.notification_test_btn = QPushButton("发送测试")
        self.notification_test_btn.setToolTip("立刻按当前设置发一条测试通知，不写报警记录。")
        self.notification_test_btn.clicked.connect(self.notification_test_requested)
        layout.addWidget(self.notification_test_btn)

        self.notification_enabled_cb.toggled.connect(self._emit_notification)
        self.notification_url_edit.textChanged.connect(self._emit_notification)
        self.notification_format_combo.currentIndexChanged.connect(self._emit_notification)
        self.notification_screenshot_cb.toggled.connect(self._emit_notification)
        self.notification_video_cb.toggled.connect(self._emit_notification)
        return box

    def _emit_notification(self, value: object = None) -> None:
        del value
        self.notification_changed.emit()

    def set_notification_status(self, text: str, *, ok: bool | None = None) -> None:
        color = {True: "#4CAF50", False: "#E53935", None: "#7A8A99"}[ok]
        self.notification_status_label.setStyleSheet(f"color: {color}; font-size: 11px;")
        self.notification_status_label.setText(text)

    def set_notification_controls_enabled(self, enabled: bool) -> None:
        # 地址与格式在运行中也能改：改完下一条报警就用新设置，不必停下检测。
        self.notification_enabled_cb.setEnabled(enabled)
        self.notification_url_edit.setEnabled(enabled)
        self.notification_format_combo.setEnabled(enabled)
        self.notification_screenshot_cb.setEnabled(enabled)
        self.notification_test_btn.setEnabled(enabled)

    def _build_retention_box(self) -> QGroupBox:
        box = QGroupBox("取证留存")
        box.setToolTip(
            "默认关闭：不勾选就不会自动删除任何文件。\n"
            "勾选后按下面的规则清理报警截图；报警记录本身一直保留（它是纯文本，体积可以忽略）。\n"
            "天数或上限填 0 表示那条规则不生效。\n"
            "更早的记录仍会显示在报警记录表里，只是详情中的截图会缺失。"
        )
        layout = QVBoxLayout(box)

        # 删除类功能必须有显式开关，而且默认关着 —— 程序不该在用户还没看过设置的时候
        # 就动他的取证材料。
        self.retention_enabled_cb = QCheckBox("自动清理过期截图")
        self.retention_enabled_cb.setToolTip(
            "勾选后才会按下面的规则自动清理；不勾选时连「立即清理」也不会删东西。"
        )
        layout.addWidget(self.retention_enabled_cb)

        days_row = QHBoxLayout()
        days_row.addWidget(QLabel("截图保留:"))
        self.retention_days_spin = QSpinBox()
        self.retention_days_spin.setRange(0, MAX_RETENTION_DAYS)
        self.retention_days_spin.setSuffix(" 天")
        self.retention_days_spin.setValue(DEFAULT_RETENTION_DAYS)
        self.retention_days_spin.setToolTip("早于这个天数的截图会被清理；0 表示不按天数清理。")
        days_row.addWidget(self.retention_days_spin, 1)
        layout.addLayout(days_row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("容量上限:"))
        self.retention_mb_spin = QSpinBox()
        self.retention_mb_spin.setRange(0, MAX_RETENTION_MB)
        self.retention_mb_spin.setSingleStep(128)
        self.retention_mb_spin.setSuffix(" MB")
        self.retention_mb_spin.setValue(DEFAULT_RETENTION_MB)
        self.retention_mb_spin.setToolTip(
            "截图目录超过这个体积时，从最旧的开始清理；0 表示不限。"
        )
        size_row.addWidget(self.retention_mb_spin, 1)
        layout.addLayout(size_row)

        self.storage_label = QLabel("正在统计占用…")
        self.storage_label.setStyleSheet("color: #7A8A99; font-size: 11px;")
        self.storage_label.setWordWrap(True)
        layout.addWidget(self.storage_label)

        self.prune_btn = QPushButton("立即清理")
        self.prune_btn.setToolTip("按上面的规则清理一次，并刷新占用统计。")
        self.prune_btn.clicked.connect(self.prune_requested)
        layout.addWidget(self.prune_btn)

        self.retention_enabled_cb.toggled.connect(self._on_retention_toggled)
        self.retention_days_spin.valueChanged.connect(self._emit_retention)
        self.retention_mb_spin.valueChanged.connect(self._emit_retention)
        self._apply_retention_enabled_state()
        return box

    def _on_retention_toggled(self, enabled: bool) -> None:
        del enabled
        self._apply_retention_enabled_state()
        self._emit_retention(0)

    def _apply_retention_enabled_state(self) -> None:
        """开关关着的时候，数值框与「立即清理」都不该可点 —— 免得看起来像能用。"""
        enabled = self.retention_enabled_cb.isChecked()
        self.retention_days_spin.setEnabled(enabled)
        self.retention_mb_spin.setEnabled(enabled)
        self.prune_btn.setEnabled(enabled)

    def _emit_retention(self, value: int) -> None:
        del value
        self.retention_changed.emit(
            self.retention_days_spin.value(), self.retention_mb_spin.value()
        )

    def set_retention(self, enabled: bool, days: int, megabytes: int) -> None:
        for widget, value in (
            (self.retention_enabled_cb, enabled),
            (self.retention_days_spin, days),
            (self.retention_mb_spin, megabytes),
        ):
            widget.blockSignals(True)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            else:
                widget.setValue(int(value))
            widget.blockSignals(False)
        self._apply_retention_enabled_state()

    def retention(self) -> tuple[bool, int, int]:
        return (
            self.retention_enabled_cb.isChecked(),
            self.retention_days_spin.value(),
            self.retention_mb_spin.value(),
        )

    def set_storage_usage(self, text: str) -> None:
        self.storage_label.setText(text)

    def set_retention_controls_enabled(self, enabled: bool) -> None:
        """整组控件随检测运行状态启用/禁用（运行中不给改）。"""
        self.retention_enabled_cb.setEnabled(enabled)
        if enabled:
            self._apply_retention_enabled_state()
        else:
            self.retention_days_spin.setEnabled(False)
            self.retention_mb_spin.setEnabled(False)
            self.prune_btn.setEnabled(False)

    def set_operation_mode(self, operation_mode: str) -> None:
        self.video_radio.setChecked(operation_mode == "video")
        self.monitor_radio.setChecked(operation_mode != "video")

    def set_mode_enabled(self, enabled: bool) -> None:
        self.video_radio.setEnabled(enabled)
        self.monitor_radio.setEnabled(enabled)

    def set_cpu_low_power(self, enabled: bool) -> None:
        self.cpu_low_power_cb.blockSignals(True)
        self.cpu_low_power_cb.setChecked(enabled)
        self.cpu_low_power_cb.blockSignals(False)

    def set_cpu_low_power_available(self, available: bool, reason: str = "") -> None:
        self.cpu_low_power_cb.setEnabled(True)
        self.cpu_low_power_cb.setToolTip(
            reason if not available else "仅在 CPU 推理时使用 OpenVINO INT8、512 输入和每 4 帧检测。"
        )

    def set_cpu_low_power_enabled(self, enabled: bool) -> None:
        self.cpu_low_power_cb.setEnabled(enabled)

    def _emit_mode(self, checked: bool) -> None:
        if checked:
            self.mode_changed.emit("video")
        else:
            self.mode_changed.emit("monitor")


class ProfileRow(QWidget):
    apply_requested = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(name)
        label.setToolTip(name)
        layout.addWidget(label, 1)
        apply_button = QPushButton("应用")
        delete_button = QPushButton("删除")
        apply_button.clicked.connect(lambda: self.apply_requested.emit(self.name))
        delete_button.clicked.connect(lambda: self.delete_requested.emit(self.name))
        layout.addWidget(apply_button)
        layout.addWidget(delete_button)
        self.apply_button = apply_button
        self.delete_button = delete_button

    def set_actions_enabled(self, enabled: bool) -> None:
        self.apply_button.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)


class ZonePanel(QGroupBox):
    color_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("警戒区域", parent)
        layout = QVBoxLayout(self)

        profile_header = QHBoxLayout()
        profile_header.addWidget(QLabel("配置组:"))
        self.profile_new_btn = QPushButton("新建")
        self.profile_save_btn = QPushButton("保存")
        self.profile_save_as_btn = QPushButton("另存为")
        self.profile_status = QLabel("已保存")
        profile_header.addWidget(self.profile_new_btn)
        profile_header.addWidget(self.profile_save_btn)
        profile_header.addWidget(self.profile_save_as_btn)
        profile_header.addWidget(self.profile_status)
        layout.addLayout(profile_header)

        self.profile_list = QListWidget()
        self.profile_list.setMaximumHeight(150)
        layout.addWidget(self.profile_list)

        # 列表
        self.zone_list = QListWidget()
        row_height = max(self.zone_list.sizeHintForRow(0), self.fontMetrics().height() + 8)
        self.zone_list.setFixedHeight(row_height * 5 + 6)
        layout.addWidget(self.zone_list)

        # 新增 / 删除
        list_btn_row = QHBoxLayout()
        self.add_zone_btn = QPushButton("+ 新增区域")
        self.del_zone_btn = QPushButton("- 删除区域")
        list_btn_row.addWidget(self.add_zone_btn)
        list_btn_row.addWidget(self.del_zone_btn)
        layout.addLayout(list_btn_row)

        # 参数区
        param_box = QGroupBox("当前区域参数")
        param_layout = QVBoxLayout(param_box)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("名称:"))
        self.name_edit = QLineEdit()
        name_row.addWidget(self.name_edit)
        param_layout.addLayout(name_row)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("颜色:"))
        self.color_btn = QPushButton()
        self.color_btn.setFixedWidth(48)
        self._set_color_btn("#E53935")
        self.color_btn.clicked.connect(self._pick_color)
        color_row.addWidget(self.color_btn)
        color_row.addStretch()
        param_layout.addLayout(color_row)

        dwell_row = QHBoxLayout()
        dwell_row.addWidget(QLabel("停留阈值(s):"))
        self.dwell_spin = QDoubleSpinBox()
        self.dwell_spin.setRange(0.5, 60.0)
        self.dwell_spin.setSingleStep(0.5)
        self.dwell_spin.setValue(2.0)
        dwell_row.addWidget(self.dwell_spin)
        param_layout.addLayout(dwell_row)

        cooldown_row = QHBoxLayout()
        cooldown_row.addWidget(QLabel("冷却时间(s):"))
        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setRange(1.0, 120.0)
        self.cooldown_spin.setSingleStep(1.0)
        self.cooldown_spin.setValue(10.0)
        cooldown_row.addWidget(self.cooldown_spin)
        param_layout.addLayout(cooldown_row)

        layout.addWidget(param_box)

        # 绘图操作按钮
        draw_row = QHBoxLayout()
        self.clear_poly_btn = QPushButton("清空多边形")
        self.close_poly_btn = QPushButton("闭合多边形")
        draw_row.addWidget(self.clear_poly_btn)
        draw_row.addWidget(self.close_poly_btn)
        layout.addLayout(draw_row)

        hint = QLabel("左键加点 · 拖动顶点 · 右键删点 · 双击闭合")
        hint.setStyleSheet("color: #7A8A99; font-size: 10px;")
        layout.addWidget(hint)

        self._current_color = "#E53935"

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._current_color), self, "选择区域颜色")
        if color.isValid():
            self._current_color = color.name()
            self._set_color_btn(self._current_color)
            self.color_changed.emit(self._current_color)

    def _set_color_btn(self, color: str) -> None:
        self._current_color = color
        self.color_btn.setStyleSheet(
            f"background-color: {color}; border: 1px solid #555; border-radius: 3px;"
        )
        self.color_btn.setText(color)

    def current_color(self) -> str:
        return self._current_color

    def set_color(self, color: str) -> None:
        self._set_color_btn(color)

    def populate(self, zones: list[ZoneDefinition], active: ZoneDefinition | None) -> None:
        self.zone_list.blockSignals(True)
        self.zone_list.clear()
        for zone in zones:
            item = QListWidgetItem(zone.name)
            item.setData(Qt.ItemDataRole.UserRole, zone)
            self.zone_list.addItem(item)
        self.zone_list.blockSignals(False)
        if active is not None:
            self._select_zone(active)

    def _select_zone(self, zone: ZoneDefinition) -> None:
        for i in range(self.zone_list.count()):
            item = self.zone_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) is zone:
                self.zone_list.setCurrentRow(i)
                break

    def load_zone_params(self, zone: ZoneDefinition) -> None:
        self.name_edit.blockSignals(True)
        self.dwell_spin.blockSignals(True)
        self.cooldown_spin.blockSignals(True)
        self.name_edit.setText(zone.name)
        self.dwell_spin.setValue(zone.dwell_seconds)
        self.cooldown_spin.setValue(zone.cooldown_seconds)
        self.set_color(zone.color)
        self.name_edit.blockSignals(False)
        self.dwell_spin.blockSignals(False)
        self.cooldown_spin.blockSignals(False)

    def set_params_enabled(self, enabled: bool) -> None:
        self.name_edit.setEnabled(enabled)
        self.color_btn.setEnabled(enabled)
        self.dwell_spin.setEnabled(enabled)
        self.cooldown_spin.setEnabled(enabled)
        self.del_zone_btn.setEnabled(enabled)
        self.clear_poly_btn.setEnabled(enabled)
        self.close_poly_btn.setEnabled(enabled)


# ---------------------------------------------------------------------------
# 右侧面板：测试播放控制区
# ---------------------------------------------------------------------------

class PlaybackPanel(QGroupBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("播放控制（测试模式）", parent)
        self.setEnabled(False)
        layout = QVBoxLayout(self)

        btn_row = QHBoxLayout()
        self.play_pause_btn = QPushButton("⏸ 暂停")
        self.step_btn = QPushButton("⏭ 单帧")
        btn_row.addWidget(self.play_pause_btn)
        btn_row.addWidget(self.step_btn)
        layout.addLayout(btn_row)

        self.loop_cb = QCheckBox("循环播放")
        self.loop_cb.setChecked(True)
        layout.addWidget(self.loop_cb)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("速度:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 16)  # 0.25× ~ 4× (步长 0.25)
        self.speed_slider.setValue(4)      # 默认 1×
        self.speed_label = QLabel("1.00×")
        self.speed_label.setFixedWidth(40)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self.speed_slider)
        speed_row.addWidget(self.speed_label)
        layout.addLayout(speed_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        layout.addWidget(self.progress_bar)
        self.position_label = QLabel("00:00 / 00:00")
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.position_label)

        self._paused = False

    def _on_speed_changed(self, value: int) -> None:
        speed = value * 0.25
        self.speed_label.setText(f"{speed:.2f}×")

    def speed(self) -> float:
        return self.speed_slider.value() * 0.25

    def set_progress(self, ratio: float, duration_seconds: float = 0.0) -> None:
        ratio = max(0.0, min(1.0, ratio))
        self.progress_bar.setValue(int(ratio * 1000))
        current = duration_seconds * ratio
        self.position_label.setText(
            f"{self._format_time(current)} / {self._format_time(duration_seconds)}"
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    def set_file_mode(self, is_file: bool, running: bool) -> None:
        self.setEnabled(is_file)
        self.play_pause_btn.setEnabled(is_file and running)
        self.step_btn.setEnabled(is_file and running)

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self.play_pause_btn.setText("▶ 继续" if paused else "⏸ 暂停")

    def toggle_paused(self) -> bool:
        self._paused = not self._paused
        self.set_paused(self._paused)
        return self._paused


# ---------------------------------------------------------------------------
# 右侧面板：报警记录区
# ---------------------------------------------------------------------------

class AlarmDetailDialog(QDialog):
    def __init__(
        self,
        event: AlarmEvent,
        resolver: Callable[[str], Path | None] | None = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("报警详情")
        self.resize(720, 620)
        layout = QVBoxLayout(self)
        mode_label = "视频模式" if event.operation_mode == "video" else "监控模式"
        details = [
            ("进入时间", event.format_event_time(event.entered_at_seconds, precision=3)),
            ("报警时间", event.format_event_time(event.alarm_at_seconds, precision=3)),
            ("退出时间", event.format_event_time(event.exited_at_seconds, precision=3)),
            ("闯入时长", f"{event.duration_seconds:.2f}s" if event.duration_seconds is not None else "未结算"),
            ("状态", event.status_label),
            ("运行模式", mode_label),
            ("视频源", event.source),
            ("警戒区域", event.zone_name),
            ("目标 ID", event.track_id),
            ("进入取证", event.entry_screenshot_path or "无"),
            ("报警取证", event.alarm_screenshot_path or "无"),
        ]
        for label, value in details:
            layout.addWidget(QLabel(f"{label}: {value}"))
        screenshots = QHBoxLayout()
        for title, screenshot_path in (
            ("进入取证", event.entry_screenshot_path or event.screenshot_path),
            ("报警取证", event.alarm_screenshot_path),
        ):
            screenshot_label = QLabel(f"{title}不可用")
            screenshot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            screenshot_label.setMinimumSize(330, 240)
            resolved = resolver(screenshot_path) if resolver is not None else None
            if resolved is None and screenshot_path:
                # 解析器没给（或没找到）时退回直接按路径打开：相对路径、被移动过的
                # 记录都靠解析器处理，这里只是最后的兜底。
                direct = Path(screenshot_path)
                resolved = direct if direct.is_file() else None
            if resolved is not None:
                pixmap = QPixmap(str(resolved))
                if not pixmap.isNull():
                    screenshot_label.setPixmap(
                        pixmap.scaled(
                            330,
                            300,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                    screenshots.addWidget(screenshot_label)
                    continue
            # 显示不出来时说清是哪一种：本来没报警 / 没写进去 / 文件被清理或删掉。
            reason = EventStore.unavailable_reason(event, screenshot_path)
            screenshot_label.setText(f"{title}不可用\n{reason}")
            screenshot_label.setWordWrap(True)
            screenshots.addWidget(screenshot_label)
        layout.addLayout(screenshots)


class EventPanel(QGroupBox):
    HEADERS = [
        "记录时间", "视频源", "区域", "目标ID", "进入时刻", "报警时刻",
        "退出时刻", "状态", "闯入时长", "进入取证", "报警取证",
    ]

    def __init__(
        self,
        resolver: Callable[[str], Path | None] | None = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__("报警记录", parent)
        # 记录里存的是相对 events 目录的路径，由 EventStore 解析成实际文件
        # （旧记录存的是绝对路径，解析器还负责按文件名兜底找回）。
        self._resolver = resolver
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, len(self.HEADERS))
        self._rows_by_session: dict[str, int] = {}
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableWidget::item:hover { background-color: transparent; }"
            "QTableWidget::item:selected { background-color: rgba(1, 174, 231, 128); }"
        )
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(220)
        self.table.cellDoubleClicked.connect(self._show_details)
        layout.addWidget(self.table)
        self.clear_btn = QPushButton("清空记录")
        layout.addWidget(self.clear_btn)

    def _show_details(self, row: int, column: int) -> None:
        item = self.table.item(row, 0)
        event = item.data(Qt.ItemDataRole.UserRole) if item else None
        if isinstance(event, AlarmEvent):
            AlarmDetailDialog(event, self._resolver, self).exec()

    def append_event(self, event: AlarmEvent) -> None:
        row = self._rows_by_session.get(event.session_id)
        if row is None:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._rows_by_session[event.session_id] = row
        for column, value in enumerate(event.to_row()):
            item = self.table.item(row, column) or QTableWidgetItem()
            item.setText(value)
            item.setToolTip(value)
            if column == 0:
                item.setData(Qt.ItemDataRole.UserRole, event)
            self.table.setItem(row, column, item)
        self.table.scrollToBottom()

    def clear(self) -> None:
        self.table.setRowCount(0)
        self._rows_by_session.clear()


class MainWindow(QMainWindow):
    # 通知结果是从发送线程回调回来的，必须走信号排队到 GUI 线程再碰控件。
    notification_result = Signal(bool, str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("行人警戒区域监控")
        self.resize(1280, 800)

        self.config_store = ConfigStore(CONFIG_PATH)
        self.profile_store = ProfileStore(PROFILES_DIR)
        self.profile_dirty = False
        self.event_store = EventStore(EVENTS_DIR)
        self.config = self._load_config()
        self.zones = self.config.zones
        self.active_zone: ZoneDefinition | None = self.zones[0] if self.zones else None
        self.worker: DetectionWorker | None = None
        self.video_duration = 0.0
        self._alarm_overlay_enabled = False

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self._save_config)

        # 截图留存：一小时扫一次目录。扫描 200 来个文件的耗时在毫秒级，代价可以忽略；
        # 真正的意义是「程序连着跑几个月也不会把磁盘吃满」，而不是精确到点就清。
        self._retention_timer = QTimer(self)
        self._retention_timer.setInterval(60 * 60 * 1000)
        self._retention_timer.timeout.connect(self._run_retention)

        # 占用统计要遍历截图目录，所以不能挂在「任何设置变了」上：在通知地址框里每敲
        # 一个字符都扫一遍，等到目录里几万张图时打字会明显卡顿。防抖到 250 ms，
        # 连点数值框也只扫一次。
        self._usage_timer = QTimer(self)
        self._usage_timer.setSingleShot(True)
        self._usage_timer.setInterval(250)
        self._usage_timer.timeout.connect(self._refresh_storage_usage)

        self.video_widget = VideoWidget()
        self.source_panel = SourcePanel()
        self.settings_panel = SettingsPanel()
        self.zone_panel = ZonePanel()
        self.playback_panel = PlaybackPanel()
        self.event_panel = EventPanel(self.event_store.resolve_screenshot)

        self._settings_binder = SettingsBinder(self.settings_panel.field_bindings())
        self.notification_dispatcher = NotificationDispatcher()
        self.notification_result.connect(self._on_notification_result)

        self._build_layout()
        self._load_controls()
        self._connect_signals()
        self._load_event_history()
        self._log_effective_settings()
        self._refresh_storage_usage()
        self._retention_timer.start()
        # 启动时先让窗口画出来再扫目录：目录很大时这一步不该拖慢启动。放在事件循环
        # 里做还有个好处 —— 自动化测试不跑事件循环，就不会在无准备的情况下动磁盘。
        # 配置读坏而回落默认值时不自动清理（见 _run_retention）。
        QTimer.singleShot(0, self._run_retention)

    def _build_layout(self) -> None:
        self.settings_btn = QPushButton("设置")
        home_content = QWidget()
        home_layout = QVBoxLayout(home_content)
        home_layout.addWidget(self.settings_btn)
        home_layout.addWidget(self.source_panel)
        home_layout.addWidget(self.zone_panel)
        home_layout.addWidget(self.playback_panel)
        home_layout.addWidget(self.event_panel)
        home_layout.addStretch()

        settings_content = QWidget()
        settings_layout = QVBoxLayout(settings_content)
        settings_layout.addWidget(self.settings_panel)
        settings_layout.addStretch()

        self.sidebar_pages = QStackedWidget()
        self.sidebar_pages.addWidget(home_content)
        self.sidebar_pages.addWidget(settings_content)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.sidebar_pages)
        scroll.setMinimumWidth(420)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.video_widget)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([850, 430])
        self.setCentralWidget(splitter)

    def _load_config(self) -> AppConfig:
        try:
            loaded = self.config_store.load()
        except Exception as error:  # noqa: BLE001 - 配置文件坏成什么样都不该让程序起不来
            # 这里刻意捕获所有异常：原来只接 (OSError, ValueError, TypeError)，而
            # 「zones 被改成对象」这类坏数据抛的是 AttributeError，会一路冒到 main()
            # 之外 —— 用户每次启动都只看到「程序出错」，只能自己想到去删 config.json。
            # 解析不了就把坏文件改名留档再走默认配置，别再覆盖它。
            logger.exception("Unable to load configuration")
            self._config_trusted = False
            backup = self.config_store.quarantine()
            hint = f"\n原文件已备份为 {backup.name}。" if backup else ""
            QMessageBox.warning(
                self,
                "配置读取失败",
                f"将使用默认配置。{hint}\n{error}",
            )
            return AppConfig()
        self._config_trusted = True
        return loaded

    def _load_controls(self) -> None:
        self._apply_operation_mode(self.config.operation_mode, persist=False)
        self.source_panel.source_edit.setText(
            self.config.video_source
            if self.config.operation_mode == "video"
            else self.config.monitor_source
        )
        self.source_panel.refresh_source_history(
            self.config.file_history
            if self.config.operation_mode == "video"
            else self.config.camera_history
        )
        device_options = enumerate_inference_devices()
        restored = self.source_panel.set_device_options(
            device_options,
            self.config.inference_device,
        )
        if not restored:
            unavailable = self.config.inference_device
            self.config.inference_device = "cpu"
            self.source_panel.set_status(
                f"已保存的设备 {unavailable} 当前不可用，已切换到 CPU"
            )
            self._schedule_config_save()
        elif len(device_options) == 1:
            self.source_panel.set_status("当前 PyTorch 未检测到 CUDA，仅可使用 CPU")
        else:
            self.source_panel.set_status(
                f"检测到 {len(device_options) - 1} 个可用 GPU"
            )
        self._settings_binder.load(self.config)
        self._update_cpu_low_power_availability()
        self._refresh_notification_status()
        self.playback_panel.loop_cb.setChecked(self.config.loop_playback)
        speed_value = max(1, min(16, round(self.config.playback_speed / 0.25)))
        self.playback_panel.speed_slider.setValue(speed_value)
        if self.config.active_profile and self.profile_store.exists(self.config.active_profile):
            self.zones = self.profile_store.load(self.config.active_profile).zones
            self.active_zone = self.zones[0] if self.zones else None
        self.video_widget.set_zones(self.zones)
        self.video_widget.set_active_zone(self.active_zone)
        self._refresh_zone_panel()
        self._refresh_profiles()
        self._set_profile_dirty(False)
        self._update_playback_enabled()

    def _connect_signals(self) -> None:
        self.source_panel.open_btn.clicked.connect(self.start_detection)
        self.source_panel.stop_btn.clicked.connect(self.stop_detection)
        self.source_panel.armed_cb.toggled.connect(self._set_armed)
        self.settings_btn.clicked.connect(lambda: self.sidebar_pages.setCurrentIndex(1))
        self.settings_panel.back_btn.clicked.connect(
            lambda: self.sidebar_pages.setCurrentIndex(0)
        )
        self.settings_panel.mode_changed.connect(self._apply_operation_mode)
        self.settings_panel.cpu_low_power_changed.connect(self._on_setting_changed)
        self.settings_panel.retention_changed.connect(self._on_retention_changed)
        self.settings_panel.notification_changed.connect(self._on_setting_changed)
        self.settings_panel.prune_requested.connect(self._prune_now)
        self.settings_panel.notification_test_requested.connect(self._send_test_notification)
        self.zone_panel.profile_new_btn.clicked.connect(self._new_profile)
        self.zone_panel.profile_save_btn.clicked.connect(self._save_profile)
        self.zone_panel.profile_save_as_btn.clicked.connect(self._save_profile_as)
        self.source_panel.device_combo.currentIndexChanged.connect(
            self._on_device_changed
        )

        self.zone_panel.zone_list.currentItemChanged.connect(self._on_zone_selected)
        self.zone_panel.add_zone_btn.clicked.connect(self._add_zone)
        self.zone_panel.del_zone_btn.clicked.connect(self._delete_zone)
        self.zone_panel.name_edit.editingFinished.connect(self._update_zone_name)
        self.zone_panel.color_changed.connect(self._update_zone_color)
        self.zone_panel.dwell_spin.valueChanged.connect(self._update_zone_dwell)
        self.zone_panel.cooldown_spin.valueChanged.connect(self._update_zone_cooldown)
        self.zone_panel.clear_poly_btn.clicked.connect(self.video_widget.clear_active_zone)
        self.zone_panel.close_poly_btn.clicked.connect(self._close_polygon)

        self.playback_panel.play_pause_btn.clicked.connect(self._toggle_pause)
        self.playback_panel.step_btn.clicked.connect(self._step_frame)
        self.playback_panel.loop_cb.toggled.connect(self._set_loop_playback)
        self.playback_panel.speed_slider.valueChanged.connect(self._set_playback_speed)

        self.video_widget.zone_changed.connect(self._on_zone_geometry_changed)
        self.video_widget.mapping_changed.connect(self._on_mapping_changed)
        self.video_widget.frame_size_changed.connect(self._on_frame_size_changed)
        self.event_panel.clear_btn.clicked.connect(self._clear_events)

    def _load_event_history(self) -> None:
        self.event_panel.clear()
        try:
            for event in self.event_store.load_recent(
                operation_mode=self.config.operation_mode
            ):
                self.event_panel.append_event(event)
        except OSError as error:
            logger.exception("Unable to load alarm event history")
            self.source_panel.set_status(f"报警记录读取失败: {error}")

    def _set_profile_controls_enabled(self, enabled: bool) -> None:
        self.zone_panel.profile_save_btn.setEnabled(enabled)
        self.zone_panel.profile_save_as_btn.setEnabled(enabled)
        self.zone_panel.profile_new_btn.setEnabled(enabled)
        for index in range(self.zone_panel.profile_list.count()):
            row = self.zone_panel.profile_list.itemWidget(self.zone_panel.profile_list.item(index))
            if row is not None:
                row.set_actions_enabled(enabled)

    def _set_profile_dirty(self, dirty: bool) -> None:
        self.profile_dirty = dirty
        name = self.config.active_profile or "未命名配置组"
        self.zone_panel.profile_status.setText("未保存" if dirty else "已保存")
        self.zone_panel.profile_status.setToolTip(f"当前配置组: {name}")

    def _refresh_profiles(self) -> None:
        current = self.config.active_profile
        summaries = self.profile_store.list_profiles()
        profile_names = [summary.name for summary in summaries]
        if current and current not in profile_names:
            profile_names.append(current)
        self.zone_panel.profile_list.clear()
        for name in profile_names:
            item = QListWidgetItem(self.zone_panel.profile_list)
            row = ProfileRow(name)
            row.apply_requested.connect(self._request_profile_change)
            row.delete_requested.connect(self._delete_profile)
            item.setSizeHint(row.sizeHint())
            self.zone_panel.profile_list.addItem(item)
            self.zone_panel.profile_list.setItemWidget(item, row)
        self.zone_panel.profile_status.setToolTip(
            f"当前配置组: {current or '未命名配置组'}"
        )

    def _refresh_zone_panel(self) -> None:
        self.zone_panel.populate(self.zones, self.active_zone)
        has_active = self.active_zone is not None
        self.zone_panel.set_params_enabled(has_active)
        if self.active_zone is not None:
            self.zone_panel.load_zone_params(self.active_zone)

    def _ask_profile_save(self, message: str) -> str:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("配置组有未保存修改")
        dialog.setText(message)
        save_button = dialog.addButton("保存并继续", QMessageBox.ButtonRole.AcceptRole)
        discard_button = dialog.addButton("放弃修改", QMessageBox.ButtonRole.DestructiveRole)
        # 「取消」按钮只需要出现在对话框里；它不是保存/放弃之外的那种选择，所以不接引用。
        dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is save_button:
            return "save"
        if clicked is discard_button:
            return "discard"
        return "cancel"

    def _request_profile_change(self, name: str) -> None:
        if not name:
            return
        if name == self.config.active_profile:
            if not self.zones:
                QMessageBox.information(self, "配置组为空", "当前配置组尚未添加警戒区域。")
            return
        if self.profile_dirty:
            answer = self._ask_profile_save("是否保存当前修改后再切换配置组？")
            if answer == "cancel":
                self._refresh_profiles()
                return
            if answer == "save" and not self._save_profile():
                self._refresh_profiles()
                return
        self._load_profile(name)

    def _load_profile(self, name: str) -> None:
        if not name or (self.worker is not None and self.worker.isRunning()):
            return
        try:
            profile = self.profile_store.load(name)
        except ValueError as error:
            QMessageBox.warning(self, "加载配置组失败", str(error))
            return
        self.zones = profile.zones
        self.active_zone = self.zones[0] if self.zones else None
        self.config.active_profile = name
        self.video_widget.set_zones(self.zones)
        self.video_widget.set_active_zone(self.active_zone)
        self._refresh_zone_panel()
        self._zones_updated(mark_profile_dirty=False)
        self._set_profile_dirty(False)
        self._schedule_config_save()

    def _save_profile(self) -> bool:
        name = self.config.active_profile.strip()
        if not name:
            name, accepted = QInputDialog.getText(self, "保存配置组", "配置组名称:")
            if not accepted or not name.strip():
                return False
            name = name.strip()
        try:
            self.profile_store.save(ZoneProfile(name=name, zones=self.zones))
        except ValueError as error:
            QMessageBox.warning(self, "保存配置组失败", str(error))
            return False
        self.config.active_profile = name
        self._refresh_profiles()
        self._set_profile_dirty(False)
        self._schedule_config_save()
        self.source_panel.set_status(f"配置组已保存: {name}")
        return True

    def _save_profile_as(self) -> None:
        name, accepted = QInputDialog.getText(self, "配置组另存为", "新配置组名称:")
        if not accepted or not name.strip():
            return
        name = name.strip()
        if self.profile_store.exists(name):
            QMessageBox.warning(self, "另存为失败", "该配置组名称已存在。")
            return
        old_name = self.config.active_profile
        self.config.active_profile = name
        if not self._save_profile():
            self.config.active_profile = old_name

    def _new_profile(self) -> None:
        if self.profile_dirty:
            answer = self._ask_profile_save("是否保存当前修改后新建配置组？")
            if answer == "cancel":
                return
            if answer == "save" and not self._save_profile():
                return
        name, accepted = QInputDialog.getText(self, "新建配置组", "配置组名称:")
        if not accepted or not name.strip():
            return
        name = name.strip()
        if self.profile_store.exists(name):
            QMessageBox.warning(self, "新建失败", "该配置组名称已存在。")
            return
        self.config.active_profile = name
        self.zones = []
        self.active_zone = None
        self.video_widget.set_zones(self.zones)
        self.video_widget.set_active_zone(self.active_zone)
        self._refresh_zone_panel()
        self._refresh_profiles()
        self._set_profile_dirty(True)

    def _delete_profile(self, name: str) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if QMessageBox.question(
            self,
            "删除配置组",
            f"确定删除配置组“{name}”吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            self.profile_store.delete(name)
        except ValueError as error:
            QMessageBox.warning(self, "删除配置组失败", str(error))
            return
        if self.config.active_profile == name:
            self.config.active_profile = ""
            self._set_profile_dirty(False)
        self._refresh_profiles()
        self._schedule_config_save()

    def _on_zone_selected(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        del previous
        self.active_zone = (
            current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        )
        self.video_widget.set_active_zone(self.active_zone)
        self.zone_panel.set_params_enabled(self.active_zone is not None)
        if self.active_zone is not None:
            self.zone_panel.load_zone_params(self.active_zone)

    def _add_zone(self) -> None:
        colors = ["#E53935", "#00A86B", "#1E88E5", "#F9A825", "#8E24AA"]
        zone = ZoneDefinition(
            name=self._unique_zone_name(),
            color=colors[len(self.zones) % len(colors)],
        )
        self.zones.append(zone)
        self.active_zone = zone
        self.video_widget.set_zones(self.zones)
        self.video_widget.set_active_zone(zone)
        self._refresh_zone_panel()
        self._zones_updated()
        self.source_panel.set_status("请在视频画面中左键添加区域顶点")

    def _unique_zone_name(self) -> str:
        existing = {zone.name for zone in self.zones}
        index = 1
        while f"区域 {index}" in existing:
            index += 1
        return f"区域 {index}"

    def _delete_zone(self) -> None:
        if self.active_zone is None:
            return
        answer = QMessageBox.question(
            self,
            "删除区域",
            f"确定删除“{self.active_zone.name}”吗？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.zones.remove(self.active_zone)
        self.active_zone = self.zones[0] if self.zones else None
        self.video_widget.set_zones(self.zones)
        self.video_widget.set_active_zone(self.active_zone)
        self._refresh_zone_panel()
        self._zones_updated()

    def _update_zone_name(self) -> None:
        if self.active_zone is None:
            return
        name = self.zone_panel.name_edit.text().strip()
        duplicate = any(zone is not self.active_zone and zone.name == name for zone in self.zones)
        if not name or duplicate:
            reason = "区域名称不能为空" if not name else "区域名称不能重复"
            QMessageBox.warning(self, "名称无效", reason)
            self.zone_panel.name_edit.setText(self.active_zone.name)
            return
        if name == self.active_zone.name:
            return
        self.active_zone.name = name
        self._refresh_zone_panel()
        self._zones_updated()

    def _update_zone_color(self, color: str) -> None:
        if self.active_zone is None:
            return
        self.active_zone.color = color
        self.video_widget.update()
        self._zones_updated()

    def _update_zone_dwell(self, value: float) -> None:
        if self.active_zone is not None:
            self.active_zone.dwell_seconds = value
            self._zones_updated()

    def _update_zone_cooldown(self, value: float) -> None:
        if self.active_zone is not None:
            self.active_zone.cooldown_seconds = value
            self._zones_updated()

    def _close_polygon(self) -> None:
        if self.active_zone is None:
            return
        if len(self.active_zone.polygon) < 3:
            QMessageBox.information(self, "无法闭合", "至少需要三个顶点才能闭合区域。")
            return
        self.video_widget.close_active_zone()

    def _on_zone_geometry_changed(self, zone: ZoneDefinition) -> None:
        if zone is self.active_zone:
            self._zones_updated()

    def _zones_updated(self, mark_profile_dirty: bool = True) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.set_zones(self.zones)
        if mark_profile_dirty:
            self._set_profile_dirty(True)
        self._schedule_config_save()

    def _apply_operation_mode(self, operation_mode: str, persist: bool = True) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if operation_mode not in {"monitor", "video"}:
            operation_mode = "monitor"
        previous_mode = self.config.operation_mode
        if previous_mode == "video":
            self.config.video_source = self.source_panel.get_source()
        else:
            self.config.monitor_source = self.source_panel.get_source()
        self.config.operation_mode = operation_mode
        self._alarm_overlay_enabled = False
        self.video_widget.clear_alarm()
        self.settings_panel.set_operation_mode(operation_mode)
        self.source_panel.set_operation_mode(operation_mode)
        self.source_panel.source_edit.setText(
            self.config.video_source if operation_mode == "video" else self.config.monitor_source
        )
        self.source_panel.refresh_source_history(
            self.config.file_history if operation_mode == "video" else self.config.camera_history
        )
        self.playback_panel.setVisible(operation_mode == "video")
        self._load_event_history()
        self._update_playback_enabled()
        if persist:
            self._schedule_config_save()

    def _is_file_mode(self) -> bool:
        return self.config.operation_mode == "video"

    def _on_device_changed(self, index: int) -> None:
        del index
        self._update_cpu_low_power_availability()
        self.source_panel.set_status(
            f"推理设备已选择：{self.source_panel.device_combo.currentText()}"
        )
        self._schedule_config_save()

    def _on_setting_changed(self, *ignored: object) -> None:
        """设置页里任何一项变了：同步进内存配置，再走防抖落盘。

        一次性同步整组而不是逐字段处理，是因为绑定表就是这一组 —— 逐字段又要回到
        「加一个设置改五处」的老路，而那正是绑定表要消掉的东西。
        """
        del ignored
        self._settings_binder.store(self.config)
        # 通知状态栏是「当前配置能不能发出去」的实时指示，改完就刷新（它不碰磁盘）。
        self._refresh_notification_status()
        self._schedule_config_save()

    def _on_retention_changed(self, *ignored: object) -> None:
        """留存设置变了：同步配置，并重新统计占用。

        占用统计要遍历截图目录，所以只在这两个留存控件变化时才安排 —— 别的设置项
        变化（比如在地址框里打字）不该触发目录扫描。
        """
        del ignored
        self._on_setting_changed()
        self._usage_timer.start()

    def _log_effective_settings(self) -> None:
        """把这次真正生效的设置记进日志。

        排查现场问题时，「他到底配了什么」和「他装的是哪一版」一样重要，而这两件事
        以前都只能靠问。这里记的是内存里生效的那份配置，不是文件内容 —— 配置读坏而
        回落默认值的情况也就能一眼看出来。
        """
        logger.info(
            "生效设置: 模式=%s 视频源=%s 设备=%s 低功耗=%s 模型=%s "
            "留存=%s天/%sMB 远程通知=%s 配置可信=%s",
            self.config.operation_mode,
            self.source_panel.get_source(),
            self.source_panel.selected_device(),
            self.config.cpu_low_power_preset,
            self.config.model_path,
            self.config.screenshot_retention_days,
            self.config.screenshot_retention_mb,
            (
                f"已启用 → {self.config.notification_url} "
                f"({self.config.notification_format})"
                if self.config.notification_enabled
                else "关闭"
            ),
            getattr(self, "_config_trusted", False),
        )

    def _notification_settings(self) -> NotificationSettings:
        return NotificationSettings(
            enabled=self.config.notification_enabled,
            url=self.config.notification_url,
            payload_format=self.config.notification_format,
            include_screenshot=self.config.notification_include_screenshot,
        )

    def _has_usable_notification(self) -> bool:
        return self._notification_settings().usable

    def _refresh_notification_status(self) -> None:
        """没在发送时，通知状态栏显示当前配置是否可用。"""
        settings = self._notification_settings()
        if not settings.enabled:
            self.settings_panel.set_notification_status("已关闭（不会发出任何数据）")
        elif not settings.usable:
            self.settings_panel.set_notification_status(
                "已启用，但地址为空或不是 http(s) 地址 —— 报警时发不出去",
                ok=False,
            )
        else:
            detail = "附带截图" if settings.include_screenshot else "不带截图"
            if not self.config.notification_in_video_mode:
                detail += "；视频模式不发"
            self.settings_panel.set_notification_status(
                f"已启用 → {settings.url}（{detail}）", ok=True
            )

    def _notify_alarm(self, event: AlarmEvent) -> None:
        settings = self._notification_settings()
        if event.operation_mode == "video" and not self.config.notification_in_video_mode:
            # 视频模式（测试播放）的报警默认不发通知：拿一段视频试跑一遍，群机器人就会
            # 被刷一串「闯入报警」。本地蜂鸣被刷无所谓，往外部广播不一样。判据用事件
            # 自己的运行模式，而不是当前配置的模式 —— 报警来自哪个视频源，它自己最清楚。
            logger.info("视频模式（测试播放）的报警未发送远程通知（可在设置页打开）")
            return
        job = build_job(event, settings)
        if job is None:
            if settings.enabled:
                # 「开了但发不出去」必须说出来。这是最危险的一种失败：报警响了、记录
                # 写了，操作员以为通知也发了，实际上什么都没出去。
                logger.warning(
                    "报警通知未发出：通知已启用但地址不可用（%r）", settings.url
                )
                self.source_panel.set_status(
                    "警报（远程通知未发出：请检查通知地址）"
                )
                self._refresh_notification_status()
            return
        self.notification_dispatcher.notify(job, on_result=self._on_notification_result_async)

    def _on_notification_result_async(self, result: NotificationResult) -> None:
        """在发送线程里被调用：只允许发信号，不能碰控件。"""
        self.notification_result.emit(result.ok, result.message)

    def _on_notification_result(self, ok: bool, message: str) -> None:
        self.source_panel.set_status(f"远程通知：{message}")
        self.settings_panel.set_notification_status(message, ok=ok)

    def _send_test_notification(self) -> None:
        settings = self._notification_settings()
        if not settings.usable:
            self._refresh_notification_status()
            self.source_panel.set_status("远程通知：请先启用并填写 http(s) 地址")
            return
        # 测试通知只在本机内存里造一条假事件，不写 events\、不发声、不进表格。
        job = build_job(synthetic_event(), settings)
        if job is None:
            return
        self.settings_panel.set_notification_status("正在发送测试通知…")
        self.notification_dispatcher.notify(job, on_result=self._on_notification_result_async)

    def _retention_policy(self) -> RetentionPolicy:
        """当前生效的清理策略。开关关着时返回一个「什么都不做」的策略。

        开关是唯一的闸门：自动清理与「立即清理」都走这里，所以关掉之后程序不可能
        删掉任何截图 —— 用户不必去猜「这个按钮到底会不会真的动手」。
        """
        if not self.config.screenshot_retention_enabled:
            return RetentionPolicy()
        return RetentionPolicy(
            days=self.config.screenshot_retention_days,
            max_bytes=self.config.screenshot_retention_mb * 1024 * 1024,
        )

    def _refresh_storage_usage(self) -> None:
        count, size = directory_usage(self.event_store.screenshot_dir)
        enabled, days, megabytes = self.settings_panel.retention()
        if not enabled:
            state = "自动清理已关闭"
        else:
            limit = "不限" if megabytes <= 0 else f"{megabytes} MB"
            state = f"容量上限 {limit}，保留 {days or '不限'} 天"
        self.settings_panel.set_storage_usage(
            f"已存 {count} 张 · {format_size(size)}（{state}）"
        )

    def _run_retention(self) -> None:
        """定时/启动时的自动清理。

        两道闸门，任一没打开就什么都不删：

        * 用户没显式开启自动清理（默认就是关的）；
        * 配置刚从坏文件回落成默认值 —— 那份默认值不代表用户的意愿，拿它去删取证
          材料是不可接受的。手动「立即清理」也受第一道闸门约束，但不受第二道约束
          （那是用户看着界面按的）。
        """
        if not getattr(self, "_config_trusted", False):
            logger.info("配置未被可信读取，跳过自动截图清理")
            self._refresh_storage_usage()
            return
        policy = self._retention_policy()
        if not policy.enabled:
            self._refresh_storage_usage()
            return
        result = prune_screenshots(self.event_store.screenshot_dir, policy)
        if result.removed:
            self.source_panel.set_status(f"取证留存：{result.describe()}")
        self._refresh_storage_usage()

    def _prune_now(self) -> None:
        result = prune_screenshots(
            self.event_store.screenshot_dir, self._retention_policy()
        )
        self._refresh_storage_usage()
        if self.config.screenshot_retention_enabled:
            self.source_panel.set_status(f"取证留存：{result.describe()}")
        else:
            self.source_panel.set_status("取证留存：自动清理未开启，未删除任何文件")

    def _update_cpu_low_power_availability(self) -> None:
        device = self.source_panel.selected_device()
        if device != "cpu":
            self.settings_panel.set_cpu_low_power_enabled(False)
            self.settings_panel.cpu_low_power_cb.setToolTip(
                "CPU 低功耗模式仅适用于 CPU 推理。"
            )
            return
        resolution = resolve_inference_policy(
            self.config.model_path,
            device,
            self.config.cpu_low_power_preset,
        )
        if self.config.cpu_low_power_preset and not resolution.available:
            self.settings_panel.set_cpu_low_power_available(
                False,
                f"CPU 低功耗模式不可用：{resolution.unavailable_reason}",
            )
            return
        self.settings_panel.set_cpu_low_power_available(True)

    def _update_playback_enabled(self) -> None:
        running = self.worker is not None and self.worker.isRunning()
        self.playback_panel.set_file_mode(self._is_file_mode(), running)

    def start_detection(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        source = self.source_panel.get_source()
        if not source:
            QMessageBox.warning(self, "缺少视频源", "请输入摄像头索引或视频文件路径。")
            return
        if self._is_file_mode() and not Path(source).is_file():
            QMessageBox.warning(self, "文件不存在", f"找不到视频文件：\n{source}")
            return

        spec = VideoSourceSpec(
            value=source,
            operation_mode=self.config.operation_mode,
            loop_playback=self.playback_panel.loop_cb.isChecked(),
            speed=self.playback_panel.speed(),
        )
        model_path = Path(self.config.model_path)
        if not model_path.is_absolute():
            # 分组式布局的 models/ 优先，扁平式（模型直接放 exe 同级）作为兼容回退；
            # 两处都没有就保留相对路径，让 ultralytics 沿用它按名下载的既有行为。
            resolved = app_paths.resource(
                f"models/{model_path.as_posix()}", model_path.as_posix()
            )
            if resolved.exists():
                model_path = resolved

        self._save_config()
        selected_device = self.source_panel.selected_device()
        resolution = resolve_inference_policy(
            str(model_path),
            selected_device,
            self.config.cpu_low_power_preset,
        )
        if not resolution.available:
            QMessageBox.warning(
                self,
                "CPU 低功耗模式不可用",
                resolution.unavailable_reason,
            )
            self.source_panel.set_status(
                f"CPU 低功耗模式不可用：{resolution.unavailable_reason}"
            )
            return
        self.worker = DetectionWorker(
            spec=spec,
            model_path=resolution.policy.model_path,
            device=selected_device,
            zones=self.zones,
            event_store=self.event_store,
            policy=resolution.policy,
            parent=self,
        )
        self._connect_worker(self.worker)

        self.video_duration = 0.0
        self._alarm_overlay_enabled = True
        self.playback_panel.set_progress(0.0, 0.0)
        self.playback_panel.set_paused(False)
        self.source_panel.set_running(True)
        self.settings_panel.set_mode_enabled(False)
        self.settings_panel.set_cpu_low_power_enabled(False)
        self.settings_panel.set_retention_controls_enabled(False)
        self._set_profile_controls_enabled(False)
        self.playback_panel.set_file_mode(spec.is_file, True)
        self.source_panel.set_status(f"正在打开视频源… 推理设备: {selected_device}")
        self.worker.start()

    def _connect_worker(self, worker: DetectionWorker) -> None:
        """把检测线程的信号接到界面上。

        单独成一段，是为了能被测试直接驱动 —— 其中 frame_ready 的接法最容易出错：
        它要在投递完成之后回执给**发出这一帧的那个** worker（点击「停止/打开」可能
        已经把它换掉了，用 self.worker 去清标志会把新 worker 的待显示状态误清掉），
        而这类错误只有真跑起来、还要界面恰好跟不上时才现形。
        """
        worker.frame_ready.connect(
            lambda frame, source=worker: self._on_frame_ready(source, frame)
        )
        worker.event_ready.connect(self._on_alarm_event)
        worker.event_updated.connect(self._on_alarm_event_updated)
        worker.status_changed.connect(self.source_panel.set_status)
        worker.source_opened.connect(self._on_source_opened)
        worker.progress_changed.connect(self._on_progress_changed)
        worker.finished.connect(self._on_worker_finished)

    def stop_detection(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        self._alarm_overlay_enabled = False
        self.video_widget.clear_alarm()
        self.source_panel.stop_btn.setEnabled(False)
        self.source_panel.set_status("正在停止检测…")
        self.worker.stop()

    def _set_armed(self, armed: bool) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        self.worker.set_armed(armed)
        self._alarm_overlay_enabled = armed
        if not armed:
            self.video_widget.clear_alarm()
            self.source_panel.set_status("已撤防：检测与报警已暂停，视频预览继续。")
        else:
            self.source_panel.set_status("已恢复警戒：检测与报警已启用。")

    def _on_worker_finished(self) -> None:
        finished_worker = self.worker
        self.worker = None
        if finished_worker is not None:
            finished_worker.deleteLater()
        self._alarm_overlay_enabled = False
        self.video_widget.clear_alarm()
        self.source_panel.set_running(False)
        self.settings_panel.set_mode_enabled(True)
        self._update_cpu_low_power_availability()
        self.settings_panel.set_retention_controls_enabled(True)
        self._set_profile_controls_enabled(True)
        self.playback_panel.set_paused(False)
        self._update_playback_enabled()

    def _on_frame_ready(self, source: DetectionWorker, frame: np.ndarray) -> None:
        self.video_widget.set_frame(frame)
        # 帧已经交给画面（颜色转换与拷贝都在 set_frame 里同步做完了），回执允许下一帧。
        # worker 靠这个回执判断界面跟不跟得上，跟不上就丢显示帧而不是排队 —— 否则
        # 队列里的帧会按秒堆起来（1080p 一帧 6 MB）。重绘是异步的，所以真实排队的
        # 帧数最多会比「已画完的帧」多出一两帧，但不会无上限。
        source.frame_consumed()

    def _on_source_opened(self, width: int, height: int, duration: float) -> None:
        self.video_duration = duration
        self.config.source_size = [width, height]
        self.playback_panel.set_progress(0.0, duration)
        source = self.source_panel.get_source()
        if self._is_file_mode():
            self.config.file_history = add_history_entry(
                self.config.file_history,
                source,
                file_source=True,
            )
            history = self.config.file_history
        else:
            self.config.camera_history = add_history_entry(
                self.config.camera_history,
                source,
            )
            history = self.config.camera_history
        self.source_panel.refresh_source_history(history)
        kind = "视频文件" if self._is_file_mode() else "摄像头"
        self.source_panel.set_status(f"{kind}已打开：{width}×{height}")
        self._schedule_config_save()

    def _on_progress_changed(self, ratio: float) -> None:
        self.playback_panel.set_progress(ratio, self.video_duration)

    def _on_alarm_event_updated(self, event: AlarmEvent) -> None:
        if event.operation_mode == self.config.operation_mode:
            self.event_panel.append_event(event)

    def _on_alarm_event(self, event: AlarmEvent) -> None:
        if (
            self._alarm_overlay_enabled
            and event.operation_mode == self.config.operation_mode
        ):
            self.video_widget.show_alarm(event)
        self.source_panel.set_status(
            f"警报：目标 {event.track_id} 进入区域“{event.zone_name}”"
        )
        # 通知不看 _alarm_overlay_enabled（那只是界面上的红条），也不按运行模式过滤：
        # 闯入就是闯入，撤防时压根不会有事件走到这里。
        self._notify_alarm(event)

    def _toggle_pause(self) -> None:
        if self.worker is None or not self.worker.isRunning() or not self._is_file_mode():
            return
        paused = self.playback_panel.toggle_paused()
        self.worker.set_paused(paused)

    def _step_frame(self) -> None:
        if self.worker is None or not self.worker.isRunning() or not self._is_file_mode():
            return
        self.playback_panel.set_paused(True)
        self.worker.step()

    def _set_loop_playback(self, enabled: bool) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.set_loop_playback(enabled)
        self._schedule_config_save()

    def _set_playback_speed(self, value: int) -> None:
        del value
        if self.worker is not None and self.worker.isRunning():
            self.worker.set_speed(self.playback_panel.speed())
        self._schedule_config_save()

    def _on_mapping_changed(self, mapping: dict[str, float]) -> None:
        self.config.display_to_original_scale = mapping

    def _on_frame_size_changed(self, width: int, height: int) -> None:
        self.config.source_size = [width, height]
        self._schedule_config_save()

    def _clear_events(self) -> None:
        answer = QMessageBox.question(
            self,
            "清空报警记录",
            "确定清空报警记录吗？已保存的截图文件将保留。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.event_store.clear(self.config.operation_mode)
            self.event_panel.clear()
        except OSError as error:
            logger.exception("Unable to clear alarm event history")
            QMessageBox.warning(self, "清空失败", str(error))

    def _schedule_config_save(self) -> None:
        self._save_timer.start()

    def _save_config(self) -> None:
        current_source = self.source_panel.get_source()
        if self.config.operation_mode == "video":
            self.config.video_source = current_source
        else:
            self.config.monitor_source = current_source
        self.config.source = current_source
        self.config.source_type = "file" if self.config.operation_mode == "video" else "camera"
        self.config.test_mode = self.config.operation_mode == "video"
        self.config.loop_playback = self.playback_panel.loop_cb.isChecked()
        self.config.playback_speed = self.playback_panel.speed()
        self.config.inference_device = self.source_panel.selected_device()
        # 设置页那一组按绑定表整组回写（低功耗、留存、远程通知都在表里）。
        self._settings_binder.store(self.config)
        self.config.zones = self.zones
        self.config.display_to_original_scale = self.video_widget.coordinate_mapping()
        try:
            self.config_store.save(self.config)
        except OSError:
            logger.exception("Unable to save configuration")

    def closeEvent(self, event: object) -> None:
        self._save_timer.stop()
        self._retention_timer.stop()
        self._save_config()
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            if not self.worker.wait(10_000):
                QMessageBox.warning(
                    self,
                    "检测仍在停止",
                    "检测线程尚未退出，请稍后再次关闭窗口。",
                )
                event.ignore()
                return
        # 给已经入队的通知一点时间发完；发不完的那几条只影响通知，本地记录与截图都在。
        self.notification_dispatcher.close()
        event.accept()
