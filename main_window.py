from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

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
    QSlider,
    QStyledItemDelegate,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from alarm_service import EventStore
from config_store import ConfigStore
from compute_devices import enumerate_inference_devices
from detection_worker import DetectionWorker
from inference_profiles import resolve_inference_policy
from models import AlarmEvent, AppConfig, ZoneDefinition, ZoneProfile
from profile_store import ProfileStore
from video_source import VideoSourceSpec
from video_widget import VideoWidget

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
EVENTS_DIR = APP_DIR / "events"
PROFILES_DIR = APP_DIR / "profiles"


# ---------------------------------------------------------------------------
# 右侧面板：视频源区
# ---------------------------------------------------------------------------

class SourcePanel(QGroupBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("视频源", parent)
        layout = QVBoxLayout(self)

        # 来源输入行
        src_row = QHBoxLayout()
        self.source_label = QLabel("监控来源:")
        self.source_edit = QLineEdit("0")
        self.source_edit.setPlaceholderText("摄像头索引或实时流地址")
        self.browse_btn = QPushButton("…")
        self.browse_btn.setFixedWidth(28)
        self.browse_btn.setToolTip("选择视频文件")
        self.browse_btn.clicked.connect(self._browse_file)
        src_row.addWidget(self.source_label)
        src_row.addWidget(self.source_edit)
        src_row.addWidget(self.browse_btn)
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


# ---------------------------------------------------------------------------
# 右侧面板：区域管理区
# ---------------------------------------------------------------------------

class SettingsPanel(QGroupBox):
    mode_changed = Signal(str)
    cpu_low_power_changed = Signal(bool)

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
        self.back_btn = QPushButton("返回主页")
        layout.addWidget(self.video_radio)
        layout.addWidget(self.monitor_radio)
        layout.addWidget(self.cpu_low_power_cb)
        layout.addStretch()
        layout.addWidget(self.back_btn)
        self.video_radio.toggled.connect(self._emit_mode)
        self.cpu_low_power_cb.toggled.connect(self.cpu_low_power_changed)

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
    def __init__(self, event: AlarmEvent, parent: Optional[QWidget] = None) -> None:
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
            ("状态", "未触发报警" if not event.alarmed else event.status),
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
            if screenshot_path:
                pixmap = QPixmap(screenshot_path)
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
        layout.addLayout(screenshots)


class EventPanel(QGroupBox):
    HEADERS = [
        "记录时间", "视频源", "区域", "目标ID", "进入时刻", "报警时刻",
        "退出时刻", "状态", "闯入时长", "进入取证", "报警取证",
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("报警记录", parent)
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
            AlarmDetailDialog(event, self).exec()

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

        self.video_widget = VideoWidget()
        self.source_panel = SourcePanel()
        self.settings_panel = SettingsPanel()
        self.zone_panel = ZonePanel()
        self.playback_panel = PlaybackPanel()
        self.event_panel = EventPanel()

        self._build_layout()
        self._load_controls()
        self._connect_signals()
        self._load_event_history()

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
            return self.config_store.load()
        except (OSError, ValueError, TypeError) as error:
            logger.exception("Unable to load configuration")
            QMessageBox.warning(self, "配置读取失败", f"将使用默认配置。\n{error}")
            return AppConfig()

    def _load_controls(self) -> None:
        self._apply_operation_mode(self.config.operation_mode, persist=False)
        self.source_panel.source_edit.setText(
            self.config.video_source
            if self.config.operation_mode == "video"
            else self.config.monitor_source
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
        self.settings_panel.set_cpu_low_power(self.config.cpu_low_power_preset)
        self._update_cpu_low_power_availability()
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
        self.settings_btn.clicked.connect(lambda: self.sidebar_pages.setCurrentIndex(1))
        self.settings_panel.back_btn.clicked.connect(
            lambda: self.sidebar_pages.setCurrentIndex(0)
        )
        self.settings_panel.mode_changed.connect(self._apply_operation_mode)
        self.settings_panel.cpu_low_power_changed.connect(self._on_cpu_low_power_changed)
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
        cancel_button = dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
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

    def _on_cpu_low_power_changed(self, enabled: bool) -> None:
        self.config.cpu_low_power_preset = enabled
        self._schedule_config_save()

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
        if not model_path.is_absolute() and (APP_DIR / model_path).exists():
            model_path = APP_DIR / model_path

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
        self.worker.frame_ready.connect(self.video_widget.set_frame)
        self.worker.event_ready.connect(self._on_alarm_event)
        self.worker.event_updated.connect(self._on_alarm_event_updated)
        self.worker.status_changed.connect(self.source_panel.set_status)
        self.worker.source_opened.connect(self._on_source_opened)
        self.worker.progress_changed.connect(self._on_progress_changed)
        self.worker.finished.connect(self._on_worker_finished)

        self.video_duration = 0.0
        self._alarm_overlay_enabled = True
        self.playback_panel.set_progress(0.0, 0.0)
        self.playback_panel.set_paused(False)
        self.source_panel.set_running(True)
        self.settings_panel.set_mode_enabled(False)
        self.settings_panel.set_cpu_low_power_enabled(False)
        self._set_profile_controls_enabled(False)
        self.playback_panel.set_file_mode(spec.is_file, True)
        self.source_panel.set_status(f"正在打开视频源… 推理设备: {selected_device}")
        self.worker.start()

    def stop_detection(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        self._alarm_overlay_enabled = False
        self.video_widget.clear_alarm()
        self.source_panel.stop_btn.setEnabled(False)
        self.source_panel.set_status("正在停止检测…")
        self.worker.stop()

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
        self._set_profile_controls_enabled(True)
        self.playback_panel.set_paused(False)
        self._update_playback_enabled()

    def _on_source_opened(self, width: int, height: int, duration: float) -> None:
        self.video_duration = duration
        self.config.source_size = [width, height]
        self.playback_panel.set_progress(0.0, duration)
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
        self.config.cpu_low_power_preset = self.settings_panel.cpu_low_power_cb.isChecked()
        self.config.zones = self.zones
        self.config.display_to_original_scale = self.video_widget.coordinate_mapping()
        try:
            self.config_store.save(self.config)
        except OSError:
            logger.exception("Unable to save configuration")

    def closeEvent(self, event: object) -> None:
        self._save_timer.stop()
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
        event.accept()
