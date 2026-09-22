from __future__ import annotations

import html
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from PySide6.QtCore import (
    QEvent,
    QModelIndex,
    QObject,
    QPoint,
    QSize,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QKeyEvent,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QApplication,
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
    QScroller,
    QScrollerProperties,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import app_paths
from alarm_service import EventStore
from config_store import ConfigStore
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
    Moment,
    ZoneDefinition,
    ZoneProfile,
    parse_clock,
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

if TYPE_CHECKING:
    # 只给类型检查看：这条链在运行时是**延迟导入**的（见 start_detection）——
    # detection_worker 要 ultralytics + supervision + trackers，实测约 8 秒，压在启动
    # 路径上就是一段白屏等待，而它只有「真的开始推理」时才需要。
    from detection_worker import DetectionWorker

logger = logging.getLogger(__name__)

APP_DIR = app_paths.APP_DIR
CONFIG_PATH = app_paths.data("config.json")
EVENTS_DIR = app_paths.data("events")
PROFILES_DIR = app_paths.data("profiles")


# ---------------------------------------------------------------------------
# 输入方式：触摸滑动、Shift + 滚轮
# ---------------------------------------------------------------------------

def enable_touch_scrolling(*areas: QAbstractScrollArea) -> None:
    """让这些区域可以用手指拖着滚（触摸屏、带触摸的工控一体机）。

    Qt 的部件默认不认触摸拖动：手指按住一拖，事件落到控件上就变成鼠标拖动 —— 在表格里
    是「选中了一行」，在滚动区里什么也不做（实测）。``QScroller`` 把**单指拖动**接管成
    带惯性的滚动。

    只抓 ``TouchGesture``，不抓 ``LeftMouseButtonGesture``：后者会让鼠标按住拖动也变成
    滚动，那会破坏表格里拖选，更会破坏视频画面上按住拖动加点、拖顶点这些画区域的操作。
    单指点一下（没有拖动）不会被接管，仍旧是普通的点击。
    """
    for area in areas:
        viewport = area.viewport()
        QScroller.grabGesture(viewport, QScroller.ScrollerGestureType.TouchGesture)
        scroller = QScroller.scroller(viewport)
        properties = scroller.scrollerProperties()
        # 斜着拖的时候锁住先动的那个方向：不然表格会跟着左右晃，横向一格一格地跑。
        properties.setScrollMetric(
            QScrollerProperties.ScrollMetric.AxisLockThreshold, 0.2
        )
        scroller.setScrollerProperties(properties)


class ShiftWheelScrollsSideways(QObject):
    """按住 Shift 滚轮 → 横向滚动（Windows 上几乎所有列表/表格都是这个约定）。

    Qt 的部件默认没有这条：实测 Shift+滚轮与普通滚轮一样只滚纵向。做法是把纵向的滚动量
    搬成横向、去掉 Shift 再发给同一个控件，让 Qt 自己按它那套换算去滚 —— 这样手感与
    普通滚轮一致，也自动兼顾了触摸板的像素级滚动。

    没有横向可滚时**放行**，仍旧纵向滚：否则窗口够宽、列都放得下的时候，Shift+滚轮会
    变成什么都不动。
    """

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if event.type() != QEvent.Type.Wheel:
            return False
        if not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            return False
        area = _enclosing_scroll_area(watched)
        if area is None:
            return False
        bar = area.horizontalScrollBar()
        if bar is None or bar.maximum() <= bar.minimum():
            return False
        sideways = _sideways_wheel(event)
        if sideways is None:
            return False
        QApplication.sendEvent(watched, sideways)
        return True


def install_shift_wheel_scroll(application: QApplication) -> ShiftWheelScrollsSideways:
    """装上 Shift+滚轮横向滚动（应用级，一次覆盖所有列表、表格与滚动区）。

    返回过滤器本身：它挂在 application 名下（父子关系），不会被垃圾回收 —— 过滤器对象
    一旦被回收，事件就再也不经过它，而且没有任何报错。
    """
    scroll_filter = ShiftWheelScrollsSideways(application)
    application.installEventFilter(scroll_filter)
    return scroll_filter


def _enclosing_scroll_area(watched: object) -> QAbstractScrollArea | None:
    """滚轮事件是发给内容控件（视口）的，往上找它属于哪个滚动区域。"""
    if isinstance(watched, QAbstractScrollArea):
        return watched
    parent = watched.parent() if isinstance(watched, QObject) else None
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea):
            return parent
        parent = parent.parent()
    return None


def _sideways_wheel(event: QWheelEvent) -> QWheelEvent | None:
    """把纵向的滚动量搬到横向；没有量可搬时返回 None。"""
    pixel = event.pixelDelta()
    angle = event.angleDelta()
    if pixel.y():
        pixel = QPoint(pixel.y(), 0)
    if angle.y():
        angle = QPoint(angle.y(), 0)
    if pixel.isNull() and angle.isNull():
        return None
    return QWheelEvent(
        event.position(),
        event.globalPosition(),
        pixel,
        angle,
        event.buttons(),
        Qt.KeyboardModifier.NoModifier,
        event.phase(),
        event.inverted(),
    )


# ---------------------------------------------------------------------------
# 后台探测：可用推理设备
# ---------------------------------------------------------------------------

class DeviceProbe(QThread):
    """在后台算可用推理设备，算完发 ``ready``。

    ``compute_devices.enumerate_inference_devices`` 要 ``import torch``（实测约 4 秒），
    压在主线程上就是启动时的一段白屏等待。放到线程里：窗口先出来、状态栏写着
    「正在检测推理设备…」，探测完再把下拉框填上并放开「打开」。
    """

    ready = Signal(list)

    def run(self) -> None:
        try:
            from compute_devices import enumerate_inference_devices

            options = enumerate_inference_devices()
        except Exception:  # noqa: BLE001 - 探测失败按「只有 CPU」处理，不该让启动挂掉
            logger.exception("枚举推理设备失败")
            options = [("cpu", "CPU")]
        self.ready.emit(options)


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
        # 状态栏里的文字会长得离谱（"检测运行中：CPU 低功耗模式（OpenVINO INT8，512，
        # 每 4 帧检测），推理设备: cpu"），而不换行的 QLabel 的最小宽度就是整行文字的
        # 宽度 —— 一条状态就能把整个右侧面板撑宽，超出视口的部分全被裁掉（"删除"按钮、
        # 配置组状态、颜色值都会被切掉一截）。所以这里必须换行，并且告诉布局"我的宽度
        # 不影响你"：面板窄了就让文字多折几行，而不是把面板撑宽。
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #AAB4BE; font-size: 11px;")
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
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
        # 文字换行之后可能折成好几行，也可能因为面板窄而被截得看不全；完整内容挂到
        # 悬停提示上，保证任何情况下都读得到。
        self.status_label.setToolTip(text)

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
            "仅在 CPU 推理时使用 OpenVINO INT8、512 输入和每 4 帧检测，"
            "并且只检测警戒区范围（外扩 30%）。\n"
            "区域内的人反而检得更可靠（同样的输入像素全花在要害处），"
            "代价是区域外的行人不再画框。"
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
            reason
            if not available
            else (
                "仅在 CPU 推理时使用 OpenVINO INT8、512 输入和每 4 帧检测，"
                "并且只检测警戒区范围（外扩 30%）。\n"
                "区域内的人反而检得更可靠（同样的输入像素全花在要害处），"
                "代价是区域外的行人不再画框。"
            )
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

        # 配置组与区域两个列表：触摸屏上用手指标着滚（列表长了才划得动）
        enable_touch_scrolling(self.profile_list, self.zone_list)

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

class PreviewImageLabel(QLabel):
    """详情里的截图预览：双击交给系统默认程序打开原图。

    这里显示的是缩放后的预览，而取证要看的是原图 —— 目标手里拿的什么、区域边界压在哪、
    有没有第二个同伙，都得放大才看得出来。所以双击直接把原文件交给系统默认程序（Windows
    上就是默认图片查看器），程序里不必再做一个看图器。

    预览按控件当前宽度等比缩放：详情栏是能拖宽拖窄的，图跟着栏宽走。定死一个尺寸的话，
    栏一变窄图的右边就被裁掉，而裁掉的正是画面里靠边的那些人和边界线。
    """

    activated = Signal(str)
    # 预览的高度上限。取证图基本是 16:9，按宽度缩放后本来也高不到哪去；这条是给竖屏
    # 画面留的，否则一张 1080×1920 的图会占掉好几屏高度。
    PREVIEW_MAX_HEIGHT = 300
    # 预览的下限。低于这个尺寸就看不清"目标手里拿的什么、边界压在哪"，所以它是常量下限，
    # 而不是由当前这张图算出来的。
    MIN_WIDTH = 330
    MIN_HEIGHT = 240

    def __init__(
        self,
        path: str,
        pixmap: QPixmap | None = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._path = path
        self._source = QPixmap(pixmap) if pixmap is not None else QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        # 宽度上告诉布局「我不影响你」：预览的宽度跟着详情栏走，而不是反过来由图片尺寸
        # 决定栏宽 —— QLabel 有 pixmap 时最小宽度就是图片宽度，弹窗会因此再也拖不窄。
        # 高度上则按原图宽高比给布局一个**纯函数**（见 heightForWidth）：高度如果只由
        # 「当前这张 pixmap 多大」决定，布局算出的内容高度就会依赖上一轮缩放的结果 ——
        # 缩放窗口时 QScrollArea 的滚动条于是反复开关，而它的 updateScrollBars 是同步
        # 回调，会一路递归到栈溢出（实测转储里同一条链重复了 453 层）。
        policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setToolTip(f"双击用系统默认程序打开原图：\n{path}")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rescale()

    def heightForWidth(self, width: int) -> int:
        """宽度 → 高度：只由原图宽高比决定，与当前这张 pixmap 无关。

        这样布局问多少次都是同一个答案，缩放时不会出现"这一轮算出的高度和上一轮不一样"
        的死循环。上限给竖屏画面留着，下限是看清取证细节的最小尺寸。
        """
        if self._source.isNull() or self._source.width() <= 0 or width <= 0:
            return self.MIN_HEIGHT
        ratio = self._source.height() / self._source.width()
        return max(self.MIN_HEIGHT, min(self.PREVIEW_MAX_HEIGHT, round(width * ratio)))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        # 只有宽度变了才重新缩放：详情栏是纵向滚动的，高度变化很频繁，而高度变化对
        # 「按宽度缩放」这件事没有影响 —— 重算一遍白花时间，还会让每条尺寸变化都走一遍
        # setPixmap → 布局失效的链路。
        if event.size().width() != event.oldSize().width():
            self._rescale()

    def _rescale(self) -> None:
        if self._source.isNull() or self.width() <= 0:
            return
        self.setPixmap(
            self._source.scaled(
                self.width(),
                self.PREVIEW_MAX_HEIGHT,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def mouseDoubleClickEvent(self, event: object) -> None:
        del event
        self.activated.emit(self._path)


def _mode_label(operation_mode: str) -> str:
    """运行模式的中文说法。

    认不出来的（手工编辑过、或很旧的记录）说「未记录」：一律按「监控模式」说，等于替
    一条从没记过模式的记录编了个没发生过的事实。
    """
    return {"video": "视频模式", "monitor": "监控模式"}.get(operation_mode, "未记录")


def _moment_detail(event: AlarmEvent, moment: Moment) -> str:
    """详情里的一个时刻：真实钟点，视频模式下再附上它在视频里的位置。

    钟点用来对日志、和别人说的「几点几分」对上；视频位置用来在播放器里找到那一刻 ——
    两样都有用，所以附在括号里，而不是二选一。都没有（老记录）时如实说没记：推算会被
    播放速度带偏（同一段视频里，视频位置差 2.04 秒的两条记录，真实钟点差可能是 1 秒也
    可能是 4 秒），拿推算值冒充时间等于给取证材料编一个精确到秒的假数字。
    """
    clock = event.moment_clock(moment)
    offset = event.moment_offset(moment)
    if clock is not None:
        stamp = clock.strftime("%H:%M:%S")
        if offset is None or event.operation_mode != "video":
            return stamp
        return f"{stamp}（视频位置 {offset:.2f}s）"
    if offset is None:
        return "未记录"
    return f"未记录真实钟点（这条记录只存了视频位置 {offset:.2f}s）"


def _confidence_detail(event: AlarmEvent) -> str:
    """报警那一刻模型有多确定。

    误报排查里这是第一个要看的数：0.86 和 0.62 是两回事。老记录没这个字段，如实
    说没记 —— 不去推算，也不拿"没记录"充作"很确定"。
    """
    if event.alarm_confidence is None:
        return "未记录（这条记录早于该字段）"
    return f"{event.alarm_confidence:.2f}"


def _detail_rows(event: AlarmEvent) -> list[tuple[str, str]]:
    """详情里逐行显示的字段。两个弹窗共用一份，免得加字段时只改了一处。"""
    return [
        ("记录时间", event.wall_time or "未记录"),
        ("进入时间", _moment_detail(event, "entered")),
        ("报警时间", _moment_detail(event, "alarmed")),
        ("报警置信度", _confidence_detail(event)),
        ("退出时间", _moment_detail(event, "exited")),
        (
            "闯入时长",
            f"{event.duration_seconds:.2f}s"
            if event.duration_seconds is not None
            else "未结算",
        ),
        ("状态", event.status_label),
        ("运行模式", _mode_label(event.operation_mode)),
        ("视频源", event.source),
        ("警戒区域", event.zone_name),
        ("目标 ID", event.track_id),
    ]


# 详情的三种文字层次：字段名（灰，让眼睛能跳过）、值（正常文字色）、没记到的（灰）。
# 报警那两行的值有内容时用报警红：排查时最先看的就是「报没报、多确定」，让它跳出来。
DETAIL_MUTED = "#7A8A99"
DETAIL_ALARM = "#C62828"
DETAIL_ALARM_FIELDS = ("报警时间", "报警置信度")
# 详情正文字号。界面默认字号（约 12px）在这里偏小：右栏是要一行行读的，弹窗也共用。
DETAIL_FONT_SIZE = 15
DETAIL_HINT_SIZE = 12


def _detail_row_html(label: str, value: str) -> str:
    """一行详情的富文本：字段名灰、值主色，报警那两行有内容时报警红加粗。

    值里的 ``&``、``<`` 要转义 —— 视频源与路径里出现它们是常事。
    """
    escaped_label = html.escape(label)
    escaped_value = html.escape(value)
    if value.startswith("未记录") or value.startswith("未触发"):
        # 「没记到」既不是报警也不是结论，别用报警红，免得看着像"报了警"。
        value_html = f'<span style="color: {DETAIL_MUTED};">{escaped_value}</span>'
    elif label in DETAIL_ALARM_FIELDS:
        value_html = (
            f'<span style="color: {DETAIL_ALARM}; font-weight: bold;">{escaped_value}</span>'
        )
    else:
        value_html = escaped_value
    return (
        f'<span style="color: {DETAIL_MUTED};">{escaped_label}</span>: {value_html}'
    )


def _detail_label(label: str, value: str) -> QLabel:
    """详情里的一行。两个弹窗共用，字号与配色才不会各写一套。"""
    row = QLabel(_detail_row_html(label, value))
    row.setTextFormat(Qt.TextFormat.RichText)
    # 视频源与截图路径可以很长；不换行的话它们会把右栏（弹窗）撑宽，别的行跟着被裁掉。
    row.setWordWrap(True)
    row.setStyleSheet(f"font-size: {DETAIL_FONT_SIZE}px;")
    return row


def _detail_hint(text: str) -> QLabel:
    """详情里的提示与图注：比正文小一号的灰字，跟要读的字段分开。"""
    hint = QLabel(text)
    hint.setStyleSheet(f"color: {DETAIL_MUTED}; font-size: {DETAIL_HINT_SIZE}px;")
    return hint


def _screenshot_entries(event: AlarmEvent) -> list[tuple[str, str]]:
    """要显示的两张取证图：(标题, 记录里的路径)，顺序就是从上到下的顺序。

    进入那张在没有单独记录时回落到 ``screenshot_path``：旧记录只存了一个路径。
    """
    return [
        ("进入取证", event.entry_screenshot_path or event.screenshot_path),
        ("报警取证", event.alarm_screenshot_path),
    ]


def _resolve_screenshot(
    path: str,
    resolver: Callable[[str], Path | None] | None,
) -> Path | None:
    """把记录里的路径变成实际文件；找不到返回 None。

    相对路径、被搬过家的旧记录都靠解析器处理，这里只是最后的兜底：解析器没给、或者
    它没找到时，退回直接按路径打开。
    """
    if not path:
        return None
    resolved = resolver(path) if resolver is not None else None
    if resolved is not None:
        return resolved
    direct = Path(path)
    return direct if direct.is_file() else None


def _open_in_default_viewer(parent: QWidget, path: str) -> None:
    """把原图交给系统默认程序。

    用 QDesktopServices 而不是 os.startfile：不需要平台分支，行为就是"用默认程序打开"，
    而且返回 False 时能明确报错 —— 否则双击没反应，用户只会以为程序坏了。
    """
    if not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
        logger.warning("无法用系统默认程序打开 %s", path)
        QMessageBox.warning(parent, "打开失败", f"系统里没有能打开这个文件的程序：\n{path}")


def _screenshot_widget(
    title: str,
    path: str,
    event: AlarmEvent,
    resolver: Callable[[str], Path | None] | None,
    parent: QWidget,
) -> QWidget:
    """一张取证预览；显示不出来时换成写明原因的说明。

    显示不出来时说清是哪一种：本来没报警 / 没写进去 / 文件被清理或删掉。图片不可用时
    刻意不给双击入口 —— 点了没反应比没有入口更让人困惑。
    """
    resolved = _resolve_screenshot(path, resolver)
    if resolved is not None:
        pixmap = QPixmap(str(resolved))
        if not pixmap.isNull():
            preview = PreviewImageLabel(str(resolved), pixmap, parent)
            preview.activated.connect(
                lambda clicked: _open_in_default_viewer(parent, clicked)
            )
            return preview
    missing = QLabel(f"{title}不可用\n{EventStore.unavailable_reason(event, path)}")
    missing.setAlignment(Qt.AlignmentFlag.AlignCenter)
    missing.setMinimumSize(330, 240)
    missing.setWordWrap(True)
    missing.setToolTip(f"记录里的路径：{path or '（空）'}")
    return missing


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
        details = [
            *_detail_rows(event),
            ("进入取证", event.entry_screenshot_path or "无"),
            ("报警取证", event.alarm_screenshot_path or "无"),
        ]
        for label, value in details:
            layout.addWidget(_detail_label(label, value))
        layout.addWidget(_detail_hint("提示：双击截图可用系统默认程序打开原图"))
        screenshots = QHBoxLayout()
        for title, screenshot_path in _screenshot_entries(event):
            screenshots.addWidget(
                _screenshot_widget(title, screenshot_path, event, resolver, self)
            )
        layout.addLayout(screenshots)


def _configure_event_table(table: QTableWidget, headers: list[str]) -> None:
    """报警记录表的共同外观（常驻面板与查看器共用）。

    列宽策略两边不一样，各自在调用处设置：面板窄，按内容铺满；查看器可以拖，而且要把
    拖过的宽度存下来。
    """
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(True)
    table.setStyleSheet(
        "QTableWidget::item:hover { background-color: transparent; }"
        "QTableWidget::item:selected { background-color: rgba(1, 174, 231, 128); }"
    )
    table.verticalHeader().setVisible(False)
    # 横向滚动按像素。Qt 的默认是「按列」：滚动条的范围于是变成「第几列」，十列里看得见
    # 四列就是 0..6 —— 而内容实际超出七百多像素。拖起来一格跳一列，拇指的大小与位置也和
    # 看到的画面对不上，正是「横向滚动不同步」。（纵向按行是对的，不动。）
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    # 表头文字靠左，和下面的单元格对齐。Qt 默认把表头文字居中：列一宽，列名就跑到列的中间
    # 去，看着就像「表头和数据列错位」—— 列越宽越明显，所以取证路径列、拖宽过的列上尤其
    # 显眼。靠左之后列名正对着它下面那列的第一个字符，错不错位一眼就能判断。
    table.horizontalHeader().setDefaultAlignment(
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    )


class MiddleElideDelegate(QStyledItemDelegate):
    """放不下时从**中间**省略。

    「视频源」是 ``rtsp://admin:FT0628591@192.168.1.244/Streaming/Channels/101`` 这种
    地址，默认的「省略右边」一截就把通道号 —— 最有用的那一段 —— 吃掉了。中间省略之后
    头尾都在，一眼能看出是哪个点位、哪个通道。放得下时完全没影响。
    """

    def initStyleOption(self, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        super().initStyleOption(option, index)
        option.textElideMode = Qt.TextElideMode.ElideMiddle


def _record_time(event: AlarmEvent) -> datetime | None:
    """记录时间（``wall_time``）解析成 datetime；认不出来返回 None。

    程序写的是 ``"%Y-%m-%d %H:%M:%S"`` 的本地时间文本；手工编辑过、或者很旧的记录可能
    是别的写法 —— 那就不参与时间筛选，而不是让整份记录读不出来。
    """
    return parse_clock(event.wall_time)


def _time_filter_start(code: str, now: datetime) -> datetime | None:
    """时间档位的下界；「全部时间」返回 None。

    「今天」按当天 0 点算，不是「往前推 24 小时」：现场说的「今天」就是日历上的今天。
    """
    if code == "hour":
        return now - timedelta(hours=1)
    if code == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if code == "week":
        return now - timedelta(days=7)
    if code == "month":
        return now - timedelta(days=30)
    return None


class AlarmHistoryDialog(QDialog):
    """查看报警记录：左栏是全部记录加筛选，右栏是选中那条的详情与两张取证图。

    常驻面板那张表只装「当前运行模式最近 200 条」—— 它每次启动都要填一遍，不能因为记录
    文件长到几十 MB 就拖慢启动。翻记录是另一件事：这里读全量，靠上方的筛选与搜索把范围
    收窄，选中那条在右边展开成完整详情。

    两个刻意的取舍：运行模式不在右栏重复显示成英文原值，而是与筛选下拉用同一份中文说法；
    两条截图路径也不进表格 —— 它们在右边是看得见的图，11 列挤进左栏只会让每列都窄到看不清。
    """

    HEADERS = [
        "记录时间", "运行模式", "视频源", "区域", "目标ID",
        "进入时刻", "报警时刻", "退出时刻", "状态", "闯入时长",
    ]
    # 时间筛选的档位：(下拉里显示的文字, 档位代码)。用固定档位而不是两个日期选择器：
    # 现场问的是「刚才 / 今天 / 这几天有没有人进来」，翻记录时手点日历反而慢。
    TIME_FILTERS = (
        ("全部时间", ""),
        ("最近 1 小时", "hour"),
        ("今天", "today"),
        ("最近 7 天", "week"),
        ("最近 30 天", "month"),
    )
    # 表格末尾多一个空的「占位列」（不在 HEADERS 里，也不进 config）。列宽之和小于表宽时，
    # 右边会剩一块**没有 item 的空白**：隔行底色在那里断掉，看着像表没画完。占位列把它填
    # 掉，真实列的宽度一个不动。
    FILLER_COLUMN = len(HEADERS)
    # 真实列的下限。表头的最小列宽被设成 0（见 _build_list_side），下限改由这里把守：
    # 列被拖成 0 宽就看不见也抓不回来了。
    MIN_COLUMN_WIDTH = 24

    def __init__(
        self,
        store: EventStore,
        resolver: Callable[[str], Path | None] | None = None,
        parent: Optional[QWidget] = None,
        *,
        column_widths: dict[str, int] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("查看报警记录")
        # 默认的 QDialog 只有关闭按钮：记录一多、列一宽、右边还要摆两张取证图，只能靠拖
        # 边框一格一格放大，而拖不到整屏。补上最小化/最大化，再加上 F11 全屏（见
        # keyPressEvent 与 fullscreen_btn）。
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        # 和主窗口一样大：左边要放下十来列记录、右边要放下两张取证图，小弹窗里两边都只能
        # 看到一半，翻记录就得一直拖滚动条。
        self.resize(parent.size() if parent is not None else QSize(1100, 720))
        self._store = store
        self._resolver = resolver
        self._events: list[AlarmEvent] = []
        self._saved_widths = dict(column_widths or {})
        # 从全屏还原时回到"进全屏之前是不是最大化"，而不是一律回到普通尺寸。
        self._maximized_before_fullscreen = False
        # 用户拖出来的列宽（列名 → 像素），是**存盘与铺满的唯一基准**：显示宽度里可能还含着
        # 「铺满时补出来的」那部分，把那个存进 config，下次在小窗口打开就全是横向滚动条。
        self._user_widths: dict[str, int] = {}
        # 用户自己拖过的列：铺满时不再自动给它补宽 —— 拖窄了又自己变宽，等于跟用户抢。
        self._user_sized: set[str] = set()
        # 各长文本列内容要多宽（读记录时量一次，见 _measure_content_widths）。
        self._content_widths: dict[str, int] = {}
        # 我们自己调 resizeSection 时置位，免得被当成「用户拖的」记进 _user_widths。
        self._fitting = False

        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_list_side())
        splitter.addWidget(self._build_detail_side())
        # 左边记录表、右边详情与取证图：触摸屏上都能用手指拖着滚
        enable_touch_scrolling(self.table, self.details_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])
        layout.addWidget(splitter)

        self.reload()

    # -- 左栏 ---------------------------------------------------------------

    def _build_list_side(self) -> QWidget:
        side = QWidget()
        layout = QVBoxLayout(side)
        layout.setContentsMargins(0, 0, 0, 0)

        filters = QHBoxLayout()
        self.time_combo = QComboBox()
        self.time_combo.setToolTip(
            "按记录时间筛：现场问的是「刚才 / 今天 / 这几天有没有人进来」。"
        )
        for label, code in self.TIME_FILTERS:
            self.time_combo.addItem(label, code)
        self.time_combo.currentIndexChanged.connect(self._apply_filter)
        filters.addWidget(self.time_combo)
        self.mode_combo = QComboBox()
        self.mode_combo.currentIndexChanged.connect(self._apply_filter)
        filters.addWidget(self.mode_combo)
        self.status_combo = QComboBox()
        self.status_combo.currentIndexChanged.connect(self._apply_filter)
        filters.addWidget(self.status_combo)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索：时间 / 视频源 / 区域 / 目标ID / 状态")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._apply_filter)
        filters.addWidget(self.search_edit, 1)
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setToolTip(
            "重新读一遍记录文件：检测一直在写，翻记录时新报的警靠它出现。"
        )
        self.refresh_btn.clicked.connect(self.reload)
        filters.addWidget(self.refresh_btn)
        self.fullscreen_btn = QPushButton("全屏")
        self.fullscreen_btn.setToolTip(
            "整屏查看记录（F11 切换；全屏时按 Esc 先退出全屏，再按一次才关窗）。"
        )
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        filters.addWidget(self.fullscreen_btn)
        layout.addLayout(filters)

        self.table = QTableWidget(0, len(self.HEADERS) + 1)
        _configure_event_table(self.table, [*self.HEADERS, ""])
        header = self.table.horizontalHeader()
        # 列宽交给用户拖：要看的是「谁、什么时候、哪个区域」，不同点位关心的列不一样，
        # 按内容铺一遍只是第一次打开的起点。
        #
        # 最后一列**不**吸收剩余宽度（setStretchLastSection 默认就是关的，这里写出来是
        # 因为开过一次）：开着它的话，拖中间任何一条边界时最后一列都会跟着补偿，看上去
        # 就是「拖一个动两个」。
        #
        # 空出来的那块由末尾的**占位列**（不在 HEADERS 里）填：它没有内容，宽度等于
        # 「表宽 − 各列之和」，所以拖动真实列时它悄悄补偿也不会被看见，而那片"没上底色"
        # 的空白没有了。见 _fit_columns。
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        # 表头的最小列宽默认是十几像素（按字体算），而占位列经常需要比它更窄：剩下的空白
        # 只有几像素时，占位列被顶到十几像素就会让列宽之和超过表宽，冒出一条本不该有的
        # 横向滚动条。设成 0，真实列的下限改由 MIN_COLUMN_WIDTH 把守。
        header.setMinimumSectionSize(0)
        # 「视频源」放不下时从中间省略（见 MiddleElideDelegate）：很长的 RTSP 地址里，
        # 通道号在末尾，省略右边会把最有用的那一段吃掉。
        self.table.setItemDelegateForColumn(
            self.HEADERS.index("视频源"), MiddleElideDelegate(self.table)
        )
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setMinimumHeight(220)
        # 按内容铺一遍时只看前 100 行：两千多条记录时它会逐行问 delegate，实测这一项在
        # 常驻面板上就要 1.6 秒（见 EventPanel.load_events）。列宽本来就只是起点。
        header.setResizeContentsPrecision(100)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        # 表宽变了（放大窗口、拖分隔条、竖滚动条出现）就重新铺一遍。挂 geometriesChanged
        # 而不是在表格的 Resize 事件里做：表格收到 Resize 时视口与表头都还没跟着变，那时
        # 读到的宽度是旧的（实测 638 vs 736），铺出来就是错的。
        header.geometriesChanged.connect(self._fit_columns)
        self._restore_column_widths()
        # 连 sectionResized 必须放在恢复之后：恢复（以及「没存过宽度时按内容铺一遍」）会
        # 连着触发一串 sectionResized，早接上就会把那些程序化的改动记成「用户拖的」——
        # 一列被记成用户拖过，铺满时就不再给它补宽了。
        header.sectionResized.connect(self._on_section_resized)
        layout.addWidget(self.table, 1)

        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        return side

    def _restore_column_widths(self) -> None:
        """把上次拖过的列宽放回去；一条都没有时先按内容铺一遍。

        按列名找，不按下标：以后加一列，按下标存的那份会整体错位。放回去的是**用户宽度**
        （_user_widths），当前显示宽度由 _fit_columns 按表宽再算一遍。
        """
        header = self.table.horizontalHeader()
        if not self._saved_widths:
            self.table.resizeColumnsToContents()
            for column, name in enumerate(self.HEADERS):
                self._user_widths[name] = max(
                    header.sectionSize(column), self.MIN_COLUMN_WIDTH
                )
            return
        # 恢复也会触发 sectionResized：不挡住的话每一列都会被记成「用户拖过」，铺满时
        # 就再也不会给被截断的列补宽了（见 _on_section_resized）。
        self._fitting = True
        try:
            for column, name in enumerate(self.HEADERS):
                # 只存了一部分时（比如以后加了新列），没存过的那列按内容铺；配置里被手改
                # 成 0 或负数的也在这里兜回下限。
                width = self._saved_widths.get(name) or header.sectionSizeHint(column)
                width = max(width, self.MIN_COLUMN_WIDTH)
                self._user_widths[name] = width
                header.resizeSection(column, width)
        finally:
            self._fitting = False

    # -- 列宽铺满 -----------------------------------------------------------

    def _measure_content_widths(self) -> None:
        """量一遍每一列「内容要多宽」，供铺满时把显示不全的列补到够显示。

        用 QFontMetrics 直接量文字，不走 sizeHintForColumn：后者要过 delegate，两千多条
        记录上每次缩放都算一遍太慢（实测：两万多次文字宽度约 30 ms，够便宜）。只在读记录
        时算一次，筛选变化不重算 —— 否则搜索框里每敲一个字都要多花一次，列宽还会跟着
        搜索词跳。
        """
        metrics = self.table.fontMetrics()
        header = self.table.horizontalHeader()
        # 单元格的左右内边距：QCommonStyle 给 CT_ItemViewItem 加的就是这个
        # （PM_FocusFrameHMargin + 1，左右各一份）。不能拿表头的 sectionSizeHint 减文字
        # 宽度来推 —— 表头那一套内边距和单元格不是同一个数。
        margin = (
            self.table.style().pixelMetric(
                QStyle.PixelMetric.PM_FocusFrameHMargin, None, self.table
            )
            + 1
        )
        padding = 2 * margin
        widest = dict.fromkeys(self.HEADERS, 0)
        for event in self._events:
            row = self._row_values(event)
            for column, name in enumerate(self.HEADERS):
                width = metrics.horizontalAdvance(row[column])
                if width > widest[name]:
                    widest[name] = width
        self._content_widths = {
            name: max(
                # 表头文字自己也要放得下
                header.sectionSizeHint(column),
                padding + widest[name],
            )
            for column, name in enumerate(self.HEADERS)
        }

    def _fit_columns(self) -> None:
        """按表宽把列铺满：先把显示不全的列补到够显示，剩下的交给末尾的占位列。

        每次都从**用户宽度**算起，不在上一次的结果上叠加 —— 否则窗口来回缩放会把补出来
        的宽度越滚越大。只由视口尺寸变化触发；拖动单条边界走 _on_section_resized，那条
        路只动占位列，免得用户在拖一列时看到另一列跟着变。
        """
        header = self.table.horizontalHeader()
        viewport = self.table.viewport().width()
        if viewport <= 0 or not self._user_widths:
            return
        widths = dict(self._user_widths)
        slack = viewport - sum(widths.values())
        if slack > 0:
            # 先满足「差得少」的列：这样能整列显示全的列最多；补不动的长文本列（很长的
            # RTSP 地址就是）拿剩下的。**没有上限** —— 空白本来就是白放着的，全给它也是
            # 赚的，剩下的才轮到占位列。
            deficits = {
                name: self._content_widths.get(name, 0) - widths[name]
                for name in self.HEADERS
                if name not in self._user_sized
            }
            for name in sorted(
                (item for item, need in deficits.items() if need > 0),
                key=lambda item: deficits[item],
            ):
                grow = min(slack, deficits[name])
                widths[name] += grow
                slack -= grow
                if slack <= 0:
                    slack = 0
                    break
        self._fitting = True
        try:
            for column, name in enumerate(self.HEADERS):
                if header.sectionSize(column) != widths[name]:
                    header.resizeSection(column, widths[name])
            header.resizeSection(self.FILLER_COLUMN, max(0, slack))
        finally:
            self._fitting = False

    def _on_section_resized(self, column: int, old: int, new: int) -> None:
        """用户拖动某条边界之后。

        只把差额还给占位列：其它真实列跟着动，就又成了「拖一个动两个」。占位列是空的，
        它变宽变窄看不见。
        """
        if self._fitting or column == self.FILLER_COLUMN:
            return
        header = self.table.horizontalHeader()
        name = self.HEADERS[column]
        # 拖成 0 宽的列看不见也抓不回来（表头最小列宽已被设成 0），这里兜住下限。
        new = max(new, self.MIN_COLUMN_WIDTH)
        if new != header.sectionSize(column):
            self._fitting = True
            try:
                header.resizeSection(column, new)
            finally:
                self._fitting = False
        self._user_widths[name] = new
        # 按拖动方向决定要不要继续自动补宽：**拖窄**了就不再补（用户就是要它窄），
        # **拖宽**了继续跟着空白走 —— 否则「自己拖宽一点想把长地址看清」这个动作反而会
        # 让这一列从此不再被补宽，越拖越看不全。
        if new < old:
            self._user_sized.add(name)
        else:
            self._user_sized.discard(name)
        filler = max(0, header.sectionSize(self.FILLER_COLUMN) - (new - old))
        self._fitting = True
        try:
            header.resizeSection(self.FILLER_COLUMN, filler)
        finally:
            self._fitting = False

    def column_widths(self) -> dict[str, int]:
        """用户拖出来的列宽（列名 → 像素）。关窗时由主窗口写进 config.json。

        返回的是**用户的**宽度，不是当前显示宽度：铺满与补宽出来的那部分不该被记成
        「用户拖成这样的」，否则在放大的窗口里关一次窗，下次在小窗口打开就全是横向滚动条。
        """
        header = self.table.horizontalHeader()
        return {
            name: self._user_widths.get(name, header.sectionSize(column))
            for column, name in enumerate(self.HEADERS)
        }

    # -- 窗口本身 -----------------------------------------------------------

    def toggle_fullscreen(self) -> None:
        """全屏 / 还原。

        翻记录时经常要同时看很多列、右边还要摆两张取证图，把窗口摊满整屏比一格一格拖
        边框快得多。还原时回到「进全屏之前是不是最大化」，而不是一律回到普通尺寸。
        """
        if self.isFullScreen():
            if self._maximized_before_fullscreen:
                self.showMaximized()
            else:
                self.showNormal()
        else:
            self._maximized_before_fullscreen = self.isMaximized()
            self.showFullScreen()
        self._sync_fullscreen_button()

    def _sync_fullscreen_button(self) -> None:
        self.fullscreen_btn.setText("还原" if self.isFullScreen() else "全屏")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
            return
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            # 全屏下 Esc 先退出全屏。直接关掉整个查看器会让人以为刚筛出来的记录丢了。
            self.toggle_fullscreen()
            return
        super().keyPressEvent(event)

    # -- 右栏 ---------------------------------------------------------------

    def _build_detail_side(self) -> QWidget:
        """右栏整体可滚动：详情十来行加两张图，本来就比一屏高。"""
        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        # 竖滚动条常驻。默认是「需要时才出现」，而它一出现就把视口宽度削掉十几像素，
        # 内容跟着重新折行、高度又变 —— QScrollArea 的 updateScrollBars 是**同步**回调
        # （内容控件收到 Resize 就再算一次），尺寸在这两个状态之间来回摆时会一路递归到
        # 栈溢出。让视口宽度恒定，这条回路就不存在了；代价是内容不高时右边也留着一条
        # 滚动条的宽度。
        self.details_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.details_content = QWidget()
        self.details_layout = QVBoxLayout(self.details_content)
        self.details_scroll.setWidget(self.details_content)
        return self.details_scroll

    def _clear_details(self) -> None:
        """清空右栏。

        只往这个布局里放控件、不放子布局：取出来的控件直接与父级脱钩，子布局里的控件
        却仍挂在 details_content 上，会留在右栏里出不去。
        """
        while self.details_layout.count():
            widget = self.details_layout.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)

    def _show_details(self, event: AlarmEvent | None) -> None:
        self._clear_details()
        if event is None:
            hint = _detail_hint("在左侧选一条记录，这里显示它的详情与两张取证图。")
            hint.setWordWrap(True)
            self.details_layout.addWidget(hint)
            self.details_layout.addStretch()
            return
        for label, value in _detail_rows(event):
            self.details_layout.addWidget(_detail_label(label, value))
        self.details_layout.addWidget(
            _detail_hint("提示：双击截图可用系统默认程序打开原图")
        )
        # 两张取证图从上到下排（进入在前、报警在后）。右栏是竖的，并排摆每张只剩半栏宽，
        # 而取证图里要看清的是人和区域边界。
        for title, path in _screenshot_entries(event):
            self.details_layout.addWidget(_detail_hint(title))
            self.details_layout.addWidget(
                _screenshot_widget(title, path, event, self._resolver, self)
            )
        self.details_layout.addStretch()

    # -- 记录与筛选 ---------------------------------------------------------

    def reload(self) -> None:
        """重新读一遍记录文件，再重画下拉与表格。

        打开时和点「刷新」时都走这里：检测线程在查看器开着的时候照样在写记录，刷新读的
        是磁盘上的最新状态。
        """
        selected = self._selected_event()
        try:
            self._events = self._store.load_all()
        except OSError as error:
            logger.exception("Unable to load alarm event history")
            QMessageBox.warning(self, "读取报警记录失败", str(error))
            self._events = []
        # 长文本列要多少宽度在这里量一次（供铺满时补宽），筛选与缩放都不再重算。
        self._measure_content_widths()
        self._fill_filter(
            self.mode_combo,
            "全部模式",
            sorted({_mode_label(event.operation_mode) for event in self._events}),
        )
        self._fill_filter(
            self.status_combo,
            "全部状态",
            sorted({event.status_label for event in self._events}),
        )
        self._apply_filter()
        if selected is None:
            # 默认落在最新那条上：打开一个空白的右栏，还得自己先猜哪条该看。
            self._select_row(self.table.rowCount() - 1)
        else:
            self._select_session(selected.session_id)

    @staticmethod
    def _fill_filter(combo: QComboBox, placeholder: str, values: list[str]) -> None:
        """按记录里实际出现过的值填下拉，并保住用户已经选的那一项。

        不写死选项列表：这个点位可能从来没有过视频模式的记录，列一个选了就空表的下拉项
        没有意义。填的过程中关掉信号 —— 重填会连着触发好几次筛选。
        """
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(placeholder, "")
        for value in values:
            combo.addItem(value, value)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)

    def _matches(self, event: AlarmEvent, window_start: datetime | None) -> bool:
        if window_start is not None:
            recorded = _record_time(event)
            # 认不出时间的记录不参与时间筛选：说它在窗口内是编的，说它不在也是编的。
            # 计数那行会显示「显示 N / 共 M 条」，被筛掉多少看得见。
            if recorded is None or recorded < window_start:
                return False
        mode = self.mode_combo.currentData() or ""
        if mode and _mode_label(event.operation_mode) != mode:
            return False
        status = self.status_combo.currentData() or ""
        if status and event.status_label != status:
            return False
        # 搜索只匹配表格里看得见的列（不含两条截图路径）：搜出来的行必须一眼能看出为什么
        # 命中 —— 否则搜「12」会命中一堆文件名里带 12 的记录，而表里没有任何 12。
        # 空格分开的多个词之间是「与」：搜「北侧 12」要求区域和目标 ID 都命中，比只能搜
        # 一个词的用法更接近「找那条记录」。
        terms = self.search_edit.text().lower().split()
        if not terms:
            return True
        haystack = " ".join(self._row_values(event)).lower()
        return all(term in haystack for term in terms)

    @staticmethod
    def _row_values(event: AlarmEvent) -> list[str]:
        """表格一行：只放「认出是哪条记录」用得上的列。

        截图路径那两列不要 —— 它们在右栏是看得见的图；要看路径的话，鼠标停在图上有。
        """
        row = event.to_row()  # 记录时间…闯入时长在前九列，两条截图路径在后两列
        return [row[0], _mode_label(event.operation_mode), *row[1:9]]

    def _apply_filter(self, *ignored: object) -> None:
        del ignored
        selected = self._selected_event()
        # 时间窗口在这里算一次：它是「现在」往前推出来的，逐条记录各算一次既慢又会
        # 在跨秒时算出两个不同的下界。
        window_start = _time_filter_start(
            self.time_combo.currentData() or "", datetime.now()
        )
        visible = [
            event for event in self._events if self._matches(event, window_start)
        ]
        # 重填期间关掉选中信号：搜索框里打字是连着来的，每敲一个字都重画一次右栏（两张图
        # 要重新解码）会明显发顿。重填完再把选中行按会话 ID 找回来。
        #
        # 先清掉选中状态：选中是按行号记的，而重填之后同一行已经是另一条记录了 —— 不清的话
        # 右栏会显示成「用户没选过的那条」的详情。
        #
        # 能复用的单元格就复用：一次筛选是几千行 × 十来列，每次都新建 QTableWidgetItem 的
        # 话，在两千条记录上单次重填要 200 ms 上下，敲一个字卡一下。注意已有单元格不能再
        # 走一遍 setItem —— Qt 会当成「这个单元格已经有主了」并打一条警告。
        self.table.blockSignals(True)
        self.table.clearSelection()
        self.table.setRowCount(len(visible))
        for row, event in enumerate(visible):
            for column, value in enumerate(self._row_values(event)):
                item = self.table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(row, column, item)
                item.setText(value)
                item.setToolTip(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, event)
            # 占位列也要有 item：Qt 只给有 item 的格子画行底色，没有 item 的那片就是纯白
            # —— 空白区看着像"表没画完"，原因就在这里（实测同一行：真实列内 #f7f7f7、
            # 空白区 #ffffff，条纹在那里断掉）。
            if self.table.item(row, self.FILLER_COLUMN) is None:
                self.table.setItem(row, self.FILLER_COLUMN, QTableWidgetItem())
        self.table.blockSignals(False)
        self.count_label.setText(f"显示 {len(visible)} / 共 {len(self._events)} 条")
        restored = selected is not None and self._select_session(selected.session_id)
        if not restored:
            # 原来选的那条被筛掉了（或者本来就没选）：右栏回到提示语，而不是留着上一条
            # 的详情让人以为它还在表里。
            self._show_details(self._selected_event())

    def _selected_event(self) -> AlarmEvent | None:
        items = self.table.selectedItems()
        if not items:
            return None
        item = self.table.item(items[0].row(), 0)
        event = item.data(Qt.ItemDataRole.UserRole) if item else None
        return event if isinstance(event, AlarmEvent) else None

    def _select_row(self, row: int) -> None:
        if row < 0 or row >= self.table.rowCount():
            return
        self.table.selectRow(row)
        self.table.scrollToItem(self.table.item(row, 0))

    def _select_session(self, session_id: str) -> bool:
        """把选中行放回指定会话；它不在当前筛选结果里时返回 False。"""
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            event = item.data(Qt.ItemDataRole.UserRole) if item else None
            if isinstance(event, AlarmEvent) and event.session_id == session_id:
                self._select_row(row)
                return True
        return False

    def _on_selection_changed(self) -> None:
        self._show_details(self._selected_event())


class EventPanel(QGroupBox):
    HEADERS = [
        "记录时间", "视频源", "区域", "目标ID", "进入时刻", "报警时刻",
        "退出时刻", "状态", "闯入时长", "进入取证", "报警取证",
    ]
    # 吸收多余宽度的那一列（「退出时刻」）。定义成常量：列宽模式要按它恢复，两处写死
    # 数字迟早会对不上。
    STRETCH_COLUMN = 6

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
        _configure_event_table(self.table, self.HEADERS)
        # 触摸屏上手指在表上直接拖着滚（面板这张表最长，手指划比重拖滚动条自然）
        enable_touch_scrolling(self.table)
        # 面板只有四百来像素宽，装不下十一个按内容铺开的列，所以这里固定按内容铺满，
        # 让「退出时刻」吸收多余宽度（查看器那边才是可拖的）。
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.STRETCH_COLUMN, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(220)
        self.table.cellDoubleClicked.connect(self._show_details)
        layout.addWidget(self.table)
        # 面板上原来这个位置是「清空记录」。它换成了查看器：报警记录按留存策略是永久保留的
        # （只清截图），而一屏 200 条、还不带筛选的表，翻旧记录时基本没用。
        self.view_btn = QPushButton("查看记录")
        self.view_btn.setToolTip(
            "打开记录查看器：左边全部记录（可筛选、可搜索），右边选中那条的详情与两张取证图。"
        )
        layout.addWidget(self.view_btn)

    def _show_details(self, row: int, column: int) -> None:
        item = self.table.item(row, 0)
        event = item.data(Qt.ItemDataRole.UserRole) if item else None
        if isinstance(event, AlarmEvent):
            AlarmDetailDialog(event, self._resolver, self).exec()

    def append_event(self, event: AlarmEvent) -> None:
        self._write_row(event)
        self.table.scrollToBottom()

    def load_events(self, events: list[AlarmEvent]) -> None:
        """成批填表（启动时那 200 条）。

        填的时候必须把列宽模式切掉 ResizeToContents：11 列都按内容自适应时，每写一个
        单元格都会触发一次列宽重算，而每次重算要遍历所有行 —— 200 条 × 11 列下来是
        **秒级**（实测启动时这一项就占 1.6 秒）。填完再恢复模式，Qt 只重算一遍。
        """
        header = self.table.horizontalHeader()
        self.table.setUpdatesEnabled(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        try:
            for event in events:
                self._write_row(event)
        finally:
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(self.STRETCH_COLUMN, QHeaderView.ResizeMode.Stretch)
            self.table.setUpdatesEnabled(True)
        self.table.scrollToBottom()

    def _write_row(self, event: AlarmEvent) -> None:
        row = self._rows_by_session.get(event.session_id)
        if row is None:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._rows_by_session[event.session_id] = row
        for column, value in enumerate(event.to_row()):
            item = self.table.item(row, column)
            if item is None:
                # 同一个单元格不能再 setItem 一次：Qt 会当成「已经有主了」打警告，而这条
                # 路径每次会话状态变化都会走一遍（进入 / 报警 / 结束各一次）。
                item = QTableWidgetItem()
                self.table.setItem(row, column, item)
            item.setText(value)
            item.setToolTip(value)
            if column == 0:
                item.setData(Qt.ItemDataRole.UserRole, event)

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
        self._retention_timer.start()
        # 下面这些都不影响窗口能不能用，全部推到窗口画出来之后再做 —— 启动时它们是一段
        # 白屏等待（实测：设备探测要 import torch 约 4 秒，占用统计要遍历截图目录几百个
        # 文件，低功耗可用性检查要 import openvino）。现在窗口立刻可用，状态栏先写着
        # 「正在检测推理设备…」；「生效设置」那条日志也等设备探测回来再写（要记真实设备）。
        #
        # 放在事件循环里做还有个好处：自动化测试不跑事件循环，就不会在无准备的情况下
        # 动磁盘、也不会去 import torch。配置读坏而回落默认值时不自动清理（见 _run_retention）。
        QTimer.singleShot(0, self._probe_devices)
        QTimer.singleShot(0, self._refresh_storage_usage)
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

        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setWidget(self.sidebar_pages)
        self.sidebar_scroll.setMinimumWidth(420)
        # 侧栏是最需要手指滑动的地方：面板竖着叠，窗口矮时要滚半天
        enable_touch_scrolling(self.sidebar_scroll)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.video_widget)
        splitter.addWidget(self.sidebar_scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([850, 430])
        self.setCentralWidget(splitter)

    def _load_config(self) -> AppConfig:
        self._config_write_refused = False
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
            # 改名也可能失败（文件被同步盘/杀软/别的程序占着）。那时原文件还在原处，
            # 而且我们读不动它 —— 这份文件只能由人来处置，本次运行就一律不落盘：
            # 否则用户关窗时那份默认配置会把它盖掉，区域与设置一起没了（改名失败时
            # 连备份都没有）。注意判据是「改名失败且文件仍在」，不是「配置不可信」——
            # 隔离成功时文件已经被挪走，本会话的改动理应照常存下来。
            self._config_writable = backup is not None or not self.config_store.path.exists()
            if backup:
                hint = f"\n原文件已备份为 {backup.name}，可手工修好后放回。"
            elif self._config_writable:
                hint = "\n原文件不存在。"
            else:
                hint = (
                    "\n原文件没能挪走（可能被其他程序占用），本次运行不会写入配置，"
                    "原文件保持不动。"
                )
            QMessageBox.warning(
                self,
                "配置读取失败",
                f"将使用默认配置。{hint}\n{error}",
            )
            return AppConfig()
        self._config_trusted = True
        self._config_writable = True
        return loaded

    def _load_controls(self) -> None:
        # 这一步内部会载入报警记录（见 _apply_operation_mode），所以下面不再重复载入。
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
        # 推理设备要 import torch 才知道（实测约 4 秒），不能压在这条启动路径上：交给
        # 后台线程探测（见 _probe_devices），结果回来之前「打开」先禁用着 —— 否则用户
        # 可能带着一个还没填好的设备列表就开始检测。
        self.source_panel.set_status("正在检测推理设备…")
        self.source_panel.open_btn.setEnabled(False)
        self._settings_binder.load(self.config)
        self._refresh_notification_status()
        self.playback_panel.loop_cb.setChecked(self.config.loop_playback)
        speed_value = max(1, min(16, round(self.config.playback_speed / 0.25)))
        self.playback_panel.speed_slider.setValue(speed_value)
        self._restore_active_profile()
        self.video_widget.set_zones(self.zones)
        self.video_widget.set_active_zone(self.active_zone)
        self._refresh_zone_panel()
        self._refresh_profiles()
        self._set_profile_dirty(False)
        self._update_playback_enabled()

    def _probe_devices(self) -> None:
        """在后台算可用推理设备（理由见 _load_controls）。"""
        self._device_probe = DeviceProbe(self)
        self._device_probe.ready.connect(self._apply_device_options)
        self._device_probe.start()

    def _apply_device_options(self, device_options: list[tuple[str, str]]) -> None:
        """设备探测结果回来了：填下拉框、恢复上次选的设备、放开「打开」。"""
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
        self.source_panel.open_btn.setEnabled(True)
        self._update_cpu_low_power_availability()
        # 「生效设置」这条日志要记的是真正生效的设备，所以等探测回来再写（见 __init__）。
        self._log_effective_settings()

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
        self.event_panel.view_btn.clicked.connect(self._show_event_history)

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

    def _restore_active_profile(self) -> None:
        """把 config.json 里记着的那个配置组读回来。

        读不出来时不能把异常留给调用方 —— 这段跑在 ``__init__`` 里，异常会让窗口根本
        建不起来，用户每次启动都只看到一个「程序出错」，只能自己猜到去 profiles 目录
        里删文件。于是改成：坏文件改名留档，区域用 config.json 里那一份（上次退出时
        写下的，正是当时生效的区域），再把这件事说出来。指向它的指针一并摘掉 ——
        那个配置组已经不在原处了，留着只会让界面上挂一条点不动的条目。
        """
        name = self.config.active_profile
        if not name:
            return
        if not self.profile_store.is_valid_name(name):
            # config.json 能手改，改出一个带斜杠/冒号的名字就永远对应不上文件。留着
            # 它只会让配置组列表挂一条点不动的条目，顺手摘掉。
            logger.warning("config.json 里的配置组名 %r 不是合法的文件名，已清除", name)
            self.config.active_profile = ""
            self._schedule_config_save()
            return
        if not self.profile_store.exists(name):
            # 名字合法、文件不在（被手工删掉或挪走）。指针留着：用户重新点一次
            # 「保存配置组」就把它写回去了。区域本身来自 config.json，不受影响。
            return
        try:
            profile = self.profile_store.load(name)
        except ValueError as error:
            backup = self.profile_store.quarantine(name)
            self.config.active_profile = ""
            self._schedule_config_save()
            hint = (
                f"原文件已备份为 {backup.name}，可手工修好后放回 profiles 目录。"
                if backup
                else "原文件仍在 profiles 目录里，未能改名留档。"
            )
            logger.warning("配置组 %s 读取失败，已改用 config.json 里的区域：%s", name, error)
            QMessageBox.warning(
                self,
                "配置组读取失败",
                f"配置组“{name}”读不出来，已改用 config.json 里的区域。\n{hint}\n{error}",
            )
            self.source_panel.set_status(f"配置组“{name}”读取失败，已用 config.json 里的区域")
            return
        self.zones = profile.zones
        self.active_zone = self.zones[0] if self.zones else None

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
        try:
            # 立刻建出空文件，而不是攒到用户点「保存配置组」再建：否则这个名字只是
            # 内存里的一个指针 —— 列表里挂着一条没有对应文件的条目，切到别的配置组
            # 之后再点它的「应用」只会报「加载配置组失败」。先建文件，让界面上的
            # 每一条都是真的。
            self.profile_store.save(ZoneProfile(name=name, zones=[]))
        except ValueError as error:
            QMessageBox.warning(self, "新建失败", str(error))
            return
        self.config.active_profile = name
        self.zones = []
        self.active_zone = None
        self.video_widget.set_zones(self.zones)
        self.video_widget.set_active_zone(self.active_zone)
        self._refresh_zone_panel()
        self._refresh_profiles()
        self._set_profile_dirty(True)
        self._schedule_config_save()

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
        回落默认值的情况也就能一眼看出来。可写位一起记：它决定了这次运行会不会把
        设置写回文件，「改了设置重启就丢」能靠这两项一眼对上。
        """
        logger.info(
            "生效设置: 模式=%s 视频源=%s 设备=%s 低功耗=%s 模型=%s "
            "留存=%s天/%sMB 远程通知=%s 配置可信=%s 配置可写=%s",
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
            getattr(self, "_config_writable", True),
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
        # 推理那条链（ultralytics + supervision + trackers，实测约 8 秒）是延迟导入的：
        # 它只有真的开始推理时才需要，压在启动路径上只是白屏。这里先让状态栏把话说出来
        # 再导 —— 否则窗口会卡住好几秒，而界面上一个字都不变，看着像死机。
        self.source_panel.set_status("正在加载推理组件…")
        QApplication.processEvents()
        from detection_worker import DetectionWorker

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

    def _show_event_history(self) -> None:
        """打开记录查看器。

        查看器自己读全量记录、也不按运行模式过滤（面板那张表只装当前模式的最近 200 条），
        所以打开时会重新读一遍文件 —— 检测正在跑的话，刚报的那条警也在里面。

        列宽在这里进、出：进去的是上次拖过的宽度，出来的是这次拖成的样子，关窗即存。
        """
        dialog = AlarmHistoryDialog(
            self.event_store,
            self.event_store.resolve_screenshot,
            self,
            column_widths=self.config.history_column_widths,
        )
        dialog.exec()
        self.config.history_column_widths = dialog.column_widths()
        self._schedule_config_save()

    def _schedule_config_save(self) -> None:
        self._save_timer.start()

    def _save_config(self) -> None:
        if not self._config_writable:
            # 启动时读不到 config.json、又没能把它改名留档：这份文件只能由人来处置，
            # 本次运行一律不落盘（理由见 _load_config）。不说一声的话，用户改完设置
            # 关窗，会以为都存下了。
            if not self._config_write_refused:
                self._config_write_refused = True
                logger.warning("config.json 未能读取且未能备份，本次运行不写入配置")
                self.source_panel.set_status(
                    "设置未写入：启动时读不到 config.json 且无法备份，原文件保持不动"
                )
            return
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
            self._save_active_profile()
        except OSError as error:
            # 磁盘满、目录只读、文件被占用都会走到这里。界面不提示的话，用户看到的
            # 是一份「改了也存不下」的设置，而没有任何线索。
            logger.exception("Unable to save configuration")
            self.source_panel.set_status(f"配置未能保存：{error}")

    def _save_active_profile(self) -> None:
        """区域有改动时，把当前配置组也一起落盘。

        以前只有「保存配置组」按钮会写 ``profiles\\名字.json``，而**启动时配置组优先**
        （_restore_active_profile 用它覆盖 config.json 里的区域）—— 于是画完区域直接
        关窗，改动只躺在 config.json 里，下次启动被配置组盖掉，等于白画。现在跟着
        防抖保存一起写，状态栏那句「未保存」也会跟着变成「已保存」。

        没有当前配置组（从没命名过）时就只写 config.json —— 下次启动会回落到它，
        改动同样不会丢，所以不必替用户凭空造一个配置组。
        """
        name = self.config.active_profile
        if not self.profile_dirty or not name or not self.profile_store.exists(name):
            return
        try:
            self.profile_store.save(ZoneProfile(name=name, zones=self.zones))
        except ValueError as error:
            # 名字非法之类：配置组没写成，但 config.json 里那份已经落盘了。
            logger.warning("配置组自动保存失败：%s", error)
            return
        self._set_profile_dirty(False)

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
