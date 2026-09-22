"""触摸滑动与 Shift+滚轮横向滚动。

钉住的契约：

* 手指在滚动区域上拖着滚（QScroller 接管**单指**拖动），但**只抓单指手势** —— 抓了
  鼠标拖动的话，表格里的拖选、视频画面上按住加点/拖顶点都会被抢走；
* 视频画面刻意不抓：那里手指拖动本来就该等价于鼠标拖动（画警戒区）；
* Shift+滚轮 = 横向滚动；横向没得滚时放行，仍旧纵向滚（否则窗口够宽、列都放得下时
  Shift+滚轮会变成什么都不动）。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea, QScroller, QWidget

import main_window
from alarm_service import EventStore
from main_window import (
    AlarmHistoryDialog,
    MainWindow,
    install_shift_wheel_scroll,
)


def send_wheel(
    target: QWidget,
    *,
    dx: int = 0,
    dy: int = 0,
    modifiers=Qt.KeyboardModifier.NoModifier,
) -> None:
    position = QPointF(target.rect().center())
    QApplication.sendEvent(
        target,
        QWheelEvent(
            position,
            target.mapToGlobal(position),
            QPoint(0, 0),
            QPoint(dx, dy),
            Qt.MouseButton.NoButton,
            modifiers,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        ),
    )


def swipe(device, widget: QWidget, start: QPoint, end: QPoint, steps: int = 3) -> None:
    """模拟一次手指拖动。

    每个触点**分开提交**并留出间隔：同一次提交里的触点时间戳相同、速度算成 0，QScroller
    会当成「没动」（实测状态一直停在 Inactive）。这是合成触摸事件的坑，不是产品问题。
    """
    press = QTest.touchEvent(widget, device)
    press.press(0, start, widget.window())
    press.commit()
    for index in range(1, steps + 1):
        QTest.qWait(30)
        ratio = index / steps
        point = QPoint(
            round(start.x() + (end.x() - start.x()) * ratio),
            round(start.y() + (end.y() - start.y()) * ratio),
        )
        move = QTest.touchEvent(widget, device)
        move.move(0, point, widget.window())
        move.commit()


class TouchScrollingTests(unittest.TestCase):
    """``enable_touch_scrolling`` 抓了哪些区域、抓的是哪种手势。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        # 触摸设备只建一次（要在 QApplication 之后）
        cls.touch = QTest.createTouchDevice()

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = EventStore(self.root / "events")

    def fill_rows(self, count: int = 60) -> None:
        from models import AlarmEvent

        for index in range(count):
            self.store._append(
                {
                    "schema_version": 2,
                    "action": "opened",
                    "event": EventStore._event_payload(
                        AlarmEvent(
                            source="rtsp://camera/live",
                            zone_name=f"区域{index}",
                            track_id=str(index),
                            entered_at_seconds=1.0,
                            alarm_at_seconds=3.0,
                            wall_time="2026-09-21 10:00:00",
                            operation_mode="monitor",
                            session_id=f"session-{index}",
                        )
                    ),
                }
            )

    def test_the_viewer_can_be_swiped(self) -> None:
        dialog = AlarmHistoryDialog(self.store, self.store.resolve_screenshot, None)
        self.addCleanup(dialog.deleteLater)

        for area in (dialog.table, dialog.details_scroll):
            with self.subTest(area=type(area).__name__):
                self.assertTrue(
                    QScroller.hasScroller(area.viewport()),
                    "这个区域没装上手指滑动",
                )

    def test_swiping_the_table_scrolls_it(self) -> None:
        """真的拖一次：触摸拖动 = 滚动（不只是「装上了」这个结构检查）。"""
        self.fill_rows()
        dialog = AlarmHistoryDialog(self.store, self.store.resolve_screenshot, None)
        self.addCleanup(dialog.deleteLater)
        dialog.resize(900, 700)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()
        table = dialog.table
        bar = table.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0, "这条要有纵向溢出")
        bar.setValue(bar.maximum() // 2)
        self.app.processEvents()
        before = bar.value()
        viewport = table.viewport()
        center = viewport.rect().center()

        swipe(
            self.touch,
            viewport,
            QPoint(center.x(), center.y() - 60),
            QPoint(center.x(), center.y() + 60),
        )
        self.app.processEvents()

        self.assertLess(bar.value(), before, "手指往下拖应该把内容往下带（滚动值变小）")

    def test_the_main_window_can_be_swiped(self) -> None:
        """侧栏、常驻报警记录表、配置组与区域两个列表 —— 触摸屏上最常划的几处。"""
        for name, path in (
            ("CONFIG_PATH", self.root / "config.json"),
            ("EVENTS_DIR", self.root / "events"),
            ("PROFILES_DIR", self.root / "profiles"),
        ):
            patcher = patch.object(main_window, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        window = MainWindow()
        self.addCleanup(window.deleteLater)

        areas = {
            "侧栏": window.sidebar_scroll,
            "报警记录表": window.event_panel.table,
            "配置组列表": window.zone_panel.profile_list,
            "区域列表": window.zone_panel.zone_list,
        }
        for label, area in areas.items():
            with self.subTest(area=label):
                self.assertTrue(
                    QScroller.hasScroller(area.viewport()), f"{label}没装上手指滑动"
                )

    def test_the_video_widget_keeps_mouse_dragging(self) -> None:
        """视频画面上手指拖动 = 画警戒区（加点、拖顶点），不能被 QScroller 接管。"""
        with (
            patch.object(main_window, "CONFIG_PATH", self.root / "config.json"),
            patch.object(main_window, "EVENTS_DIR", self.root / "events"),
            patch.object(main_window, "PROFILES_DIR", self.root / "profiles"),
        ):
            window = MainWindow()
        self.addCleanup(window.deleteLater)

        self.assertFalse(
            QScroller.hasScroller(window.video_widget),
            "视频画面上不该装手指滑动",
        )

    def test_a_mouse_drag_still_selects_instead_of_scrolling(self) -> None:
        """鼠标按住拖动仍然是拖选，不能被滚动接管。

        只抓 ``TouchGesture`` 就是为了这个：抓了 ``LeftMouseButtonGesture`` 的话，表格里的
        拖选、视频上画区域都会变成滚动 —— 这条测试会在那种改动下失败。
        """
        self.fill_rows()
        dialog = AlarmHistoryDialog(self.store, self.store.resolve_screenshot, None)
        self.addCleanup(dialog.deleteLater)
        dialog.resize(900, 700)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()
        table = dialog.table
        bar = table.verticalScrollBar()
        bar.setValue(bar.maximum() // 2)
        self.app.processEvents()
        before = bar.value()
        viewport = table.viewport()

        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=QPoint(200, 300))
        QTest.mouseMove(viewport, QPoint(200, 200))
        self.app.processEvents()
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=QPoint(200, 200))
        self.app.processEvents()

        self.assertEqual(bar.value(), before, "鼠标拖动被当成滚动了")
        self.assertEqual(len(table.selectionModel().selectedRows()), 1, "鼠标拖动该选中一行")


class ShiftWheelTests(unittest.TestCase):
    """Shift+滚轮横向滚动（应用级事件过滤器）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = EventStore(self.root / "events")
        # 走 main.py 用的那条安装路径，测试与运行期不会是两套
        self.scroll_filter = install_shift_wheel_scroll(self.app)
        self.addCleanup(self.app.removeEventFilter, self.scroll_filter)

    def wide_viewer(self) -> AlarmHistoryDialog:
        """列宽之和远超表宽 —— 横向一定有得滚。"""
        dialog = AlarmHistoryDialog(self.store, self.store.resolve_screenshot, None)
        self.addCleanup(dialog.deleteLater)
        dialog.resize(900, 700)
        dialog.show()
        self.addCleanup(dialog.close)
        for column in range(len(AlarmHistoryDialog.HEADERS)):
            dialog.table.setColumnWidth(column, 200)
        self.app.processEvents()
        return dialog

    def fill_rows(self, count: int = 60) -> None:
        from models import AlarmEvent

        for index in range(count):
            self.store._append(
                {
                    "schema_version": 2,
                    "action": "opened",
                    "event": EventStore._event_payload(
                        AlarmEvent(
                            source="rtsp://camera/live",
                            zone_name=f"区域{index}",
                            track_id=str(index),
                            entered_at_seconds=1.0,
                            alarm_at_seconds=3.0,
                            wall_time="2026-09-21 10:00:00",
                            operation_mode="monitor",
                            session_id=f"session-{index}",
                        )
                    ),
                }
            )

    def test_shift_wheel_scrolls_the_table_sideways(self) -> None:
        self.fill_rows()
        dialog = self.wide_viewer()
        table = dialog.table
        self.assertGreater(table.horizontalScrollBar().maximum(), 0, "这条要有横向溢出")
        table.horizontalScrollBar().setValue(0)
        vertical = table.verticalScrollBar().value()

        send_wheel(
            table.viewport(), dy=-120, modifiers=Qt.KeyboardModifier.ShiftModifier
        )

        self.assertGreater(table.horizontalScrollBar().value(), 0, "横向没滚")
        self.assertEqual(
            table.verticalScrollBar().value(), vertical, "Shift+滚轮不该同时也纵向滚"
        )

    def test_a_plain_wheel_still_scrolls_vertically(self) -> None:
        self.fill_rows()
        dialog = self.wide_viewer()
        table = dialog.table
        table.horizontalScrollBar().setValue(0)
        table.verticalScrollBar().setValue(table.verticalScrollBar().maximum() // 2)
        before = table.verticalScrollBar().value()

        send_wheel(table.viewport(), dy=-120)

        self.assertNotEqual(table.verticalScrollBar().value(), before, "普通滚轮该照旧纵向滚")
        self.assertEqual(table.horizontalScrollBar().value(), 0, "普通滚轮不该横向滚")

    def test_shift_wheel_falls_back_to_vertical_without_sideways_room(self) -> None:
        """横向没得滚时放行、仍旧纵向滚。

        否则窗口够宽、十列都放得下的时候，Shift+滚轮会变成什么都不动 —— 现场会觉得
        「按住 Shift 就滚不动了」。
        """
        self.fill_rows()
        dialog = AlarmHistoryDialog(self.store, self.store.resolve_screenshot, None)
        self.addCleanup(dialog.deleteLater)
        dialog.resize(1600, 700)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()
        table = dialog.table
        for column in range(len(AlarmHistoryDialog.HEADERS)):
            table.setColumnWidth(column, 40)
        self.app.processEvents()
        self.assertEqual(table.horizontalScrollBar().maximum(), 0, "这条不该有横向溢出")
        table.verticalScrollBar().setValue(table.verticalScrollBar().maximum() // 2)
        before = table.verticalScrollBar().value()

        send_wheel(
            table.viewport(), dy=-120, modifiers=Qt.KeyboardModifier.ShiftModifier
        )

        self.assertNotEqual(
            table.verticalScrollBar().value(), before, "没有横向可滚时该退化成纵向滚动"
        )

    def test_it_works_on_a_plain_scroll_area(self) -> None:
        """QScrollArea（右栏、侧栏都是）同样生效。"""
        area = QScrollArea()
        self.addCleanup(area.deleteLater)
        content = QWidget()
        content.setMinimumWidth(2000)
        area.setWidget(content)
        area.resize(400, 300)
        area.show()
        self.addCleanup(area.close)
        self.app.processEvents()
        self.assertGreater(area.horizontalScrollBar().maximum(), 0)

        send_wheel(
            area.viewport(), dy=-120, modifiers=Qt.KeyboardModifier.ShiftModifier
        )

        self.assertGreater(area.horizontalScrollBar().value(), 0)


if __name__ == "__main__":
    unittest.main()
