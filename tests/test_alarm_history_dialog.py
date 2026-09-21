"""「查看记录」查看器：左栏全部记录加筛选搜索，右栏详情与两张取证图。

这里钉住的契约：

* 查看器看的是**全部**记录 —— 不是常驻面板那最近 200 条，也不只当前运行模式；
* 筛选与搜索真的收窄表格，选中那条的详情在右栏，两张取证图按「进入在上、报警在下」
  从上到下排开；
* 双击交出去的必须是**原图文件的真实路径**（不是预览、不是相对路径），图片不可用时
  则不给双击入口；
* 面板上那个位置原来是「清空记录」，现在是打开查看器的按钮。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
    QLabel,
    QSplitter,
    QWidget,
)

import main_window
from alarm_service import EventStore
from main_window import AlarmHistoryDialog, EventPanel, PreviewImageLabel
from models import AlarmEvent


class AlarmHistoryDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = EventStore(self.root / "events")
        self.frame = np.full((40, 60, 3), 120, dtype=np.uint8)

    def record(
        self,
        *,
        zone: str = "北侧入口",
        track: str = "12",
        mode: str = "monitor",
        alarm: bool = True,
        wall_time: str | None = None,
    ) -> AlarmEvent:
        """真写一条记录（含两张取证截图），跟检测线程落盘的那条路径一致。"""
        event = AlarmEvent(
            source="rtsp://camera/live",
            zone_name=zone,
            track_id=track,
            entered_at_seconds=1.0,
            alarm_at_seconds=3.0 if alarm else None,
            wall_time=wall_time or self.now(),
            operation_mode=mode,
            exited_at_seconds=6.0,
            duration_seconds=5.0,
        )
        self.store.open_session(event, self.frame)
        if alarm:
            self.store.mark_alarmed(event, self.frame)
        self.store.close_session(event)
        return event

    @staticmethod
    def now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def days_ago(cls, days: float) -> str:
        return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

    def dialog(self, parent: QWidget | None = None) -> AlarmHistoryDialog:
        dialog = AlarmHistoryDialog(self.store, self.store.resolve_screenshot, parent)
        self.addCleanup(dialog.deleteLater)
        return dialog

    def detail_widgets(self, dialog: AlarmHistoryDialog) -> list[object]:
        layout = dialog.details_layout
        return [layout.itemAt(index).widget() for index in range(layout.count())]

    def detail_text(self, dialog: AlarmHistoryDialog) -> str:
        return "\n".join(
            widget.text()
            for widget in self.detail_widgets(dialog)
            if isinstance(widget, QLabel)
        )

    def detail_sequence(self, dialog: AlarmHistoryDialog) -> list[str]:
        """右栏里控件的顺序：文字控件给它的文字，预览图给它打开的原图路径。"""
        sequence = []
        for widget in self.detail_widgets(dialog):
            if isinstance(widget, PreviewImageLabel):
                sequence.append(widget.toolTip().splitlines()[-1])
            elif isinstance(widget, QLabel):
                sequence.append(widget.text())
        return sequence

    def column(self, dialog: AlarmHistoryDialog, header: str) -> list[str]:
        index = AlarmHistoryDialog.HEADERS.index(header)
        return [
            dialog.table.item(row, index).text()
            for row in range(dialog.table.rowCount())
        ]

    # -- 左栏 ---------------------------------------------------------------

    def test_the_viewer_lists_every_record_of_every_mode(self) -> None:
        self.record(zone="北侧入口", mode="monitor")
        self.record(zone="南侧通道", mode="video")

        dialog = self.dialog()

        self.assertEqual(self.column(dialog, "区域"), ["北侧入口", "南侧通道"])
        self.assertEqual(self.column(dialog, "运行模式"), ["监控模式", "视频模式"])

    def test_the_viewer_is_not_limited_to_the_panels_recent_200(self) -> None:
        """面板那张表只装最近 200 条（启动时填一遍，不能拖慢启动）；翻记录要的是全部。"""
        for index in range(205):
            event = AlarmEvent(
                source="rtsp://camera/live",
                zone_name=f"区域{index}",
                track_id=str(index),
                entered_at_seconds=1.0,
                alarm_at_seconds=3.0,
                wall_time="2026-09-21 10:00:00",
                operation_mode="monitor",
                session_id=f"session-{index}",
            )
            # 直接写 JSONL：这条测的是条数，不必为 205 条记录各存一张截图。
            self.store._append(
                {
                    "schema_version": 2,
                    "action": "opened",
                    "event": EventStore._event_payload(event),
                }
            )

        dialog = self.dialog()

        self.assertEqual(len(self.store.load_recent(operation_mode="monitor")), 200)
        self.assertEqual(dialog.table.rowCount(), 205)

    def test_search_narrows_the_table_and_can_be_cleared(self) -> None:
        self.record(zone="北侧入口", track="12")
        self.record(zone="南侧通道", track="77")

        dialog = self.dialog()
        dialog.search_edit.setText("南侧")

        self.assertEqual(self.column(dialog, "区域"), ["南侧通道"])
        self.assertEqual(dialog.count_label.text(), "显示 1 / 共 2 条")

        dialog.search_edit.setText("")

        self.assertEqual(len(self.column(dialog, "区域")), 2)

    def test_search_requires_every_word_to_match(self) -> None:
        """空格分开的多个词是「与」：搜「南侧 77」找的是那条记录，不是所有含南侧的。"""
        self.record(zone="北侧入口", track="12")
        self.record(zone="南侧通道", track="77")

        dialog = self.dialog()
        dialog.search_edit.setText("南侧 77")

        self.assertEqual(self.column(dialog, "区域"), ["南侧通道"])

        dialog.search_edit.setText("南侧 12")

        self.assertEqual(dialog.table.rowCount(), 0)

    def test_search_only_matches_what_the_table_shows(self) -> None:
        """截图路径（连同它里面的会话 ID）不进表格，也就不该被搜到。

        否则搜一串数字会命中一堆「表里根本看不到这个字」的记录，用户没法判断为什么它在。
        """
        event = self.record(zone="北侧入口", track="12")
        self.assertIn(event.session_id, event.alarm_screenshot_path, "文件名里确实带着它")

        dialog = self.dialog()
        dialog.search_edit.setText(event.session_id)

        self.assertEqual(dialog.table.rowCount(), 0)

    def test_the_time_filter_narrows_by_record_time(self) -> None:
        self.record(zone="刚刚", wall_time=self.days_ago(0))
        self.record(zone="三天前", wall_time=self.days_ago(3))
        self.record(zone="四十天前", wall_time=self.days_ago(40))

        dialog = self.dialog()
        self.assertEqual(len(self.column(dialog, "区域")), 3, "默认是全部时间")

        dialog.time_combo.setCurrentIndex(dialog.time_combo.findData("hour"))
        self.assertEqual(self.column(dialog, "区域"), ["刚刚"])

        dialog.time_combo.setCurrentIndex(dialog.time_combo.findData("today"))
        self.assertEqual(self.column(dialog, "区域"), ["刚刚"])

        dialog.time_combo.setCurrentIndex(dialog.time_combo.findData("week"))
        self.assertEqual(self.column(dialog, "区域"), ["刚刚", "三天前"])

        dialog.time_combo.setCurrentIndex(dialog.time_combo.findData("month"))
        self.assertEqual(self.column(dialog, "区域"), ["刚刚", "三天前"])

        dialog.time_combo.setCurrentIndex(0)
        self.assertEqual(len(self.column(dialog, "区域")), 3)

    def test_today_means_since_midnight_not_the_last_24_hours(self) -> None:
        """现场说的「今天」是日历上的今天。"""
        just_after_midnight = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        if just_after_midnight > datetime.now() - timedelta(hours=1):
            # 刚过零点时「今天」与「最近 1 小时」几乎重合，这条就没什么可测的了。
            self.skipTest("现在是凌晨，两个档位本来就几乎一样")
        self.record(zone="零点后", wall_time=just_after_midnight.strftime("%Y-%m-%d %H:%M:%S"))

        dialog = self.dialog()
        dialog.time_combo.setCurrentIndex(dialog.time_combo.findData("today"))
        self.assertEqual(self.column(dialog, "区域"), ["零点后"])

        dialog.time_combo.setCurrentIndex(dialog.time_combo.findData("hour"))
        self.assertEqual(dialog.table.rowCount(), 0)

    def test_the_time_filter_ignores_records_with_an_unreadable_time(self) -> None:
        """时间认不出来的记录不进时间窗口：说它在窗口内是编的，说它不在也是编的。"""
        self.record(zone="手改过的", wall_time="不知道什么时候")

        dialog = self.dialog()
        self.assertEqual(self.column(dialog, "区域"), ["手改过的"])

        dialog.time_combo.setCurrentIndex(dialog.time_combo.findData("month"))

        self.assertEqual(dialog.table.rowCount(), 0)
        self.assertEqual(dialog.count_label.text(), "显示 0 / 共 1 条")

    def test_the_time_filter_combines_with_the_search(self) -> None:
        self.record(zone="北侧入口", track="12", wall_time=self.days_ago(0))
        self.record(zone="北侧入口", track="77", wall_time=self.days_ago(40))
        self.record(zone="南侧通道", track="99", wall_time=self.days_ago(0))

        dialog = self.dialog()
        dialog.time_combo.setCurrentIndex(dialog.time_combo.findData("week"))
        dialog.search_edit.setText("北侧")

        self.assertEqual(self.column(dialog, "区域"), ["北侧入口"])
        self.assertEqual(self.column(dialog, "目标ID"), ["12"])

    # -- 列宽 ---------------------------------------------------------------

    def test_the_columns_can_be_dragged_and_reported(self) -> None:
        """列宽要能自由拖：不同点位关心的列不一样，按内容铺一遍只是个起点。"""
        dialog = self.dialog()
        header = dialog.table.horizontalHeader()

        self.assertTrue(
            all(
                header.sectionResizeMode(column) == QHeaderView.ResizeMode.Interactive
                for column in range(len(AlarmHistoryDialog.HEADERS))
            ),
            "所有列都该是可拖的",
        )
        dialog.table.setColumnWidth(3, 260)

        # 拖完就是拖完的宽度，不被内容宽度顶回去。
        self.assertEqual(dialog.table.columnWidth(3), 260)
        self.assertEqual(dialog.column_widths()["区域"], 260)

    def test_dragging_one_boundary_resizes_only_that_column(self) -> None:
        """拖一条边界只该动它左边那一列。

        这里用 QTest 真的按鼠标，而不是直接 setColumnWidth：问题就出在鼠标拖动这条
        路径上 —— 最后一列开着 stretchLastSection 时，拖中间任何一条边界它都会跟着
        补偿，看上去就是「拖一个动两个」。
        """
        dialog = self.dialog()
        dialog.resize(1000, 700)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()

        for column in (0, 2, 5, len(AlarmHistoryDialog.HEADERS) - 1):
            with self.subTest(column=AlarmHistoryDialog.HEADERS[column]):
                before = self.section_sizes(dialog)

                self.drag_boundary(dialog, column, 60)

                after = self.section_sizes(dialog)
                changed = [
                    index
                    for index, (old, new) in enumerate(zip(before, after))
                    if old != new
                ]
                self.assertEqual(
                    changed, [column], f"拖第 {column} 条边界时动了这些列：{changed}"
                )
                self.assertEqual(after[column], before[column] + 60)

    @staticmethod
    def section_sizes(dialog: AlarmHistoryDialog) -> list[int]:
        header = dialog.table.horizontalHeader()
        return [header.sectionSize(c) for c in range(dialog.table.columnCount())]

    @staticmethod
    def drag_boundary(dialog: AlarmHistoryDialog, column: int, delta: int) -> None:
        """在表头上真的拖一次第 column 条边界（+delta 像素）。"""
        header = dialog.table.horizontalHeader()
        start = sum(header.sectionSize(c) for c in range(column + 1)) + header.offset() - 1
        viewport = header.viewport()
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=QPoint(start, 8))
        QTest.mouseMove(viewport, QPoint(start + delta, 8))
        QApplication.processEvents()
        QTest.mouseRelease(
            viewport, Qt.MouseButton.LeftButton, pos=QPoint(start + delta, 8)
        )
        QApplication.processEvents()

    def test_every_column_has_a_width_even_without_saved_state(self) -> None:
        dialog = self.dialog()

        widths = dialog.column_widths()

        self.assertEqual(set(widths), set(AlarmHistoryDialog.HEADERS))
        self.assertTrue(all(width > 0 for width in widths.values()), widths)

    def test_saved_column_widths_are_restored_by_name(self) -> None:
        dialog = AlarmHistoryDialog(
            self.store,
            self.store.resolve_screenshot,
            None,
            column_widths={"区域": 260, "闯入时长": 180, "早就不存在的列": 300},
        )
        self.addCleanup(dialog.deleteLater)

        self.assertEqual(
            dialog.table.columnWidth(AlarmHistoryDialog.HEADERS.index("区域")), 260
        )
        self.assertEqual(dialog.column_widths()["闯入时长"], 180)
        self.assertEqual(set(dialog.column_widths()), set(AlarmHistoryDialog.HEADERS))

    def test_the_mode_and_status_filters_narrow_the_table(self) -> None:
        self.record(zone="报了警的", mode="monitor")
        self.record(zone="只是路过的", mode="video", alarm=False)

        dialog = self.dialog()
        options = [
            dialog.status_combo.itemText(index)
            for index in range(dialog.status_combo.count())
        ]
        self.assertIn("已结束", options)
        self.assertIn("未触发报警", options)

        dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData("视频模式"))
        self.assertEqual(self.column(dialog, "区域"), ["只是路过的"])

        dialog.mode_combo.setCurrentIndex(0)
        dialog.status_combo.setCurrentIndex(dialog.status_combo.findData("未触发报警"))

        self.assertEqual(self.column(dialog, "区域"), ["只是路过的"])

    def test_the_filter_offers_only_the_values_that_actually_occur(self) -> None:
        """下拉按记录里出现过的值生成：列出选了就空表的选项没有意义。"""
        self.record(zone="北侧入口", mode="monitor")

        dialog = self.dialog()

        modes = [
            dialog.mode_combo.itemText(index)
            for index in range(dialog.mode_combo.count())
        ]
        self.assertEqual(modes, ["全部模式", "监控模式"])

    def test_refresh_picks_up_records_written_while_the_viewer_is_open(self) -> None:
        self.record(zone="北侧入口")
        dialog = self.dialog()
        self.assertEqual(dialog.table.rowCount(), 1)

        self.record(zone="南侧通道")  # 检测线程在查看器开着的时候又报了一次警
        dialog.refresh_btn.click()

        self.assertEqual(self.column(dialog, "区域"), ["北侧入口", "南侧通道"])
        # 刷新不该把用户正在看的那条弄丢。
        self.assertIn("北侧入口", self.detail_text(dialog))

    # -- 右栏 ---------------------------------------------------------------

    def test_the_newest_record_is_selected_when_the_viewer_opens(self) -> None:
        self.record(zone="旧的")
        self.record(zone="新的")

        dialog = self.dialog()

        self.assertIn("警戒区域: 新的", self.detail_text(dialog))
        self.assertNotIn("旧的", self.detail_text(dialog))

    def test_selecting_a_row_switches_the_details(self) -> None:
        self.record(zone="旧的", track="12")
        self.record(zone="新的", track="77")

        dialog = self.dialog()
        dialog.table.selectRow(0)

        self.assertIn("警戒区域: 旧的", self.detail_text(dialog))
        self.assertIn("目标 ID: 12", self.detail_text(dialog))
        self.assertNotIn("新的", self.detail_text(dialog))

    def test_filtering_away_the_selected_record_clears_the_details(self) -> None:
        """右栏留着上一条的详情会让人以为它还在表里。"""
        self.record(zone="北侧入口")
        self.record(zone="南侧通道")

        dialog = self.dialog()
        dialog.table.selectRow(0)  # 选中的是「北侧入口」
        dialog.search_edit.setText("南侧")

        self.assertNotIn("北侧入口", self.detail_text(dialog))
        self.assertIn("在左侧选一条记录", self.detail_text(dialog))

    def test_the_two_screenshots_are_stacked_with_entry_on_top(self) -> None:
        event = self.record(zone="北侧入口")
        entry = self.store.resolve_screenshot(event.entry_screenshot_path)
        alarm = self.store.resolve_screenshot(event.alarm_screenshot_path)

        dialog = self.dialog()
        sequence = self.detail_sequence(dialog)

        entry_title = sequence.index("进入取证")
        alarm_title = sequence.index("报警取证")
        self.assertLess(entry_title, alarm_title, "进入取证要排在报警取证上面")
        # 每张小标题的下面就是那张图，两张图都在右栏里。
        self.assertEqual(sequence[entry_title + 1], str(entry))
        self.assertEqual(sequence[alarm_title + 1], str(alarm))
        previews = dialog.findChildren(PreviewImageLabel)
        self.assertEqual(len(previews), 2)
        self.assertTrue(all(p.parent() is dialog.details_content for p in previews))

    def test_double_click_opens_the_original_files(self) -> None:
        event = self.record(zone="北侧入口")
        entry = self.store.resolve_screenshot(event.entry_screenshot_path)
        alarm = self.store.resolve_screenshot(event.alarm_screenshot_path)

        dialog = self.dialog()
        with patch.object(
            main_window.QDesktopServices, "openUrl", return_value=True
        ) as open_url:
            for preview in dialog.findChildren(PreviewImageLabel):
                preview.mouseDoubleClickEvent(None)

        opened = {Path(call.args[0].toLocalFile()) for call in open_url.call_args_list}
        self.assertEqual(opened, {entry, alarm})

    def test_a_record_that_never_alarmed_says_why_there_is_no_alarm_shot(self) -> None:
        self.record(zone="只是路过的", alarm=False)

        dialog = self.dialog()

        self.assertIn("未触发报警（本就没有报警截图）", self.detail_text(dialog))
        self.assertEqual(
            len(dialog.findChildren(PreviewImageLabel)), 1, "进入那张还在"
        )

    def test_a_deleted_screenshot_is_reported_as_missing(self) -> None:
        event = self.record(zone="北侧入口")
        self.store.resolve_screenshot(event.alarm_screenshot_path).unlink()

        dialog = self.dialog()

        self.assertIn("截图文件缺失", self.detail_text(dialog))
        self.assertEqual(len(dialog.findChildren(PreviewImageLabel)), 1)

    # -- 窗口本身 -----------------------------------------------------------

    def test_the_viewer_opens_at_the_size_of_the_main_window(self) -> None:
        parent = QWidget()
        self.addCleanup(parent.deleteLater)
        parent.resize(1180, 760)

        dialog = self.dialog(parent)

        self.assertEqual(dialog.size(), parent.size())

    def test_the_viewer_is_split_into_two_columns(self) -> None:
        dialog = self.dialog()

        splitter = dialog.findChild(QSplitter)
        self.assertIsNotNone(splitter)
        self.assertEqual(splitter.count(), 2)
        self.assertTrue(splitter.widget(0).isAncestorOf(dialog.table), "左栏是记录表")
        self.assertTrue(
            splitter.widget(1).isAncestorOf(dialog.details_content), "右栏是详情"
        )


class EventPanelTests(unittest.TestCase):
    """面板上那张表与它的按钮：那个位置原来叫「清空记录」，现在换成查看器。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_the_button_opens_the_viewer(self) -> None:
        panel = EventPanel()

        self.assertEqual(panel.view_btn.text(), "查看记录")
        self.assertIn("全部记录", panel.view_btn.toolTip())

    def test_there_is_no_clear_button_left(self) -> None:
        """报警记录按留存策略是永久保留的（只清截图），界面上不再留「清空」这个入口。"""
        self.assertFalse(hasattr(EventPanel(), "clear_btn"))

    def test_an_updated_session_stays_on_its_own_row(self) -> None:
        panel = EventPanel()
        event = AlarmEvent(
            source="rtsp://camera/live",
            zone_name="北侧入口",
            track_id="12",
            entered_at_seconds=1.0,
            alarm_at_seconds=None,
            wall_time="2026-09-21 10:00:00",
            operation_mode="monitor",
        )

        panel.append_event(event)  # 进入区域
        event.status = "completed"
        event.exited_at_seconds = 6.0
        event.duration_seconds = 5.0
        panel.append_event(event)  # 离开区域

        self.assertEqual(panel.table.rowCount(), 1, "同一个会话不该占两行")
        status_column = EventPanel.HEADERS.index("状态")
        self.assertEqual(panel.table.item(0, status_column).text(), "未触发报警")

    def test_updating_a_row_does_not_make_qt_complain(self) -> None:
        """同一个单元格再 setItem 一次会被 Qt 当成「已经有主了」并打警告。

        这条路径每次会话状态变化都会走一遍（进入 / 报警 / 结束各一次），警告会刷满日志，
        而真正的问题在日志里反而被淹掉。
        """
        messages: list[str] = []
        previous = QtCore.qInstallMessageHandler(
            lambda mode, context, message: messages.append(message)
        )
        self.addCleanup(QtCore.qInstallMessageHandler, previous)
        panel = EventPanel()
        event = AlarmEvent(
            source="rtsp://camera/live",
            zone_name="北侧入口",
            track_id="12",
            entered_at_seconds=1.0,
            alarm_at_seconds=3.0,
            wall_time="2026-09-21 10:00:00",
            operation_mode="monitor",
        )

        panel.append_event(event)
        event.status = "alarmed"
        panel.append_event(event)

        self.assertEqual([text for text in messages if "QTableWidget" in text], [])


if __name__ == "__main__":
    unittest.main()
