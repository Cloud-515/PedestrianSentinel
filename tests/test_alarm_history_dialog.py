"""「查看记录」查看器：左栏全部记录加筛选搜索，右栏详情与两张取证图。

这里钉住的契约：

* 查看器看的是**全部**记录 —— 不是常驻面板那最近 200 条，也不只当前运行模式；
* 筛选与搜索真的收窄表格，选中那条的详情在右栏，两张取证图按「进入在上、报警在下」
  从上到下排开；
* 双击交出去的必须是**原图文件的真实路径**（不是预览、不是相对路径），图片不可用时
  则不给双击入口；
* 面板上那个位置原来是「清空记录」，现在是打开查看器的按钮；
* 查看器要能放大到整屏（最大化/最小化按钮 + F11 全屏），并且缩放它不能把程序拖崩：
  预览的高度必须是宽度的纯函数、右栏视口宽度必须恒定（见这两条测试的说明）。
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
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QLabel,
    QSplitter,
    QWidget,
)

import alarm_service
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
        # 记录时间也是被搜的列，所以这里固定一个不含「12」的时间：默认值取的是「现在」，
        # 跑到 12 点那几个小时里这条测试就会自己失败。
        fixed = "2026-09-21 08:00:00"
        self.record(zone="北侧入口", track="12", wall_time=fixed)
        self.record(zone="南侧通道", track="77", wall_time=fixed)

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
        """真实记录列的宽度，**不含**末尾那个空的占位列。

        占位列（`AlarmHistoryDialog.FILLER_COLUMN`）会跟着拖动补偿差额，但它没有表头
        文字也没有内容，用户看不见它动 —— 所以"拖一条边界只动一列"这条契约说的是记录列。
        """
        header = dialog.table.horizontalHeader()
        return [header.sectionSize(c) for c in range(len(AlarmHistoryDialog.HEADERS))]

    @staticmethod
    def filler_width(dialog: AlarmHistoryDialog) -> int:
        header = dialog.table.horizontalHeader()
        return header.sectionSize(AlarmHistoryDialog.FILLER_COLUMN)

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

    # -- 铺满：末尾的占位列 ------------------------------------------------

    def test_the_blank_area_right_of_the_last_column_is_filled(self) -> None:
        """放大窗口后，列宽之和会小于表宽，右边剩一块空白 —— 末尾的占位列把它填掉。"""
        dialog = self.dialog()
        dialog.resize(1920, 900)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()

        table = dialog.table
        real = sum(self.section_sizes(dialog))

        self.assertGreater(real, 0)
        self.assertLess(real, table.viewport().width(), "这条测的是「列不够长」的情形")
        self.assertEqual(
            self.filler_width(dialog),
            table.viewport().width() - real,
            "占位列的宽度应该正好是「表宽 − 各列之和」",
        )

    def test_the_row_stripes_reach_the_right_edge(self) -> None:
        """占位列里必须有 item，否则那片空白连行底色都没有。

        这正是它「看起来像表没画完」的原因：Qt 只给有 item 的格子画行底色，实测同一行里
        真实列内是 #f7f7f7 而空白区是 #ffffff，隔行条纹在那里断掉。这里比像素：隔行两行
        在占位列里的颜色必须**不同** —— 相同就说明那里根本没上底色。

        不去和真实列内的像素比：测试环境没有中文字体，文字会渲染成黑色方块，采到字上
        就是黑的。
        """
        for index in range(3):
            self.record(zone=f"区域{index}", track=str(index))
        dialog = self.dialog()
        dialog.resize(1600, 700)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()

        table = dialog.table
        table.scrollToTop()
        self.app.processEvents()
        header = table.horizontalHeader()
        filler = AlarmHistoryDialog.FILLER_COLUMN
        filler_left = header.sectionViewportPosition(filler)
        self.assertGreater(header.sectionSize(filler), 40, "窗口够宽时才谈得上空白")
        self.assertIsNotNone(table.item(1, filler), "占位列的格子里没有 item")

        image = table.viewport().grab().toImage()
        x = filler_left + 20
        even = image.pixelColor(x, table.rowViewportPosition(0) + table.rowHeight(0) // 2)
        odd = image.pixelColor(x, table.rowViewportPosition(1) + table.rowHeight(1) // 2)

        self.assertNotEqual(
            even.name(), odd.name(), "占位列里没有隔行底色 —— 那片空白又回来了"
        )

    def test_the_filler_column_takes_the_slack_and_gives_it_back(self) -> None:
        """占位列的宽度就是"表宽 − 各列之和"：拖动时它吃掉差额（用户看不见它动）。"""
        dialog = self.dialog()
        dialog.resize(1600, 700)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()

        before_filler = self.filler_width(dialog)
        self.assertGreater(before_filler, 60, "这条要有足够的空白才测得出来")
        before = self.section_sizes(dialog)

        self.drag_boundary(dialog, 2, 60)  # 把「视频源」拖宽 60

        after = self.section_sizes(dialog)
        changed = [
            index for index, (old, new) in enumerate(zip(before, after)) if old != new
        ]
        self.assertEqual(changed, [2], f"拖第 2 条边界时动了这些记录列：{changed}")
        self.assertEqual(self.filler_width(dialog), before_filler - 60)

    def test_the_filler_column_is_not_a_record_column(self) -> None:
        """占位列不进 HEADERS、不进 config：存盘的那份仍然是十个记录列。"""
        dialog = self.dialog()

        self.assertEqual(
            dialog.table.columnCount(), len(AlarmHistoryDialog.HEADERS) + 1
        )
        self.assertEqual(
            dialog.table.horizontalHeaderItem(AlarmHistoryDialog.FILLER_COLUMN).text(),
            "",
        )
        self.assertEqual(set(dialog.column_widths()), set(AlarmHistoryDialog.HEADERS))

    def test_filling_widens_a_truncated_column_before_the_filler(self) -> None:
        """空白优先给"内容被截断"的列：视频源要能显示全，剩下的才给占位列。"""
        self.record(zone="北侧入口")
        dialog = AlarmHistoryDialog(
            self.store,
            self.store.resolve_screenshot,
            None,
            # 故意给一个很窄的视频源列（现场拖窄过就是这种状态）
            column_widths={"视频源": 60},
        )
        self.addCleanup(dialog.deleteLater)
        dialog.resize(1920, 900)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()

        column = AlarmHistoryDialog.HEADERS.index("视频源")
        widened = dialog.table.columnWidth(column)
        cap = dict(AlarmHistoryDialog.WIDENABLE_COLUMNS)["视频源"]

        self.assertGreater(widened, 60, "被截断的列应该被补宽")
        self.assertLessEqual(widened, cap, "补宽有上限")
        self.assertGreater(
            dialog._content_widths["视频源"], 60, "这条记录的视频源确实比 60 px 宽"
        )

    def test_the_widened_width_is_not_saved_as_a_user_width(self) -> None:
        """补宽出来的那部分不能记成"用户拖成这样"。

        否则在放大的窗口里关一次窗，下次在小窗口打开时列宽之和就超过表宽，全是横向滚动条。
        """
        self.record(zone="北侧入口")
        saved = {"视频源": 60}
        dialog = AlarmHistoryDialog(
            self.store, self.store.resolve_screenshot, None, column_widths=saved
        )
        self.addCleanup(dialog.deleteLater)
        dialog.resize(1920, 900)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()

        self.assertGreater(dialog.table.columnWidth(2), 60, "显示宽度被补宽了")
        self.assertEqual(dialog.column_widths()["视频源"], 60, "存盘的仍是用户宽度")

    def test_a_column_the_user_narrowed_is_not_widened_back(self) -> None:
        """用户拖窄过的列不再自动补宽 —— 拖窄了又自己变宽，等于跟用户抢。"""
        self.record(zone="北侧入口")
        dialog = self.dialog()
        dialog.resize(1920, 900)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()

        column = AlarmHistoryDialog.HEADERS.index("视频源")
        self.drag_boundary(dialog, column, -30)  # 用户把它拖窄
        narrowed = dialog.table.columnWidth(column)

        # 再改变一次窗口尺寸（会重跑一遍铺满）
        dialog.resize(1900, 900)
        self.app.processEvents()

        self.assertEqual(
            dialog.table.columnWidth(column), narrowed, "用户拖窄过的列又被自动放宽了"
        )
        self.assertEqual(dialog.column_widths()["视频源"], narrowed)

    def test_a_narrow_window_keeps_scrolling_instead_of_filling(self) -> None:
        """表装不下时占位列收到 0 宽，横向滚动范围仍然正好等于溢出的像素。"""
        dialog = self.dialog()
        dialog.resize(900, 700)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()

        table = dialog.table
        for column in range(len(AlarmHistoryDialog.HEADERS)):
            table.setColumnWidth(column, 150)
        self.app.processEvents()

        header = table.horizontalHeader()
        self.assertEqual(self.filler_width(dialog), 0, "装不下时占位列不该占宽度")
        self.assertEqual(
            table.horizontalScrollBar().maximum(),
            header.length() - table.viewport().width(),
        )

    def test_the_filler_never_pushes_the_columns_past_the_table_width(self) -> None:
        """扫一遍宽度：列宽之和没超过表宽时，占位列必须正好等于剩下的空白。

        表头的最小列宽默认是十几像素（按字体算）。不把它设成 0，占位列就会被顶到十几
        像素 —— 空白只剩几像素时列宽之和反而超过表宽，冒出一条本不该有的横向滚动条。
        """
        dialog = self.dialog()
        dialog.resize(900, 700)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()

        table = dialog.table
        for width in range(900, 1600, 7):
            dialog.resize(width, 700)
            self.app.processEvents()
            slack = table.viewport().width() - sum(self.section_sizes(dialog))
            with self.subTest(width=width, slack=slack):
                if slack < 0:
                    continue
                self.assertEqual(self.filler_width(dialog), slack)
                self.assertEqual(
                    table.horizontalScrollBar().maximum(), 0, "不该出现横向滚动条"
                )

    def test_a_column_cannot_be_dragged_to_nothing(self) -> None:
        """列被拖成 0 宽就看不见也抓不回来了。

        表头最小列宽被设成 0（为了让占位列能收窄到几像素），所以这条下限由代码把守。
        """
        dialog = self.dialog()
        dialog.resize(1200, 700)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()

        self.drag_boundary(dialog, 2, -500)

        self.assertGreaterEqual(
            dialog.table.columnWidth(2),
            AlarmHistoryDialog.MIN_COLUMN_WIDTH,
            "列被拖没了",
        )
        self.assertGreaterEqual(
            dialog.column_widths()["视频源"], AlarmHistoryDialog.MIN_COLUMN_WIDTH
        )

    # -- 三个时刻的显示 -----------------------------------------------------

    def test_video_mode_times_are_clocks_with_the_video_position_beside_them(self) -> None:
        """视频模式：钟点用来对日志，视频位置用来在播放器里找到那一刻 —— 两个都要。"""
        with patch.object(
            alarm_service,
            "_clock_now",
            side_effect=[
                "2026-09-21 14:51:48",  # 进入（事件建立那一刻）
                "2026-09-21 14:51:51",  # 报警
                "2026-09-21 14:51:53",  # 退出
            ],
        ):
            self.record(zone="北侧入口", mode="video", wall_time="2026-09-21 14:51:48")

        dialog = self.dialog()
        text = self.detail_text(dialog)

        self.assertIn("进入时间: 14:51:48（视频位置 1.00s）", text)
        self.assertIn("报警时间: 14:51:51（视频位置 3.00s）", text)
        self.assertIn("退出时间: 14:51:53（视频位置 6.00s）", text)
        # 表格里三列也是真实钟点，不再出现「15.867s」那种。
        self.assertEqual(self.column(dialog, "进入时刻"), ["14:51:48"])
        self.assertEqual(self.column(dialog, "报警时刻"), ["14:51:51"])
        self.assertEqual(self.column(dialog, "退出时刻"), ["14:51:53"])

    def test_monitor_mode_times_are_clocks_without_milliseconds(self) -> None:
        """监控模式的三个时刻本来就是钟点，只是不再显示日期与毫秒。"""
        entered = datetime(2026, 9, 21, 14, 51, 48).timestamp()
        self.store._append(
            {
                "schema_version": 2,
                "action": "opened",
                "event": EventStore._event_payload(
                    AlarmEvent(
                        source="rtsp://camera/live",
                        zone_name="北侧入口",
                        track_id="12",
                        entered_at_seconds=entered,
                        alarm_at_seconds=entered + 3,
                        wall_time="2026-09-21 14:51:48",
                        operation_mode="monitor",
                        exited_at_seconds=entered + 6,
                        duration_seconds=6.0,
                    )
                ),
            }
        )

        dialog = self.dialog()

        self.assertEqual(self.column(dialog, "进入时刻"), ["14:51:48"])
        self.assertEqual(self.column(dialog, "报警时刻"), ["14:51:51"])
        self.assertEqual(self.column(dialog, "退出时刻"), ["14:51:54"])
        text = self.detail_text(dialog)
        self.assertIn("报警时间: 14:51:51", text)
        self.assertNotIn("14:51:51.", text, "详情里不该再出现毫秒")

    def test_an_old_record_says_the_clock_was_not_recorded(self) -> None:
        """老记录没有报警钟点（文件名里的时间戳也不是它）：如实说没记，别编一个。"""
        self.store._append(
            {
                "schema_version": 2,
                "action": "opened",
                "event": EventStore._event_payload(
                    AlarmEvent(
                        source="test.mp4",
                        zone_name="北侧入口",
                        track_id="12",
                        entered_at_seconds=1.0,
                        alarm_at_seconds=3.0,
                        wall_time="2026-08-12 11:20:07",
                        operation_mode="video",
                        exited_at_seconds=6.0,
                        duration_seconds=5.0,
                        # 老命名：里面的时间戳是会话开始，不是报警那一刻。
                        alarm_screenshot_path="screenshots/2026-08-12_11-20-07_区域_1_id0.jpg",
                    )
                ),
            }
        )

        dialog = self.dialog()
        text = self.detail_text(dialog)

        self.assertIn("报警时间: 未记录真实钟点（这条记录只存了视频位置 3.00s）", text)
        self.assertEqual(self.column(dialog, "报警时刻"), ["视频 3.00s"])

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

    def test_the_header_stays_aligned_with_the_cells_when_scrolled(self) -> None:
        """滚动、拖分隔条之后，每一列的表头 x 与表体 x 都要逐列相等。

        这就是「表头与表体对齐」本身。它们本该永远相等（表头是表格控件的一部分，不是
        另一块要手动同步的表），所以这条同时是给以后改列宽策略的人留的护栏。
        """
        dialog = self.dialog()
        dialog.resize(1000, 700)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()

        table = dialog.table
        for column in range(len(AlarmHistoryDialog.HEADERS)):
            table.setColumnWidth(column, 150)
        bar = table.horizontalScrollBar()

        for value in (0, 137, bar.maximum() // 2, bar.maximum()):
            with self.subTest(scroll=value):
                bar.setValue(value)
                self.app.processEvents()
                self.assert_header_matches_body(dialog)

        with self.subTest(action="拖分隔条改表宽"):
            dialog.findChild(QSplitter).setSizes([500, 700])
            self.app.processEvents()
            self.assert_header_matches_body(dialog)

    def assert_header_matches_body(self, dialog: AlarmHistoryDialog) -> None:
        table = dialog.table
        header = table.horizontalHeader()
        for column in range(table.columnCount()):
            self.assertEqual(
                table.columnViewportPosition(column),
                header.sectionViewportPosition(column),
                f"第 {column} 列表头与表体错位",
            )

    def test_the_horizontal_scrollbar_range_matches_the_overflow_in_pixels(self) -> None:
        """横向滚动条的范围要按像素算。

        Qt 默认按「列」算：十列里看得见四列，范围就是 0..6，而内容实际超出七百多像素
        —— 拖起来一格跳一列，拇指的大小和位置也和看到的画面对不上。
        """
        dialog = self.dialog()
        dialog.resize(1000, 700)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()

        table = dialog.table
        for column in range(len(AlarmHistoryDialog.HEADERS)):
            table.setColumnWidth(column, 150)
        self.app.processEvents()

        overflow = table.horizontalHeader().length() - table.viewport().width()
        self.assertGreater(overflow, 0, "这条测的是横向溢出时的情况")
        self.assertEqual(table.horizontalScrollBar().maximum(), overflow)

    def test_the_header_labels_line_up_with_the_cells(self) -> None:
        """表头文字靠左：Qt 默认居中，列一宽列名就跑到列中间，看着像表头错位。"""
        dialog = self.dialog()

        alignment = dialog.table.horizontalHeader().defaultAlignment()

        self.assertTrue(alignment & Qt.AlignmentFlag.AlignLeft, alignment)

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

    # -- 放大到整屏 ---------------------------------------------------------

    def test_the_viewer_window_can_be_maximized(self) -> None:
        """记录一多就得放大看：默认的 QDialog 只有关闭按钮，只能拖边框一格一格放大。"""
        dialog = self.dialog()

        flags = dialog.windowFlags()
        self.assertTrue(
            flags & Qt.WindowType.WindowMaximizeButtonHint, "要有最大化按钮"
        )
        self.assertTrue(
            flags & Qt.WindowType.WindowMinimizeButtonHint, "要有最小化按钮"
        )

    def test_the_fullscreen_button_toggles_the_window(self) -> None:
        dialog = self.dialog()
        dialog.resize(900, 700)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()

        dialog.fullscreen_btn.click()
        self.app.processEvents()
        self.assertTrue(dialog.isFullScreen())
        self.assertEqual(dialog.fullscreen_btn.text(), "还原", "按钮要变成「还原」")

        dialog.fullscreen_btn.click()
        self.app.processEvents()
        self.assertFalse(dialog.isFullScreen())
        self.assertEqual(dialog.fullscreen_btn.text(), "全屏")

    def test_f11_toggles_fullscreen_and_escape_leaves_it_first(self) -> None:
        """全屏下按 Esc 先退出全屏：一按就关掉整个查看器，会让人以为刚筛出的记录丢了。"""
        dialog = self.dialog()
        dialog.resize(900, 700)
        dialog.show()
        self.addCleanup(dialog.close)
        self.app.processEvents()

        QTest.keyClick(dialog, Qt.Key.Key_F11)
        self.app.processEvents()
        self.assertTrue(dialog.isFullScreen())

        QTest.keyClick(dialog, Qt.Key.Key_Escape)
        self.app.processEvents()
        self.assertFalse(dialog.isFullScreen())
        self.assertTrue(dialog.isVisible(), "第一次 Esc 不该关窗")

        # 退出全屏之后，Esc 恢复它本来的意思：关窗。
        QTest.keyClick(dialog, Qt.Key.Key_Escape)
        self.app.processEvents()
        self.assertFalse(dialog.isVisible())

    def test_fullscreen_restores_the_maximized_state_it_started_from(self) -> None:
        dialog = self.dialog()
        dialog.showMaximized()
        self.addCleanup(dialog.close)
        self.app.processEvents()

        dialog.toggle_fullscreen()
        self.app.processEvents()
        self.assertTrue(dialog.isFullScreen())

        dialog.toggle_fullscreen()
        self.app.processEvents()
        self.assertFalse(dialog.isFullScreen())
        self.assertTrue(dialog.isMaximized(), "原来是最大化就回最大化，不是回普通尺寸")

    def test_the_details_pane_reserves_room_for_the_scrollbar(self) -> None:
        """右栏竖滚动条常驻。

        默认的「需要时才出现」会在出现时把视口宽度削掉十几像素，内容跟着重新折行、高度
        又变；而 QScrollArea::updateScrollBars 是同步回调（内容控件一收到 Resize 就再算
        一次），尺寸在两个状态之间来回摆时会一路递归到栈溢出 —— 实机崩溃转储里同一条
        调用链重复了 453 层。视口宽度恒定，这条回路就不存在了。
        """
        dialog = self.dialog()

        self.assertEqual(
            dialog.details_scroll.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn,
        )

    def test_the_preview_height_is_a_pure_function_of_the_width(self) -> None:
        """预览高度只由原图宽高比决定，与「当前这张 pixmap 多大」无关。

        原来是「当前 pixmap 多大就多高」，于是布局算出的内容高度依赖上一轮缩放的结果，
        滚动条就会反复开关（同上一条测试的栈溢出回路）。这里把同一个宽度问两遍，中间
        夹一次真正的缩放，答案必须一样。
        """
        preview = PreviewImageLabel("shot.jpg", QPixmap(160, 90))
        self.addCleanup(preview.deleteLater)

        self.assertTrue(preview.sizePolicy().hasHeightForWidth())
        self.assertEqual(preview.heightForWidth(320), 240, "低于下限时按下限")
        self.assertEqual(preview.heightForWidth(500), 281, "16:9 按比例")
        self.assertEqual(preview.heightForWidth(1000), 300, "高于上限时按上限")

        preview.resize(500, 900)  # 触发一次真实的重新缩放
        self.app.processEvents()

        self.assertEqual(preview.heightForWidth(500), 281, "缩放之后同一个宽度还是同一个高度")

    def test_the_preview_height_does_not_depend_on_where_it_was_resized_from(self) -> None:
        """两条不同的缩放路径走到同一个宽度，高度必须一致 —— 否则就是滞回，会震荡。"""
        preview = PreviewImageLabel("shot.jpg", QPixmap(160, 90))
        self.addCleanup(preview.deleteLater)
        other = PreviewImageLabel("shot.jpg", QPixmap(160, 90))
        self.addCleanup(other.deleteLater)

        preview.resize(900, 400)
        preview.resize(600, 400)
        other.resize(300, 400)
        other.resize(600, 400)
        self.app.processEvents()

        self.assertEqual(preview.heightForWidth(600), other.heightForWidth(600))


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

    def test_the_panel_table_lines_up_and_scrolls_the_same_way(self) -> None:
        """两张表共用一份表设置：列名的对齐方式与横向滚动方式不该只在一边生效。"""
        panel = EventPanel()

        self.assertEqual(
            panel.table.horizontalScrollMode(),
            QAbstractItemView.ScrollMode.ScrollPerPixel,
        )
        self.assertTrue(
            panel.table.horizontalHeader().defaultAlignment()
            & Qt.AlignmentFlag.AlignLeft
        )

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
