"""报警记录的读取：尾部倒读必须和全量扫描给出同一个答案。

`load_recent` 原先每次都要扫完整个 JSONL。记录是永久保留的，一台有人流量的点位
一年能写几十 MB，而界面一次只看得着最后 200 条 —— 于是改成从文件尾部按块倒读。
倒读本身没什么难的，难的是它**必须和全量读等价**：会话是被 opened/alarmed/closed
三行折叠出来的，只看尾巴就很容易把一条会话拆成两半、或者把跨块的中文切碎。

所以这里的核心是那个对比测试：同一份文件，全量读和倒读的结果必须逐字段相同。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import alarm_service
from alarm_service import EventStore
from models import AlarmEvent


class TailReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = EventStore(Path(self.temporary.name) / "events")
        self.store.root.mkdir(parents=True)

    def write(self, lines: list[str]) -> None:
        self.store.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def session_lines(self, index: int, operation_mode: str = "monitor") -> list[str]:
        """一条完整的三行会话记录，区域名带中文以覆盖多字节跨块。"""
        event = {
            "session_id": f"s{index}",
            "time": "2026-09-21 10:00:00",
            "video_source": "rtsp://摄像头/主通道",
            "operation_mode": operation_mode,
            "zone_name": f"北侧通道 {index}",
            "track_id": str(index),
            "entered_at_seconds": float(index),
            "alarm_at_seconds": float(index) + 1.5,
            "exited_at_seconds": float(index) + 4.0,
            "duration_seconds": 4.0,
            "entry_screenshot_path": "",
            "alarm_screenshot_path": "",
            "screenshot_path": "",
            "status": "completed",
        }
        return [
            json.dumps(
                {"schema_version": 2, "action": "opened", "event": event},
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "schema_version": 2,
                    "action": "alarmed",
                    "session_id": f"s{index}",
                    "alarm_at_seconds": float(index) + 1.5,
                    "alarm_screenshot_path": "",
                    "screenshot_path": "",
                    "status": "alarmed",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "schema_version": 2,
                    "action": "closed",
                    "session_id": f"s{index}",
                    "exited_at_seconds": float(index) + 4.0,
                    "duration_seconds": 4.0,
                    "status": "completed",
                },
                ensure_ascii=False,
            ),
        ]

    def populate(self, count: int) -> None:
        lines: list[str] = []
        for index in range(count):
            lines.extend(self.session_lines(index))
        self.write(lines)

    def full_scan(self) -> list[AlarmEvent]:
        events, read_everything = self.store._scan_locked(None)
        self.assertTrue(read_everything)
        return events

    def snapshot(self, events: list[AlarmEvent]) -> list[tuple]:
        """取一份能代表全部关键字段的快照，用来逐个比对两条读取路径。"""
        return [
            (
                event.session_id,
                event.zone_name,
                event.track_id,
                event.operation_mode,
                event.entered_at_seconds,
                event.alarm_at_seconds,
                event.exited_at_seconds,
                event.duration_seconds,
                event.status,
                event.wall_time,
                event.source,
            )
            for event in events
        ]

    def test_tail_read_equals_full_scan_for_every_budget(self) -> None:
        self.populate(12)

        expected = self.snapshot(self.full_scan())

        for budget in (0, 1, 2, 5, 11, 12, 13, 100):
            with self.subTest(budget=budget):
                events, _ = self.store._scan_locked(budget)
                # 只多不少：预算不足时会多读，不该少给。
                self.assertGreaterEqual(len(events), min(budget, 12))
                self.assertEqual(
                    self.snapshot(events[-budget:]) if budget else [],
                    expected[-budget:] if budget else [],
                )

    def test_tail_read_equals_full_scan_at_tiny_chunk_sizes(self) -> None:
        """块小到一行都装不下时也必须等价 —— 跨块的多字节字符正是在这里现形。

        倒读返回的是「至少 budget 条」，所以比对的是它末尾那段与全量扫描的末尾那段；
        同时要求它整体是全量结果的一个后缀（不能凭空多出会话，也不能错位）。
        """
        self.populate(8)
        expected = self.snapshot(self.full_scan())

        for chunk in (1, 2, 3, 7, 16, 64, 200):
            with self.subTest(chunk_size=chunk):
                self.store.READ_CHUNK_BYTES = chunk
                events, _ = self.store._scan_locked(3)
                snapshot = self.snapshot(events)

                self.assertEqual(snapshot[-3:], expected[-3:])
                self.assertEqual(snapshot, expected[-len(snapshot) :])

    def test_tail_read_never_splits_a_session(self) -> None:
        self.populate(20)
        # 块小到装不下整个文件，才谈得上「倒读」；否则一次就读完了。
        self.store.READ_CHUNK_BYTES = 512

        events, read_everything = self.store._scan_locked(1)

        self.assertFalse(read_everything, "只要 1 条会话，不该把整份文件读完")
        # 最后一条会话必须带着它的报警时刻与结算时长（那些写在后面的行里）。
        self.assertEqual(events[-1].session_id, "s19")
        self.assertEqual(events[-1].alarm_at_seconds, 20.5)
        self.assertEqual(events[-1].duration_seconds, 4.0)
        self.assertEqual(events[-1].status, "completed")
        # 窗口边界上不该留下半条会话（只有 opened 没有后续行，或只有 closed）。
        self.assertTrue(all(event.session_id.startswith("s") for event in events))
        self.assertEqual(
            [event.session_id for event in events],
            [f"s{index}" for index in range(20 - len(events), 20)],
        )

    def test_tail_read_reports_when_it_reached_the_start(self) -> None:
        self.populate(3)

        _, read_everything = self.store._scan_locked(50)

        self.assertTrue(read_everything)

    def test_load_recent_uses_the_tail_when_no_mode_filter_is_given(self) -> None:
        self.populate(30)
        self.store.READ_CHUNK_BYTES = 256

        events = self.store.load_recent(limit=5)

        self.assertEqual([event.session_id for event in events], [f"s{i}" for i in range(25, 30)])

    def test_load_recent_with_a_mode_filter_still_returns_a_full_page(self) -> None:
        """按模式过滤时只读尾巴会少给记录，所以要往回多读几屏。"""
        lines: list[str] = []
        for index in range(40):
            mode = "video" if index < 30 else "monitor"
            lines.extend(self.session_lines(index, operation_mode=mode))
        self.write(lines)
        self.store.READ_CHUNK_BYTES = 128

        monitor = self.store.load_recent(limit=5, operation_mode="monitor")
        video = self.store.load_recent(limit=5, operation_mode="video")

        self.assertEqual([event.session_id for event in monitor], [f"s{i}" for i in range(35, 40)])
        # video 的 30 条都在更靠前的位置：必须往回读到它们，一条都不能少。
        self.assertEqual(len(video), 5)
        self.assertEqual([event.session_id for event in video], [f"s{i}" for i in range(25, 30)])

    def test_mode_filter_gives_everything_when_there_are_fewer_than_the_limit(self) -> None:
        lines: list[str] = []
        for index in range(10):
            lines.extend(self.session_lines(index, operation_mode="video"))
        self.write(lines)

        events = self.store.load_recent(limit=50, operation_mode="video")

        self.assertEqual(len(events), 10)

    def test_missing_file_and_empty_file(self) -> None:
        self.assertEqual(self.store._scan_locked(5), ([], True))
        self.write([])

        self.assertEqual(self.store.load_recent(limit=5), [])

    def test_broken_lines_survive_the_tail_read(self) -> None:
        lines = ["{不是 json", json.dumps([1, 2, 3])]
        lines.extend(self.session_lines(0))
        lines.extend(["", "{又一条坏的"])
        self.write(lines)
        self.store.READ_CHUNK_BYTES = 32

        events = self.store.load_recent(limit=5)

        self.assertEqual([event.session_id for event in events], ["s0"])

    def test_a_session_without_its_opening_line_is_not_fabricated(self) -> None:
        """尾巴里可能有半条会话（opened 在窗口之外），它不该被当成一条新记录。"""
        lines = [json.dumps({"operation_mode": "monitor", "session_id": "老会话"})]
        lines.extend(self.session_lines(9))
        lines.append(
            json.dumps(
                {
                    "schema_version": 2,
                    "action": "closed",
                    "session_id": "更老的",
                    "exited_at_seconds": 1.0,
                    "duration_seconds": 1.0,
                    "status": "completed",
                },
                ensure_ascii=False,
            )
        )
        self.write(lines)
        self.store.READ_CHUNK_BYTES = 64

        events, _ = self.store._scan_locked(1)
        ids = [event.session_id for event in events]

        # 只有 closed 行、没有 opened 行的「更老的」不能凭空变成一条记录。
        self.assertNotIn("更老的", ids)
        self.assertIn("s9", ids)
        # 窗口边界上的 s9 必须是完整的：报警时刻与结算时长都写在它后面的行里。
        complete = next(event for event in events if event.session_id == "s9")
        self.assertEqual(complete.status, "completed")
        self.assertEqual(complete.duration_seconds, 4.0)
        self.assertEqual(complete.alarm_at_seconds, 10.5)

    def test_sessions_are_counted_by_structure_not_by_substring(self) -> None:
        """区域名里出现 opened 字样不该被当成会话起点。"""
        lines = [
            json.dumps(
                {
                    "schema_version": 2,
                    "action": "opened",
                    "event": {
                        "session_id": "s1",
                        "zone_name": '写着 "action": "opened" 的区域名',
                        "operation_mode": "monitor",
                        "entered_at_seconds": 1.0,
                    },
                },
                ensure_ascii=False,
            )
        ]
        self.write(lines)

        self.assertTrue(self.store._starts_session(lines[0]))
        # alarmed / closed 行不是起点；legacy 单行记录才是。
        self.assertFalse(
            self.store._starts_session(
                json.dumps(
                    {
                        "schema_version": 2,
                        "action": "alarmed",
                        "session_id": "s1",
                        "alarm_at_seconds": 2.0,
                    }
                )
            )
        )
        self.assertTrue(
            self.store._starts_session(
                json.dumps({"session_id": "旧", "entered_at_seconds": 1.0})
            )
        )


class RealClockTests(unittest.TestCase):
    """三个时刻的**真实钟点**：写记录时记下，读记录时还原。

    视频模式下 ``*_at_seconds`` 是视频里的位置（2.00s 这种），不是钟点；真实钟点靠
    ``alarmed_wall_time`` / ``exited_wall_time`` 两个字段。这里钉住：新记录要写、要能
    读回来；加字段之前写下的记录，报警钟点还能从截图文件名里捞回来。
    """

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = EventStore(Path(self.temporary.name) / "events")

    def event(self, **overrides: object) -> AlarmEvent:
        values: dict[str, object] = {
            "source": "test.mp4",
            "zone_name": "北侧入口",
            "track_id": "0",
            "entered_at_seconds": 1.0,
            "alarm_at_seconds": 3.0,
            "wall_time": "2026-09-21 14:51:48",
            "operation_mode": "video",
            "exited_at_seconds": 6.0,
            "duration_seconds": 5.0,
        }
        values.update(overrides)
        return AlarmEvent(**values)  # type: ignore[arg-type]

    def test_a_new_session_records_the_clock_of_each_moment(self) -> None:
        frame = np.full((40, 60, 3), 120, dtype=np.uint8)
        event = self.event()

        with patch.object(
            alarm_service,
            "_clock_now",
            side_effect=[
                "2026-09-21 14:51:48",  # 进入
                "2026-09-21 14:51:51",  # 报警
                "2026-09-21 14:51:54",  # 退出
            ],
        ):
            self.store.open_session(event, frame)
            self.store.mark_alarmed(event, frame)
            self.store.close_session(event)

        loaded = self.store.load_all()[-1]
        self.assertEqual(
            loaded.to_row()[4:7], ["14:51:48", "14:51:51", "14:51:54"]
        )
        for moment in ("entered", "alarmed", "exited"):
            self.assertIsNotNone(loaded.moment_clock(moment), moment)
        # 视频位置没被顶掉：详情里还要用它。
        self.assertEqual(loaded.moment_offset("alarmed"), 3.0)

    def test_the_clocks_survive_a_reload(self) -> None:
        self.store._append(
            {
                "schema_version": 2,
                "action": "opened",
                "event": EventStore._event_payload(
                    self.event(
                        entered_wall_time="2026-09-21 14:51:48",
                        alarmed_wall_time="2026-09-21 14:51:51",
                        exited_wall_time="2026-09-21 14:51:54",
                    )
                ),
            }
        )

        loaded = self.store.load_all()[-1]

        self.assertEqual(loaded.entered_wall_time, "2026-09-21 14:51:48")
        self.assertEqual(loaded.alarmed_wall_time, "2026-09-21 14:51:51")
        self.assertEqual(loaded.exited_wall_time, "2026-09-21 14:51:54")
        self.assertEqual(loaded.to_row()[4:7], ["14:51:48", "14:51:51", "14:51:54"])

    def test_a_v2_record_written_before_the_field_existed_uses_wall_time_as_the_entry(self) -> None:
        """加字段之前的 v2 记录（没有 entered_wall_time 键）：它的 wall_time 就是进入钟点。"""
        payload = EventStore._event_payload(self.event())
        del payload["entered_wall_time"]
        self.store._append({"schema_version": 2, "action": "opened", "event": payload})

        loaded = self.store.load_all()[-1]

        self.assertEqual(loaded.entered_wall_time, "2026-09-21 14:51:48")
        self.assertEqual(loaded.format_moment("entered"), "14:51:48")

    def test_the_alarm_clock_is_recovered_from_the_screenshot_name(self) -> None:
        """加字段之前写下的记录，报警钟点还躺在截图文件名里（文件名取的就是落盘时刻）。"""
        self.store._append(
            {
                "schema_version": 2,
                "action": "opened",
                "event": EventStore._event_payload(
                    self.event(
                        alarm_screenshot_path="screenshots/20260921-143610_abc123_alarm.jpg"
                    )
                ),
            }
        )

        loaded = self.store.load_all()[-1]

        self.assertEqual(loaded.alarmed_wall_time, "2026-09-21 14:36:10")
        self.assertEqual(loaded.format_moment("alarmed"), "14:36:10")

    def test_the_old_naming_is_not_used_to_recover_the_alarm_clock(self) -> None:
        """更早的命名里那个时间戳是**会话开始**，不是报警那一刻（现有记录实测差 0 秒）。

        认它会把「进入时间」说成「报警时间」，比显示视频位置更糟。
        """
        self.store._append(
            {
                "schema_version": 2,
                "action": "opened",
                "event": EventStore._event_payload(
                    self.event(
                        alarm_screenshot_path="screenshots/2026-08-12_11-20-07_区域_1_id0.jpg"
                    )
                ),
            }
        )

        loaded = self.store.load_all()[-1]

        self.assertEqual(loaded.alarmed_wall_time, "")
        self.assertEqual(loaded.format_moment("alarmed"), "视频 3.00s")

    def test_the_exit_clock_is_never_invented(self) -> None:
        """退出那一刻没有截图、也没有字段：就说没记，不拿视频位置冒充时间。"""
        self.store._append(
            {
                "schema_version": 2,
                "action": "opened",
                "event": EventStore._event_payload(self.event()),
            }
        )

        loaded = self.store.load_all()[-1]

        self.assertIsNone(loaded.moment_clock("exited"))
        self.assertEqual(loaded.format_moment("exited"), "视频 6.00s")

    def test_rewriting_a_legacy_record_does_not_pass_its_alarm_clock_as_the_entry(self) -> None:
        """清空记录会把老记录重写成 v2 形态，重写后不能把 wall_time（报警钟点）当成进入钟点。

        这是最容易踩的一处：v2 的 opened 行按定义就是「进入时写的」，所以读的时候会拿
        wall_time 补进入钟点；而老记录被重写之后也是 v2 形态，可它的 wall_time 是报警钟点。
        """
        self.store.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.store.log_path.write_text(
            json.dumps(
                {
                    "time": "2026-09-21 14:51:48",
                    "video_source": "test.mp4",
                    "operation_mode": "video",
                    "zone_name": "北侧入口",
                    "track_id": "0",
                    "entered_at_seconds": 1.0,
                    "alarm_at_seconds": 3.0,
                    "exited_at_seconds": 6.0,
                    "duration_seconds": 5.0,
                    "screenshot_path": "",
                    "status": "completed",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        legacy = self.store.load_all()[-1]
        self.assertEqual(legacy.format_moment("entered"), "视频 1.00s")
        self.assertEqual(legacy.format_moment("alarmed"), "14:51:48")

        self.store.clear("monitor")  # 读-改-写：留下视频记录，重写成 v2 形态
        rewritten = self.store.load_all()[-1]

        self.assertEqual(
            rewritten.format_moment("entered"), "视频 1.00s", "进入钟点被顶成了报警钟点"
        )
        self.assertEqual(rewritten.format_moment("alarmed"), "14:51:48")


if __name__ == "__main__":
    unittest.main()
