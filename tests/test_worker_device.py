from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from alarm_service import AlarmPlayer, EventStore
from detection_worker import DetectionWorker
from inference_profiles import InferencePolicy
from models import AlarmEvent, SessionTransition
from video_source import VideoSourceSpec


class FakeVideoSource:
    last_instance: "FakeVideoSource | None" = None

    def __init__(self, spec: VideoSourceSpec) -> None:
        self.spec = spec
        self.width = 640
        self.height = 480
        self.duration_seconds = 10.0
        self.closed = False
        FakeVideoSource.last_instance = self

    def open(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


class WorkerDeviceTests(unittest.TestCase):
    def events_dir(self) -> Path:
        """给出一个用完就删的 events\\ 路径。

        原来这里直接用 `Path(tempfile.mkdtemp()) / "events"`，四处调用一处不清理 ——
        每跑一次测试就在 %TEMP% 里永久留下几个目录。除了越积越多，它还真误导过人：
        排查打包版是否往 %TEMP% 撒文件时，这几个目录看上去正是"程序在漏"的证据。
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name) / "events"

    def test_alarm_player_plays_configured_wav_file(self) -> None:
        audio_path = Path("warning.wav")
        player = AlarmPlayer(audio_path)
        winsound = Mock(SND_FILENAME=1, SND_NODEFAULT=2)

        with patch.dict(sys.modules, {"winsound": winsound}):
            player._play()

        winsound.PlaySound.assert_called_once_with(
            str(audio_path),
            winsound.SND_FILENAME | winsound.SND_NODEFAULT,
        )
        self.assertFalse(player._is_playing)

    def test_alarm_player_ignores_trigger_while_playing(self) -> None:
        player = AlarmPlayer()
        player._is_playing = True

        with patch("alarm_service.threading.Thread") as thread:
            player.trigger()

        thread.assert_not_called()

    def test_event_store_folds_intrusion_session_updates(self) -> None:
        root = self.events_dir()
        store = EventStore(root)
        event = AlarmEvent(
            source="camera",
            zone_name="警戒区",
            track_id="3",
            entered_at_seconds=10.0,
            alarm_at_seconds=None,
            wall_time="2026-08-13 10:00:00",
            operation_mode="video",
        )
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        store.open_session(event, frame)
        event.alarm_at_seconds = 12.0
        event.status = "alarmed"
        store.mark_alarmed(event, frame)
        event.exited_at_seconds = 15.0
        event.duration_seconds = 5.0
        event.status = "completed"
        store.close_session(event)

        events = store.load_recent()
        self.assertEqual(len(events), 1)
        loaded = events[0]
        self.assertEqual(loaded.session_id, event.session_id)
        self.assertEqual(loaded.alarm_at_seconds, 12.0)
        self.assertEqual(loaded.exited_at_seconds, 15.0)
        self.assertEqual(loaded.duration_seconds, 5.0)
        # 记录里存的是相对路径，靠 resolve_screenshot 解析成实际文件。
        self.assertIsNotNone(store.resolve_screenshot(loaded.entry_screenshot_path))
        self.assertIsNotNone(store.resolve_screenshot(loaded.alarm_screenshot_path))

    def test_event_store_loads_legacy_single_screenshot_record(self) -> None:
        root = self.events_dir()
        root.mkdir()
        screenshot = root / "legacy.jpg"
        cv2.imwrite(str(screenshot), np.zeros((4, 4, 3), dtype=np.uint8))
        legacy = {
            "time": "2026-08-13 10:00:00",
            "video_source": "camera",
            "operation_mode": "monitor",
            "zone_name": "警戒区",
            "track_id": "1",
            "entered_at_seconds": 1786588000.0,
            "alarm_at_seconds": 1786588005.0,
            "screenshot_path": str(screenshot),
        }
        (root / "alarm_events.jsonl").write_text(json.dumps(legacy) + "\n", encoding="utf-8")

        loaded = EventStore(root).load_recent()[0]
        self.assertTrue(loaded.alarmed)
        self.assertEqual(loaded.alarm_screenshot_path, str(screenshot))
        self.assertEqual(loaded.screenshot_path, str(screenshot))

    def test_engine_initialization_failure_keeps_error_status(self) -> None:
        statuses: list[str] = []
        event_store = EventStore(self.events_dir())
        worker = DetectionWorker(
            spec=VideoSourceSpec("test.mp4", operation_mode="video"),
            model_path="model.pt",
            device="cuda:0",
            zones=[],
            event_store=event_store,
        )
        worker.status_changed.connect(statuses.append)

        with (
            patch("detection_worker.VideoSource", FakeVideoSource),
            patch(
                "detection_worker.DetectionEngine",
                side_effect=RuntimeError("CUDA unavailable"),
            ) as engine_factory,
        ):
            worker.run()

        engine_factory.assert_called_once_with("model.pt", [], "cuda:0")
        self.assertTrue(FakeVideoSource.last_instance.closed)
        self.assertTrue(statuses[-1].startswith("检测错误:"))
        self.assertNotIn("检测已停止", statuses[-1])

    def test_worker_forwards_low_power_policy_to_engine(self) -> None:
        statuses: list[str] = []
        event_store = EventStore(self.events_dir())
        policy = InferencePolicy(
            model_path="model_openvino",
            device="cpu",
            imgsz=512,
            detector_interval=4,
            label="CPU 低功耗模式",
        )
        worker = DetectionWorker(
            spec=VideoSourceSpec("test.mp4", operation_mode="video"),
            model_path=policy.model_path,
            device="cpu",
            zones=[],
            event_store=event_store,
            policy=policy,
        )
        worker.status_changed.connect(statuses.append)

        with (
            patch("detection_worker.VideoSource", FakeVideoSource),
            patch(
                "detection_worker.DetectionEngine",
                side_effect=RuntimeError("OpenVINO unavailable"),
            ) as engine_factory,
        ):
            worker.run()

        engine_factory.assert_called_once_with("model_openvino", [], "cpu", policy)
        self.assertTrue(statuses[-1].startswith("检测错误:"))

    def test_disarmed_worker_keeps_preview_without_detection_side_effects(self) -> None:
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        event = AlarmEvent(
            source="test.mp4",
            zone_name="警戒区",
            track_id="7",
            entered_at_seconds=1.0,
            alarm_at_seconds=2.0,
            wall_time="2026-08-17 10:00:00",
            operation_mode="video",
        )

        class ControlledSource:
            def __init__(self, spec: VideoSourceSpec) -> None:
                self.spec = spec
                self.width = 640
                self.height = 480
                self.duration_seconds = 10.0
                self._reads = [(True, frame, 1.0), (False, None, None)]

            def open(self) -> bool:
                return True

            def read(self) -> tuple[bool, np.ndarray | None, float | None]:
                return self._reads.pop(0)

            def is_paused(self) -> bool:
                return False

            def wait_interval(self) -> float:
                return 0.001

            def restart_file(self) -> bool:
                return False

            def close(self) -> None:
                pass

        engine = Mock()
        engine.finalize_active_sessions.return_value = [SessionTransition("exited", event)]
        store = Mock()
        worker = DetectionWorker(
            spec=VideoSourceSpec("test.mp4", operation_mode="video"),
            model_path="model.pt",
            device="cpu",
            zones=[],
            event_store=store,
        )
        worker.alarm_player = Mock()
        frames: list[np.ndarray] = []
        updates: list[AlarmEvent] = []
        alarms: list[AlarmEvent] = []
        worker.frame_ready.connect(frames.append)
        worker.event_updated.connect(updates.append)
        worker.event_ready.connect(alarms.append)
        worker.set_armed(False)

        with (
            patch("detection_worker.VideoSource", ControlledSource),
            patch("detection_worker.DetectionEngine", return_value=engine),
        ):
            worker.run()

        self.assertEqual(len(frames), 1)
        np.testing.assert_array_equal(frames[0], frame)
        engine.process.assert_not_called()
        engine.reset_tracking.assert_called_once_with()
        store.open_session.assert_not_called()
        store.mark_alarmed.assert_not_called()
        store.close_session.assert_not_called()
        worker.alarm_player.trigger.assert_not_called()
        self.assertEqual(updates, [])
        self.assertEqual(alarms, [])

    def test_arm_state_edges_reset_tracking_once_each(self) -> None:
        worker = DetectionWorker(
            spec=VideoSourceSpec("test.mp4", operation_mode="video"),
            model_path="model.pt",
            device="cpu",
            zones=[],
            event_store=Mock(),
        )
        engine = Mock()

        self.assertTrue(worker._sync_armed_state(engine))
        worker.set_armed(False)
        self.assertFalse(worker._sync_armed_state(engine))
        worker.set_armed(False)
        self.assertFalse(worker._sync_armed_state(engine))
        worker.set_armed(True)
        self.assertTrue(worker._sync_armed_state(engine))

        self.assertEqual(engine.reset_tracking.call_count, 2)

    def test_disarming_before_transition_persistence_discards_transition(self) -> None:
        event = AlarmEvent(
            source="camera",
            zone_name="警戒区",
            track_id="7",
            entered_at_seconds=1.0,
            alarm_at_seconds=2.0,
            wall_time="2026-08-17 10:00:00",
            operation_mode="monitor",
        )
        store = Mock()
        worker = DetectionWorker(
            spec=VideoSourceSpec("camera", operation_mode="monitor"),
            model_path="model.pt",
            device="cpu",
            zones=[],
            event_store=store,
        )
        worker.alarm_player = Mock()
        updates: list[AlarmEvent] = []
        alarms: list[AlarmEvent] = []
        worker.event_updated.connect(updates.append)
        worker.event_ready.connect(alarms.append)
        worker.set_armed(False)

        worker._handle_transition(
            SessionTransition("alarmed", event), np.zeros((2, 2, 3), dtype=np.uint8)
        )

        store.mark_alarmed.assert_not_called()
        worker.alarm_player.trigger.assert_not_called()
        self.assertEqual(updates, [])
        self.assertEqual(alarms, [])


class ScriptedStopEvent:
    """替掉 worker 的停止事件，把每次重连退避等了多久记下来。

    重连里用的是 ``_stop_event.wait(delay)`` 而不是 ``time.sleep(delay)``，所以
    这里既能读出退避时长，又能在第 ``limit`` 次等待时假装「停止」被按下，让
    ``run()`` 立刻收尾 —— 测试不必真的睡上 1、2、4…… 秒。
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.delays: list[float] = []
        self._stopped = False

    def wait(self, timeout: float | None = None) -> bool:
        self.delays.append(timeout)
        if len(self.delays) >= self.limit:
            self._stopped = True
        return self._stopped

    def is_set(self) -> bool:
        return self._stopped

    def set(self) -> None:
        self._stopped = True


class ScriptedSource:
    """按脚本返回读帧结果的假视频源；读完脚本就一直失败。"""

    def __init__(self, spec: VideoSourceSpec, reads: list[tuple] | None = None) -> None:
        self.spec = spec
        self.width = 640
        self.height = 480
        self.duration_seconds = 0.0
        self.opens = 0
        self.closes = 0
        self._reads = list(reads or [])

    def open(self) -> bool:
        self.opens += 1
        return True

    def read(self) -> tuple:
        return self._reads.pop(0) if self._reads else (False, None, None)

    def is_paused(self) -> bool:
        return False

    def wait_interval(self) -> float:
        return 0.001

    def restart_file(self) -> bool:
        return False

    def close(self) -> None:
        self.closes += 1


class ReconnectBackoffTests(unittest.TestCase):
    """流中断后的重连：指数退避、永不放弃、等待可被停止打断。

    原来的实现是固定 ``time.sleep(2.0)`` 加「重试 5 次就 break」。对一个安防程序
    来说后者是要命的：网络抖 5 次之后线程自己退出，画面黑着、告警静着，界面上只
    写「视频播放结束」，看的人根本不知道已经没在防了。
    """

    def _worker(self, spec: VideoSourceSpec, stop_limit: int) -> DetectionWorker:
        worker = DetectionWorker(
            spec=spec,
            model_path="model.pt",
            device="cpu",
            zones=[],
            event_store=Mock(),
        )
        worker.alarm_player = Mock()
        worker._stop_event = ScriptedStopEvent(stop_limit)
        return worker

    @staticmethod
    def _engine() -> Mock:
        engine = Mock()
        engine.finalize_active_sessions.return_value = []
        # 这些测试只关心重连，不关心检测结果；给个能解包的返回值就行。
        engine.process.return_value = (np.zeros((12, 16, 3), dtype=np.uint8), [])
        return engine

    def _run(self, worker: DetectionWorker, source: ScriptedSource) -> list[str]:
        statuses: list[str] = []
        worker.status_changed.connect(statuses.append)
        with (
            patch("detection_worker.VideoSource", return_value=source),
            patch("detection_worker.DetectionEngine", return_value=self._engine()),
        ):
            worker.run()
        return statuses

    def test_reconnect_delay_doubles_then_caps(self) -> None:
        delays = [DetectionWorker._reconnect_delay(attempt) for attempt in range(1, 8)]
        self.assertEqual(delays, [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0])

    def test_reconnect_delay_survives_a_very_long_outage(self) -> None:
        """断开几小时后 attempts 会很大，2 ** attempts 直接算会溢出成 inf。"""
        self.assertEqual(DetectionWorker._reconnect_delay(10_000), 30.0)

    def test_stream_break_keeps_retrying_and_never_gives_up(self) -> None:
        spec = VideoSourceSpec("rtsp://camera", operation_mode="monitor")
        source = ScriptedSource(spec)
        worker = self._worker(spec, stop_limit=8)

        statuses = self._run(worker, source)

        # 八次重连都发生了 —— 旧实现在第 5 次就 break 了。
        self.assertEqual(
            worker._stop_event.delays,
            [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0],
        )
        self.assertNotIn("视频播放结束", statuses)
        self.assertIn("视频中断，第 8 次重连，30 秒后重试", statuses)
        # 每次重连都得先 close 再 open，否则句柄会漏；最后 finally 里还会再 close 一次。
        self.assertEqual(source.opens, 8)

    def test_successful_frame_read_resets_the_backoff(self) -> None:
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        spec = VideoSourceSpec("rtsp://camera", operation_mode="monitor")
        source = ScriptedSource(
            spec,
            reads=[
                (False, None, None),
                (False, None, None),
                (True, frame, 1.0),
                (False, None, None),
            ],
        )
        worker = self._worker(spec, stop_limit=3)

        self._run(worker, source)

        # 读到帧之后退避从头开始，不会因为之前抖过就一直等 4 秒。
        self.assertEqual(worker._stop_event.delays, [1.0, 2.0, 1.0])

    def test_stop_during_backoff_wait_ends_immediately(self) -> None:
        spec = VideoSourceSpec("rtsp://camera", operation_mode="monitor")
        source = ScriptedSource(spec)
        worker = self._worker(spec, stop_limit=1)

        statuses = self._run(worker, source)

        # 第一次等待就被打断：不再重开视频源，界面也不该看到「已重新连接」。
        self.assertEqual(worker._stop_event.delays, [1.0])
        self.assertEqual(source.opens, 1)
        self.assertNotIn("视频源已重新连接", statuses)

    def test_file_end_stops_without_any_reconnect(self) -> None:
        spec = VideoSourceSpec("test.mp4", operation_mode="video", loop_playback=False)
        source = ScriptedSource(spec)
        worker = self._worker(spec, stop_limit=1)

        statuses = self._run(worker, source)

        # 文件读到结尾是正常结束，不该退避、也不该重连。
        self.assertEqual(worker._stop_event.delays, [])
        self.assertEqual(source.opens, 1)
        self.assertIn("视频播放结束", statuses)


class ScreenshotStorageTests(unittest.TestCase):
    """取证截图的落盘与解析。

    两个真问题在这里钉住：

    * 裸 ``cv2.imwrite`` 在非 ASCII 路径下会静默失败（返回 False，文件根本没写）。
      程序里之所以一直没出事，是因为 ultralytics 导入时把 ``cv2.imwrite`` 换成了自己
      的 Unicode 安全版本 —— 也就是说"取证能不能落盘"一直在依赖第三方的猴子补丁。
      所以改成 ``imencode`` + ``write_bytes``。
    * 记录里存绝对路径时，绿色版被拷到别处（说明书要求整个文件夹一起拷贝）之后所有
      旧记录都会"取证不可用"。改成存相对 events 目录的路径，并为旧记录保留按文件名
      兜底找回。
    """

    def store(self, name: str = "events") -> EventStore:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return EventStore(Path(temporary.name) / name)

    @staticmethod
    def event() -> AlarmEvent:
        return AlarmEvent(
            source="0",
            zone_name="区域 1",
            track_id="7",
            entered_at_seconds=1.0,
            alarm_at_seconds=2.0,
            wall_time="2026-09-21 10:00:00",
            operation_mode="monitor",
        )

    def test_screenshot_is_written_under_a_non_ascii_path(self) -> None:
        """中文目录下也必须写成功 —— 发布包的目录名就是中文。"""
        store = self.store("行人警戒区域监控")

        event = store.mark_alarmed(self.event(), np.zeros((16, 16, 3), dtype=np.uint8))

        self.assertNotEqual(event.alarm_screenshot_path, "")
        resolved = store.resolve_screenshot(event.alarm_screenshot_path)
        self.assertIsNotNone(resolved)
        self.assertGreater(resolved.stat().st_size, 0)

    def test_stored_path_is_relative_to_the_events_root(self) -> None:
        """存相对路径，绿色版被拷到别处之后旧记录才不会全部失效。"""
        store = self.store()

        event = store.open_session(self.event(), np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertFalse(Path(event.entry_screenshot_path).is_absolute())
        self.assertTrue(event.entry_screenshot_path.startswith("screenshots/"))
        # 往返一次仍然能解析到同一个文件。
        loaded = store.load_recent()[0]
        self.assertEqual(
            store.resolve_screenshot(loaded.entry_screenshot_path),
            store.resolve_screenshot(event.entry_screenshot_path),
        )

    def test_resolve_finds_old_absolute_records_by_filename(self) -> None:
        """旧记录存的是绝对路径；目录换过之后按文件名也能找回。"""
        store = self.store()
        event = store.open_session(self.event(), np.zeros((8, 8, 3), dtype=np.uint8))
        filename = Path(event.entry_screenshot_path).name
        stale = str(Path("D:/旧的安装目录/events/screenshots") / filename)

        self.assertEqual(
            store.resolve_screenshot(stale),
            store.screenshot_dir / filename,
        )

    def test_resolve_returns_none_for_missing_files(self) -> None:
        store = self.store()

        self.assertIsNone(store.resolve_screenshot(""))
        self.assertIsNone(store.resolve_screenshot("screenshots/从来没有过.jpg"))

    def test_empty_frame_is_skipped_with_a_warning(self) -> None:
        """会话收尾时用占位帧，不该写出一张坏图，也不该悄悄吞掉。"""
        store = self.store()

        with self.assertLogs("alarm_service", level="WARNING") as captured:
            event = store.open_session(self.event(), np.empty((0, 0, 3), dtype=np.uint8))

        self.assertEqual(event.entry_screenshot_path, "")
        self.assertTrue(any("帧无效" in line for line in captured.output))

    def test_relative_path_keeps_foreign_paths_absolute(self) -> None:
        """不在 events 目录下的路径保留绝对形式，总比丢了强。"""
        store = self.store()

        self.assertEqual(store.relative_path("D:/别处/x.jpg"), "D:\\别处\\x.jpg")
        self.assertEqual(store.relative_path("screenshots/x.jpg"), "screenshots/x.jpg")

    def test_unavailable_reason_distinguishes_the_three_cases(self) -> None:
        """原来一律显示"不可用"，三种完全不同的情况看起来一模一样。"""
        quiet = AlarmEvent(
            source="0",
            zone_name="区域 1",
            track_id="7",
            entered_at_seconds=1.0,
            alarm_at_seconds=None,
            wall_time="2026-09-21 10:00:00",
            operation_mode="monitor",
        )
        self.assertIn("未触发报警", EventStore.unavailable_reason(quiet, ""))

        alarmed = self.event()
        self.assertIn("截图未写入", EventStore.unavailable_reason(alarmed, ""))
        self.assertIn(
            "文件缺失", EventStore.unavailable_reason(alarmed, "screenshots/x.jpg")
        )


class FrameBackpressureTests(unittest.TestCase):
    """界面跟不上时丢显示帧，但检测一帧都不能少。

    Qt 的跨线程信号是排队投递的，既不合并也不丢弃。1080p 一帧 BGR 就是 6 MB，而
    低功耗预设能跑到 100 fps，界面每帧还要做一次颜色转换和缩放绘制 —— 跟不上时
    队列按秒堆起来，一秒几百 MB，最后内存耗尽。监控模式的读取循环自己也不限速
    （文件模式靠 wait_interval 限了），所以必须由发送端丢帧。
    """

    FRAME_COUNT = 5

    def _drive(
        self,
        acknowledge: bool,
        transitions: list[SessionTransition] | None = None,
    ) -> tuple[list[np.ndarray], "Mock", list[AlarmEvent], list[AlarmEvent], DetectionWorker]:
        frames = [
            np.full((6, 8, 3), index, dtype=np.uint8) for index in range(self.FRAME_COUNT)
        ]
        spec = VideoSourceSpec("rtsp://camera/live", operation_mode="monitor")
        source = ScriptedSource(
            spec,
            reads=[(True, frame, float(index)) for index, frame in enumerate(frames)],
        )
        store = Mock()
        worker = DetectionWorker(
            spec=spec,
            model_path="model.pt",
            device="cpu",
            zones=[],
            event_store=store,
        )
        worker.alarm_player = Mock()
        # 帧读完之后源会一直失败，而监控模式会无限重连；这里让第一次退避等待就
        # 假装「停止」被按下，测试不必真睡。
        worker._stop_event = ScriptedStopEvent(1)

        delivered: list[np.ndarray] = []
        ready: list[AlarmEvent] = []
        updates: list[AlarmEvent] = []
        worker.event_ready.connect(ready.append)
        worker.event_updated.connect(updates.append)
        if acknowledge:
            # 模拟界面：画完一帧就回执，允许下一帧进来（同线程，投递是同步的）。
            worker.frame_ready.connect(lambda frame: (delivered.append(frame), worker.frame_consumed()))
        else:
            worker.frame_ready.connect(delivered.append)

        engine = Mock()
        engine.finalize_active_sessions.return_value = []
        engine.process.side_effect = (
            lambda frame, *args, **kwargs: (frame, list(transitions or []))
        )

        with (
            patch("detection_worker.VideoSource", return_value=source),
            patch("detection_worker.DetectionEngine", return_value=engine),
        ):
            worker.run()

        return delivered, engine, ready, updates, worker

    def test_without_an_acknowledgement_only_the_first_frame_is_queued(self) -> None:
        delivered, engine, _, _, _ = self._drive(acknowledge=False)

        self.assertEqual(len(delivered), 1)
        np.testing.assert_array_equal(delivered[0], np.full((6, 8, 3), 0, dtype=np.uint8))
        # 丢的只是显示：检测每帧都照跑，否则驻留计时会失真。
        self.assertEqual(engine.process.call_count, self.FRAME_COUNT)

    def test_acknowledging_each_frame_delivers_every_frame(self) -> None:
        delivered, engine, _, _, _ = self._drive(acknowledge=True)

        self.assertEqual(len(delivered), self.FRAME_COUNT)
        self.assertEqual(engine.process.call_count, self.FRAME_COUNT)
        for index, frame in enumerate(delivered):
            np.testing.assert_array_equal(frame, np.full((6, 8, 3), index, dtype=np.uint8))

    def test_dropped_frames_still_persist_their_alarms(self) -> None:
        """丢显示不能连带丢掉报警：那才是这个程序存在的意义。"""
        event = AlarmEvent(
            source="rtsp://camera/live",
            zone_name="警戒区",
            track_id="7",
            entered_at_seconds=0.0,
            alarm_at_seconds=2.0,
            wall_time="2026-09-20 10:00:00",
            operation_mode="monitor",
        )
        transitions = [SessionTransition("alarmed", event)]

        delivered, engine, ready, updates, worker = self._drive(
            acknowledge=False, transitions=transitions
        )

        self.assertEqual(len(delivered), 1)
        self.assertEqual(engine.process.call_count, self.FRAME_COUNT)
        self.assertEqual(len(ready), self.FRAME_COUNT)
        self.assertEqual(len(updates), self.FRAME_COUNT)
        self.assertEqual(worker.event_store.mark_alarmed.call_count, self.FRAME_COUNT)
        self.assertEqual(worker.alarm_player.trigger.call_count, self.FRAME_COUNT)

    def test_frame_consumed_lets_the_next_frame_through(self) -> None:
        delivered, engine, _, _, worker = self._drive(acknowledge=False)
        self.assertEqual(len(delivered), 1)
        self.assertEqual(engine.process.call_count, self.FRAME_COUNT)

        # 界面追上进度之后就能继续收帧，不是一次丢帧之后就永久静默。
        worker.frame_consumed()
        self.assertFalse(worker._frame_pending.is_set())

    def test_disarmed_preview_also_drops_frames_it_cannot_keep_up_with(self) -> None:
        spec = VideoSourceSpec("rtsp://camera/live", operation_mode="monitor")
        source = ScriptedSource(
            spec,
            reads=[
                (True, np.zeros((6, 8, 3), dtype=np.uint8), float(index))
                for index in range(self.FRAME_COUNT)
            ],
        )
        worker = DetectionWorker(
            spec=spec,
            model_path="model.pt",
            device="cpu",
            zones=[],
            event_store=Mock(),
        )
        worker._stop_event = ScriptedStopEvent(1)
        delivered: list[np.ndarray] = []
        worker.frame_ready.connect(delivered.append)
        worker.set_armed(False)
        engine = Mock()
        engine.finalize_active_sessions.return_value = []

        with (
            patch("detection_worker.VideoSource", return_value=source),
            patch("detection_worker.DetectionEngine", return_value=engine),
        ):
            worker.run()

        # 撤防时只是预览，更没必要把帧堆在队列里。
        self.assertEqual(len(delivered), 1)


class EventStoreClearTests(unittest.TestCase):
    """「清空记录」按运行模式清理。

    它是本项目里唯一会删数据的代码路径，而且按钮在检测运行中也能点 —— 所以这里
    既要钉住过滤逻辑，也要钉住「读改写之间不能松锁」这件事。
    """

    def store(self) -> EventStore:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return EventStore(Path(temporary.name) / "events")

    @staticmethod
    def _session(operation_mode: str, session_id: str) -> dict:
        return {
            "time": "2026-09-20 10:00:00",
            "video_source": "rtsp://camera/live",
            "operation_mode": operation_mode,
            "zone_name": "警戒区",
            "track_id": "7",
            "entered_at_seconds": 1.0,
            "alarm_at_seconds": 2.0,
            "session_id": session_id,
        }

    def test_clear_without_a_log_file_is_a_no_op(self) -> None:
        store = self.store()

        store.clear("monitor")

        self.assertEqual(store.load_recent(), [])

    def test_clear_by_mode_keeps_the_other_modes_records(self) -> None:
        store = self.store()
        store.root.mkdir(parents=True)
        lines = [
            self._session("monitor", "m1"),
            self._session("video", "v1"),
            self._session("monitor", "m2"),
        ]
        store.log_path.write_text(
            "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
            encoding="utf-8",
        )

        store.clear("monitor")

        remaining = store.load_recent()
        self.assertEqual([event.session_id for event in remaining], ["v1"])
        self.assertEqual(remaining[0].operation_mode, "video")
        # 详情不能只有一张「打开」的记录：报警时刻与状态要一起留下来。
        self.assertEqual(remaining[0].alarm_at_seconds, 2.0)

    def test_clear_without_a_mode_removes_everything(self) -> None:
        store = self.store()
        store.root.mkdir(parents=True)
        store.log_path.write_text(
            json.dumps(self._session("monitor", "m1")) + "\n", encoding="utf-8"
        )

        store.clear()

        self.assertFalse(store.log_path.exists())
        self.assertEqual(store.load_recent(), [])

    def test_clear_keeps_other_mode_records_beyond_ten_thousand(self) -> None:
        """以前是 load_recent(limit=10_000) 读完再重写：要留的那条只要排在
        第 10001 条之前，就会被静默丢掉 —— 为清一个模式而删掉另一个模式的记录。"""
        store = self.store()
        store.root.mkdir(parents=True)
        lines = [json.dumps({"operation_mode": "video", "session_id": "video-old"})]
        lines.extend(
            json.dumps({"operation_mode": "monitor", "session_id": f"m{index}"})
            for index in range(10_050)
        )
        store.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        store.clear("monitor")

        remaining = store.load_recent(limit=1_000_000)
        self.assertEqual([event.session_id for event in remaining], ["video-old"])

    def test_clear_keeps_screenshots_on_disk(self) -> None:
        """对话框承诺「已保存的截图文件将保留」，这里把这个契约钉住。"""
        store = self.store()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        event = AlarmEvent(
            source="rtsp://camera/live",
            zone_name="警戒区",
            track_id="7",
            entered_at_seconds=1.0,
            alarm_at_seconds=2.0,
            wall_time="2026-09-20 10:00:00",
            operation_mode="monitor",
        )
        store.open_session(event, frame)
        screenshot = store.resolve_screenshot(event.entry_screenshot_path)
        self.assertIsNotNone(screenshot)
        self.assertTrue(screenshot.is_file())

        store.clear("monitor")

        self.assertTrue(screenshot.exists())

    def test_broken_lines_do_not_hide_the_readable_records(self) -> None:
        store = self.store()
        store.root.mkdir(parents=True)
        store.log_path.write_text(
            "\n".join(
                [
                    "{不是 json",
                    json.dumps([1, 2, 3]),  # 合法 JSON，但不是对象
                    json.dumps(self._session("video", "v1")),
                    "",
                ]
            ),
            encoding="utf-8",
        )

        events = store.load_recent()

        self.assertEqual([event.session_id for event in events], ["v1"])


if __name__ == "__main__":
    unittest.main()
