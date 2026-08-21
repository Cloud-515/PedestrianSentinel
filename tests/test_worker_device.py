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
        self.assertTrue(Path(loaded.entry_screenshot_path).exists())
        self.assertTrue(Path(loaded.alarm_screenshot_path).exists())

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


if __name__ == "__main__":
    unittest.main()
