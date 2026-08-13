from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from alarm_service import AlarmPlayer, EventStore
from detection_engine import DetectionEngine
from inference_profiles import InferencePolicy
from models import AlarmEvent, SessionTransition, ZoneDefinition
from video_source import VideoSource, VideoSourceSpec

logger = logging.getLogger(__name__)


class DetectionWorker(QThread):
    frame_ready = Signal(object)
    event_ready = Signal(object)
    event_updated = Signal(object)
    status_changed = Signal(str)
    source_opened = Signal(int, int, float)
    progress_changed = Signal(float)

    def __init__(
        self,
        spec: VideoSourceSpec,
        model_path: str,
        device: str,
        zones: list[ZoneDefinition],
        event_store: EventStore,
        policy: InferencePolicy | None = None,
        parent: Optional[object] = None,
    ) -> None:
        super().__init__(parent)
        self.spec = spec
        self.model_path = model_path
        self.device = device
        self.policy = policy
        self._zones = [ZoneDefinition.from_dict(zone.to_dict()) for zone in zones]
        self.event_store = event_store
        self.alarm_player = AlarmPlayer()
        self._stop_event = threading.Event()
        self._step_event = threading.Event()
        self._control_lock = threading.Lock()
        self._paused = False
        self._loop_playback = spec.loop_playback
        self._speed = spec.speed

    def stop(self) -> None:
        self._stop_event.set()

    def set_paused(self, paused: bool) -> None:
        with self._control_lock:
            self._paused = paused

    def toggle_paused(self) -> bool:
        with self._control_lock:
            self._paused = not self._paused
            return self._paused

    def step(self) -> None:
        with self._control_lock:
            self._paused = True
        self._step_event.set()

    def set_loop_playback(self, enabled: bool) -> None:
        with self._control_lock:
            self._loop_playback = enabled

    def set_speed(self, speed: float) -> None:
        with self._control_lock:
            self._speed = max(0.1, speed)

    def set_zones(self, zones: list[ZoneDefinition]) -> None:
        with self._control_lock:
            self._zones = [ZoneDefinition.from_dict(zone.to_dict()) for zone in zones]

    def run(self) -> None:
        source = VideoSource(self.spec)
        self._step_event = threading.Event()
        if not source.open():
            self.status_changed.emit(f"无法打开视频源: {self.spec.value}")
            return
        self.source_opened.emit(source.width, source.height, source.duration_seconds)
        self.status_changed.emit(f"正在加载模型，推理设备: {self.device}")
        reconnect_attempts = 0
        last_zones = list(self._zones)
        failed = False
        last_video_time: float | None = None

        try:
            engine = (
                DetectionEngine(self.model_path, self._zones, self.device)
                if self.policy is None
                else DetectionEngine(
                    self.model_path,
                    self._zones,
                    self.device,
                    self.policy,
                )
            )
            mode_label = self.policy.label if self.policy is not None else "标准模式"
            self.status_changed.emit(f"检测运行中：{mode_label}，推理设备: {self.device}")
            while not self._stop_event.is_set():
                with self._control_lock:
                    paused = self._paused
                    loop_playback = self._loop_playback
                    speed = self._speed
                    zones = list(self._zones)
                if zones != last_zones:
                    engine.update_zones(zones)
                    last_zones = zones
                if paused and not self._step_event.is_set():
                    self.msleep(30)
                    continue
                self._step_event.clear()
                source.spec.speed = speed
                ok, frame, video_time = source.read()
                if not ok:
                    if source.is_paused():
                        self.msleep(20)
                        continue
                    if source.spec.is_file and loop_playback:
                        self._finalize_sessions(engine, last_video_time)
                        if source.restart_file():
                            engine.reset_tracking()
                            last_video_time = None
                            self.status_changed.emit("测试视频循环播放")
                            continue
                    reconnect_attempts += 1
                    if source.spec.is_file or reconnect_attempts >= 5:
                        self.status_changed.emit("视频播放结束")
                        break
                    self.status_changed.emit(f"视频中断，正在重连 ({reconnect_attempts}/5)")
                    self._finalize_sessions(engine, last_video_time)
                    source.close()
                    time.sleep(2.0)
                    if source.open():
                        engine.reset_tracking()
                        last_video_time = None
                        continue
                    continue

                reconnect_attempts = 0
                last_video_time = video_time
                annotated, transitions = engine.process(
                    frame,
                    video_time,
                    self.spec.value,
                    self.spec.operation_mode,
                )
                self.frame_ready.emit(annotated)
                if source.spec.is_file and source.duration_seconds > 0:
                    self.progress_changed.emit(min(1.0, video_time / source.duration_seconds))
                for transition in transitions:
                    self._handle_transition(transition, annotated)
                if source.spec.is_file:
                    self.msleep(max(1, int(source.wait_interval() * 1000)))
        except Exception as error:
            failed = True
            logger.exception("Detection worker stopped unexpectedly")
            self.status_changed.emit(f"检测错误: {error}")
        finally:
            if "engine" in locals():
                self._finalize_sessions(engine, last_video_time)
            source.close()
            if not failed:
                self.status_changed.emit("检测已停止")

    def _handle_transition(self, transition: SessionTransition, frame: np.ndarray) -> None:
        event = transition.event
        if transition.kind == "entered":
            self.event_store.open_session(event, frame)
            self.event_updated.emit(event)
        elif transition.kind == "alarmed":
            self.event_store.mark_alarmed(event, frame)
            self.alarm_player.trigger()
            self.event_updated.emit(event)
            self.event_ready.emit(event)
        else:
            self.event_store.close_session(event)
            self.event_updated.emit(event)

    def _finalize_sessions(self, engine: DetectionEngine, video_time: float | None) -> None:
        if video_time is None:
            return
        for transition in engine.finalize_active_sessions(video_time):
            self._handle_transition(transition, np.empty((0, 0, 3), dtype=np.uint8))
