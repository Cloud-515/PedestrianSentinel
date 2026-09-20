from __future__ import annotations

import logging
import threading
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

    # 流中断后的重连退避：第一次等 1 秒，之后翻倍，最多等 30 秒。上限是个折中 ——
    # 再长会让网络恢复后的接回变慢，再短则长时间断开时会一直高频重开设备、刷日志。
    RECONNECT_BASE_DELAY = 1.0
    RECONNECT_MAX_DELAY = 30.0

    @classmethod
    def _reconnect_delay(cls, attempts: int) -> float:
        """1、2、4、8、16、30、30…… 秒。

        指数先夹住再取幂：断开几个小时之后 attempts 会涨到很大，直接算
        ``2 ** attempts`` 得到的是天文数字（超过 1024 次方就是 inf）。
        """
        exponent = min(max(0, attempts - 1), 20)
        return min(cls.RECONNECT_BASE_DELAY * (2**exponent), cls.RECONNECT_MAX_DELAY)

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
        # 「上一帧界面还没画完」的标志。跨线程信号是排队投递的，Qt 既不合并也不
        # 丢弃，所以界面跟不上时必须由发送端自己丢帧，否则队列会无上限增长。
        self._frame_pending = threading.Event()
        self._paused = False
        self._loop_playback = spec.loop_playback
        self._speed = spec.speed
        self._armed = True
        self._arm_generation = 0
        self._applied_arm_generation = 0

    def stop(self) -> None:
        self._stop_event.set()

    def frame_consumed(self) -> None:
        """界面已经把上一帧画完了，可以收下一帧。由接收帧的槽调用。"""
        self._frame_pending.clear()

    def _emit_frame(self, frame: np.ndarray) -> None:
        """界面还没消化上一帧，这一帧就不再往队列里塞。

        1080p 一帧 BGR 是 6 MB，而界面每帧都要做一次颜色转换加缩放绘制；低功耗
        预设能跑到 100 fps，比界面快得多。此时队列里积压的是帧，而内存增长和画面
        滞后都是按秒算的（每积压一秒就是几百 MB）。监控模式下的读取循环本来也没有
        节流 —— 文件模式那边靠 wait_interval 自己限速，摄像头/网络流没有。

        只丢「显示」，检测照跑：跳过检测会让驻留计时和告警判定跟着失真，那才是
        这个程序的立身之本。标注帧也照画（不能省掉 render），因为同一帧还要用作
        告警取证截图，少画一次就会出现没有画框的取证图。
        """
        if self._frame_pending.is_set():
            return
        self._frame_pending.set()
        self.frame_ready.emit(frame)

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

    def set_armed(self, armed: bool) -> None:
        with self._control_lock:
            if self._armed == armed:
                return
            self._armed = armed
            self._arm_generation += 1

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
                    if source.spec.is_file:
                        # 文件读到结尾是正常结束，不是中断，没有重连的意义。
                        self.status_changed.emit("视频播放结束")
                        break
                    # 摄像头或网络流断了：一直重试，不放弃。安防程序不该因为网线松了
                    # 几次就自己停掉 —— 那之后画面是黑的、告警是静的，而界面上只写着
                    # 「视频播放结束」，看的人根本不知道已经没在防了。
                    reconnect_attempts += 1
                    delay = self._reconnect_delay(reconnect_attempts)
                    self.status_changed.emit(
                        f"视频中断，第 {reconnect_attempts} 次重连，{delay:.0f} 秒后重试"
                    )
                    self._finalize_sessions(engine, last_video_time)
                    source.close()
                    # 用 wait 而不是 sleep：按下停止要立刻醒，别让界面等满整个退避时间。
                    if self._stop_event.wait(delay):
                        break
                    if source.open():
                        engine.reset_tracking()
                        last_video_time = None
                        self.status_changed.emit("视频源已重新连接")
                    continue

                # 计数器在「真的读到帧」之后才归零，而不是 open() 成功就归零：RTSP
                # 地址常常能打开却一帧都不来，那种情况下退避必须继续往上涨。
                reconnect_attempts = 0
                last_video_time = video_time
                armed = self._sync_armed_state(engine)
                if not armed:
                    self._emit_frame(frame)
                    if source.spec.is_file and source.duration_seconds > 0:
                        self.progress_changed.emit(
                            min(1.0, video_time / source.duration_seconds)
                        )
                    if source.spec.is_file:
                        self.msleep(max(1, int(source.wait_interval() * 1000)))
                    continue

                annotated, transitions = engine.process(
                    frame,
                    video_time,
                    self.spec.value,
                    self.spec.operation_mode,
                )
                self._emit_frame(annotated)
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

    def _sync_armed_state(self, engine: DetectionEngine) -> bool:
        with self._control_lock:
            armed = self._armed
            arm_generation = self._arm_generation
        if arm_generation != self._applied_arm_generation:
            # 每次布撤防切换都丢弃旧轨迹，避免跨状态产生误报。
            engine.reset_tracking()
            self._applied_arm_generation = arm_generation
        return armed

    def _handle_transition(self, transition: SessionTransition, frame: np.ndarray) -> None:
        with self._control_lock:
            if not self._armed:
                return
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
