from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
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

    # 跳帧提示最多每这么久报一次：跟不上时几乎每帧都要跳，逐帧刷状态栏没法看。
    SKIP_REPORT_INTERVAL_SECONDS = 1.0

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
        show_details: bool = False,
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
        # 画面上的标签要不要带目标编号与置信度（见 DetectionEngine._box_label）。
        self._show_details = show_details
        # 「上一帧界面还没画完」的标志。跨线程信号是排队投递的，Qt 既不合并也不
        # 丢弃，所以界面跟不上时必须由发送端自己丢帧，否则队列会无上限增长。
        self._frame_pending = threading.Event()
        self._paused = False
        self._loop_playback = spec.loop_playback
        self._speed = spec.speed
        self._armed = True
        self._arm_generation = 0
        self._applied_arm_generation = 0
        self._skipped_frames = 0
        self._skip_reported_at = 0.0

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
        节流；文件模式那边靠 pace() 对齐播放进度，它管的是取帧节奏，不是投递。

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

    def set_show_details(self, show_details: bool) -> None:
        """画面标签要不要带目标编号与置信度。运行中改也立刻生效（下一帧就用新写法）。"""
        with self._control_lock:
            self._show_details = show_details

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
        last_show_details = self._show_details
        failed = False
        last_video_time: float | None = None

        try:
            engine = (
                DetectionEngine(
                    self.model_path,
                    self._zones,
                    self.device,
                    show_detection_details=self._show_details,
                )
                if self.policy is None
                else DetectionEngine(
                    self.model_path,
                    self._zones,
                    self.device,
                    self.policy,
                    show_detection_details=self._show_details,
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
                    show_details = self._show_details
                if zones != last_zones:
                    engine.update_zones(zones)
                    last_zones = zones
                if show_details != last_show_details:
                    # 标签写法变了：不用重启检测，下一帧就用新写法画。
                    engine.show_detection_details = show_details
                    last_show_details = show_details
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
                        self._pace_file_playback(source, speed)
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
                    self._pace_file_playback(source, speed)
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

    def _pace_file_playback(self, source: VideoSource, speed: float) -> None:
        """文件播放的节拍：让**视频位置**跟上墙钟，跟不上就把帧跳掉。

        原来每帧固定睡 ``1/(fps×速度)``，睡眠加在推理之上，于是推理比帧间隔慢的机器
        会慢放（1.00× 实际跑 0.25×）。现在由 ``VideoSource.pace`` 按墙钟对齐：该等的
        等，该跳的跳，播放速度始终是设定的那个。

        跳掉的帧**不参与检测**，所以必须报出来 —— 否则看的人以为一直在全量检测，而
        画面已经一段段跳过去了。
        """
        skipped = source.pace()
        if not skipped:
            return
        self._skipped_frames += skipped
        now = time.monotonic()
        if now - self._skip_reported_at >= self.SKIP_REPORT_INTERVAL_SECONDS:
            self._skip_reported_at = now
            self.status_changed.emit(
                f"推理跟不上，已跳帧 {self._skipped_frames} 张"
                f"（保持 {speed:.2f}x 播放；跳过的帧不参与检测）"
            )

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
            if not self._write_record(self.event_store.open_session, event, frame):
                return
            self.event_updated.emit(event)
        elif transition.kind == "alarmed":
            if not self._write_record(self.event_store.mark_alarmed, event, frame):
                return
            # 告警音照旧要响：写盘失败不该让"有人闯进来了"这件事也不响。
            self.alarm_player.trigger()
            self.event_updated.emit(event)
            self.event_ready.emit(event)
        else:
            if not self._write_record(self.event_store.close_session, event):
                return
            self.event_updated.emit(event)

    def _write_record(self, write: Callable[..., object], event: AlarmEvent, *args) -> bool:
        """写报警记录，失败只报出来、**不**让检测停下。

        原来这里一旦抛异常，异常会一路冒到 ``run()`` 的外层 except，整个检测线程随之
        结束 —— 于是"某一条记录写不下去"的代价是"从此不再设防"，而界面上只留一行
        「检测错误」。真实案例：裁片那条路径曾把事件里的 source 遮成了图像数组，
        ``json.dumps`` 当场抛异常，一条报警就把检测打停了。

        失败时不发界面更新：不能让表里出现一条磁盘上没有的记录。
        """
        try:
            write(event, *args)
        except (OSError, TypeError, ValueError) as error:
            logger.exception("报警记录写入失败")
            self.status_changed.emit(f"报警记录写入失败：{error}（检测继续）")
            return False
        return True

    def _finalize_sessions(self, engine: DetectionEngine, video_time: float | None) -> None:
        if video_time is None:
            return
        for transition in engine.finalize_active_sessions(video_time):
            self._handle_transition(transition, np.empty((0, 0, 3), dtype=np.uint8))
