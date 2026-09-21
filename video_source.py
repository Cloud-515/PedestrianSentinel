from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2


@dataclass
class VideoSourceSpec:
    value: str
    operation_mode: str = "monitor"
    loop_playback: bool = True
    speed: float = 1.0

    @property
    def is_file(self) -> bool:
        return self.operation_mode == "video"

    def capture_value(self) -> int | str:
        return self.value if self.is_file or not self.value.isdecimal() else int(self.value)


class VideoSource:
    # 播放对齐的容差：落后不到这一截就不折腾，免得每帧都去 seek。
    PACE_TOLERANCE_SECONDS = 0.02
    # 一次最多往前跳这么久。机器卡一大下、或用户刚拖完进度条时，别让它一口气跳过大半段。
    PACE_MAX_SKIP_SECONDS = 2.0
    # 一次要丢的帧数不超过它就用 grab() 逐帧丢，超过才用 seek。
    #
    # 实测（1080p H.264）：read() 10.9ms、grab() 4.2ms、一次 seek 442ms —— seek 要回到
    # 关键帧重新解码，贵得多。落后期望也就几帧，逐帧 grab 便宜两个数量级。
    PACE_GRAB_LIMIT = 90

    def __init__(self, spec: VideoSourceSpec) -> None:
        self.spec = spec
        self.capture: cv2.VideoCapture | None = None
        self.width = 0
        self.height = 0
        self.fps = 25.0
        self.duration_seconds = 0.0
        self._paused = False
        self._step_requested = False
        self._lock = threading.Lock()
        # 播放时钟的锚点：从某一刻起，视频位置应当按速度线性推进。见 pace()。
        self._anchor_clock = time.monotonic()
        self._anchor_position = 0.0
        self._anchor_speed = spec.speed
        self._reanchor_pending = True

    def open(self) -> bool:
        self.capture = cv2.VideoCapture(self.spec.capture_value())
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            self.close()
            return False
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 25.0)
        if self.fps <= 0 or self.fps != self.fps:
            self.fps = 25.0
        frame_count = float(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_seconds = frame_count / self.fps if self.spec.is_file else 0.0
        self._reanchor_pending = True
        return True

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused
            if not paused:
                self._step_requested = False
        if not paused:
            # 暂停期间墙钟还在走，恢复播放时必须重新起算 —— 否则第一帧就会以为
            # 「落后了几十秒」，一跳跳过一大段视频。
            self._reanchor_pending = True

    def toggle_paused(self) -> bool:
        with self._lock:
            self._paused = not self._paused
            if not self._paused:
                self._step_requested = False
            paused = self._paused
        if not paused:
            self._reanchor_pending = True
        return paused

    def request_step(self) -> None:
        with self._lock:
            self._step_requested = True
            self._paused = True
        self._reanchor_pending = True

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def read(self) -> tuple[bool, object | None, float]:
        if self.capture is None:
            return False, None, 0.0
        with self._lock:
            if self._paused and not self._step_requested:
                return False, None, self.position_seconds()
            self._step_requested = False
        timestamp = self.position_seconds()
        ok, frame = self.capture.read()
        if not ok:
            return False, None, timestamp
        if self.spec.is_file:
            timestamp = max(timestamp, float(self.capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0)
        else:
            timestamp = time.time()
        return True, frame, timestamp

    def position_seconds(self) -> float:
        if self.capture is None:
            return 0.0
        return float(self.capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0

    def restart_file(self) -> bool:
        if not self.spec.is_file:
            return False
        self.close()
        return self.open()

    def pace(self) -> int:
        """按墙钟对齐文件播放位置；返回这一次**跳过了多少帧**（0 = 没跳）。

        原来每帧固定睡 ``1/(fps×速度)``，而睡眠是加在推理时间**之上**的：推理一帧要
        0.1 秒的机器放 30fps 视频，每帧实际耗时 0.13 秒，播放只有约 0.25 倍速 ——
        界面上写着 1.00× 却在慢放（现场就是这么发现的）。这里改成按「现在应当播到
        第几秒」对齐：

        * 视频位置还没到 → 睡到那一刻（速度快也不会多睡）；
        * 已经过了 → seek 过去，把中间那些帧**跳掉**，保住播放速度。

        跳掉的帧不参与检测，所以调用方必须把跳帧数报出来（见 DetectionWorker）。
        """
        if not self.spec.is_file or self.capture is None:
            return 0
        current = self.position_seconds()
        if self._reanchor_pending or self.spec.speed != self._anchor_speed:
            self._anchor_position = current
            self._anchor_clock = time.monotonic()
            self._anchor_speed = self.spec.speed
            self._reanchor_pending = False
            return 0
        target = self._anchor_position + (
            time.monotonic() - self._anchor_clock
        ) * max(self.spec.speed, 0.1)
        lag = target - current
        if lag <= self.PACE_TOLERANCE_SECONDS:
            if lag < 0:
                time.sleep(-lag)
            return 0
        # 落后了：往前追到「现在该到的位置」，中间的帧不要了。
        landing = current + min(lag, self.PACE_MAX_SKIP_SECONDS)
        if self.duration_seconds > 0:
            landing = min(landing, self.duration_seconds)
        skip = max(0, int(round((landing - current) * self.fps)))
        if skip == 0:
            return 0
        if skip <= self.PACE_GRAB_LIMIT:
            # 逐帧丢弃：只推进位置、不解出 BGR，比 seek 便宜两个数量级（见 PACE_GRAB_LIMIT）。
            for _ in range(skip):
                if not self.capture.grab():
                    break
        else:
            self.capture.set(cv2.CAP_PROP_POS_MSEC, landing * 1000.0)
        # 锚点取**请求的落点**而不是 seek/grab 之后读回来的位置：OpenCV 的 POS_MSEC 在
        # 跳转之后返回的是上一个解出的帧的时间戳（差一帧），照它起算会每次都多跳一帧。
        self._anchor_position = landing
        self._anchor_clock = time.monotonic()
        return skip
