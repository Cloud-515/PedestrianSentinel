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

    def toggle_paused(self) -> bool:
        with self._lock:
            self._paused = not self._paused
            if not self._paused:
                self._step_requested = False
            return self._paused

    def request_step(self) -> None:
        with self._lock:
            self._step_requested = True
            self._paused = True

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

    def wait_interval(self) -> float:
        if not self.spec.is_file:
            return 0.01
        return max(0.001, 1.0 / self.fps / max(self.spec.speed, 0.1))
