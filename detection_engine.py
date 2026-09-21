from __future__ import annotations

import logging
from collections.abc import Hashable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO

from inference_profiles import InferencePolicy
from models import AlarmEvent, SessionTransition, ZoneDefinition

logger = logging.getLogger(__name__)


@dataclass
class ActiveIntrusion:
    event: AlarmEvent
    last_seen_at_seconds: float


class DetectionEngine:
    def __init__(
        self,
        model_path: str,
        zones: list[ZoneDefinition],
        device: str,
        policy: InferencePolicy | None = None,
    ) -> None:
        self.policy = policy or InferencePolicy(model_path=model_path, device=device)
        logger.info(
            "Loading model: %s on device: %s (%s)",
            self.policy.model_path,
            self.policy.device,
            self.policy.label,
        )
        self.model = YOLO(self.policy.model_path)
        self.device = self.policy.device
        self.tracker = self._new_tracker()
        self.zones = [ZoneDefinition.from_dict(zone.to_dict()) for zone in zones]
        self.active_sessions: dict[tuple[str, Hashable], ActiveIntrusion] = {}
        # 上次报警时刻，按 (区域名, 目标ID) 记。必须活得比会话长 —— 冷却要压制的
        # 恰恰是「离开又立刻回来」产生的下一个会话，而会话本身在离开时就没了。
        self._last_alarm_at: dict[tuple[str, Hashable], float] = {}
        self._processed_frames = 0

    @staticmethod
    def _new_tracker() -> ByteTrackTracker:
        return ByteTrackTracker(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            frame_rate=30,
            minimum_consecutive_frames=1,
            minimum_iou_threshold=0.1,
            high_conf_det_threshold=0.6,
        )

    def update_zones(self, zones: list[ZoneDefinition]) -> None:
        self.zones = [ZoneDefinition.from_dict(zone.to_dict()) for zone in zones]

    def reset_tracking(self) -> None:
        self.tracker = self._new_tracker()
        self.active_sessions.clear()
        # 冷却记录也要清：循环播放重头开始时 video_time 会跳回 0，留着旧时刻会让
        # 差值变成负数，把报警一直压住直到播放位置追上来。
        self._last_alarm_at.clear()
        self._processed_frames = 0

    def finalize_active_sessions(self, video_time: float) -> list[SessionTransition]:
        transitions = [
            self._close_session(key, video_time)
            for key in list(self.active_sessions)
        ]
        return transitions

    def _close_session(
        self,
        key: tuple[str, Hashable],
        video_time: float,
    ) -> SessionTransition:
        session = self.active_sessions.pop(key)
        event = session.event
        event.exited_at_seconds = video_time
        event.duration_seconds = max(0.0, video_time - event.entered_at_seconds)
        event.status = "completed"
        return SessionTransition("exited", event)

    @property
    def entry_times(self) -> dict[tuple[str, Hashable], float]:
        return {
            key: session.event.entered_at_seconds
            for key, session in self.active_sessions.items()
        }

    def _cooldown_elapsed(
        self,
        key: tuple[str, Hashable],
        zone: ZoneDefinition,
        video_time: float,
    ) -> bool:
        last_alarm_at = self._last_alarm_at.get(key)
        if last_alarm_at is None or zone.cooldown_seconds <= 0:
            return True
        elapsed = video_time - last_alarm_at
        # 时间轴倒退时差值为负 —— 循环播放回到片头、或换了视频源让 video_time 重新
        # 从 0 起算。这时按冷却已过处理，否则报警会被一直压住，直到播放位置爬回旧时刻。
        if elapsed < 0:
            return True
        return elapsed >= zone.cooldown_seconds

    def _prune_alarm_history(self, video_time: float) -> None:
        """丢掉已经不起压制作用的冷却记录。

        这个 dict 按 (区域名, 目标ID) 记，而 ByteTrack 的 ID 只增不减，长时间跑
        下来会无限增长。冷却期已过的条目对判断不再有任何影响，删掉是等价的；区域
        被改名或删除后，它名下的条目也永远不会再被查到。
        """
        if not self._last_alarm_at:
            return
        zones_by_name = {zone.name: zone for zone in self.zones}
        for key in list(self._last_alarm_at):
            zone = zones_by_name.get(key[0])
            if zone is None or self._cooldown_elapsed(key, zone, video_time):
                del self._last_alarm_at[key]

    def process(
        self,
        frame: np.ndarray,
        video_time: float,
        source: str,
        operation_mode: str = "unknown",
        render: bool = True,
    ) -> tuple[np.ndarray, list[SessionTransition]]:
        self._processed_frames += 1
        detect_this_frame = (
            (self._processed_frames - 1) % self.policy.detector_interval == 0
        )
        if detect_this_frame:
            inference_kwargs: dict[str, Any] = {
                "classes": [0],
                "verbose": False,
                "device": self.device,
            }
            if self.policy.imgsz is not None:
                inference_kwargs["imgsz"] = self.policy.imgsz
            result = self.model(frame, **inference_kwargs)[0]
            raw_detections = sv.Detections.from_ultralytics(result)
            detections = (
                self.tracker.update(raw_detections, timestamp=video_time)
                if self.policy.uses_tracker_prediction
                else self.tracker.update(raw_detections)
            )
        else:
            detections = self.tracker.tracked_objects
        track_ids = self._track_ids(detections)
        in_zone_ids: dict[str, set[Hashable]] = {zone.name: set() for zone in self.zones}
        in_zone_elapsed_seconds: dict[Hashable, float] = {}
        transitions: list[SessionTransition] = []

        for index, track_id in enumerate(track_ids):
            if track_id is None or index >= len(detections.xyxy):
                continue
            x1, y1, x2, y2 = detections.xyxy[index]
            anchor = ((float(x1) + float(x2)) / 2, float(y2))
            for zone in self.zones:
                if not zone.closed or len(zone.polygon) < 3 or not self._contains(zone, anchor):
                    continue
                in_zone_ids[zone.name].add(track_id)
                key = (zone.name, track_id)
                session = self.active_sessions.get(key)
                if session is None:
                    event = AlarmEvent(
                        source=source,
                        operation_mode=operation_mode,
                        zone_name=zone.name,
                        track_id=str(track_id),
                        entered_at_seconds=video_time,
                        alarm_at_seconds=None,
                        wall_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    session = ActiveIntrusion(event, video_time)
                    self.active_sessions[key] = session
                    transitions.append(SessionTransition("entered", event))
                session.last_seen_at_seconds = video_time
                elapsed_seconds = max(0.0, video_time - session.event.entered_at_seconds)
                in_zone_elapsed_seconds[track_id] = max(
                    in_zone_elapsed_seconds.get(track_id, 0.0), elapsed_seconds
                )
                if not session.event.alarmed and elapsed_seconds >= zone.dwell_seconds:
                    # 冷却期内不对同一目标同一区域重复报警。没有这道闸，目标在区域
                    # 边界徘徊、或 ByteTrack 丢一帧 ID 又找回，都会关掉旧会话再开一个
                    # 新的、重新计时、再次报警 —— 每次都放告警音、写 JSONL、存一张
                    # 截图。被压住的会话不会被标记成已报警，所以只要目标还在区内，
                    # 等冷却期过去它仍会补报一次，持续闯入不会被永久吞掉。
                    if self._cooldown_elapsed(key, zone, video_time):
                        session.event.alarm_at_seconds = video_time
                        session.event.status = "alarmed"
                        self._last_alarm_at[key] = video_time
                        transitions.append(SessionTransition("alarmed", session.event))

        active_keys = {
            (zone_name, track_id)
            for zone_name, ids in in_zone_ids.items()
            for track_id in ids
        }
        for key in list(self.active_sessions):
            if key not in active_keys:
                transitions.append(self._close_session(key, video_time))

        self._prune_alarm_history(video_time)

        if render:
            annotated = self._annotate(
                frame,
                detections,
                in_zone_ids,
                in_zone_elapsed_seconds,
            )
        else:
            annotated = frame
        return annotated, transitions

    @staticmethod
    def _track_ids(detections: sv.Detections) -> list[Hashable | None]:
        if detections.tracker_id is None:
            return [None] * len(detections)
        values: list[Hashable | None] = []
        for track_id in detections.tracker_id:
            value: Any = track_id.item() if hasattr(track_id, "item") else track_id
            values.append(None if value == -1 else value)
        return values

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total_milliseconds = max(0, int(seconds * 1000))
        whole_seconds, milliseconds = divmod(total_milliseconds, 1000)
        return f"{whole_seconds}.{milliseconds:03d}s"

    @staticmethod
    def _contains(zone: ZoneDefinition, point: tuple[float, float]) -> bool:
        polygon = np.asarray(zone.polygon, dtype=np.float32)
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def _annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        in_zone_ids: dict[str, set[Hashable]],
        in_zone_elapsed_seconds: dict[Hashable, float],
    ) -> np.ndarray:
        annotated = frame.copy()
        track_ids = self._track_ids(detections)
        for index, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = [int(value) for value in box]
            track_id = track_ids[index] if index < len(track_ids) else None
            inside = any(track_id in ids for ids in in_zone_ids.values())
            color = (0, 0, 255) if inside else (0, 200, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id}" if track_id is not None else "Person"
            if inside:
                label += f" IN ZONE {self._format_elapsed(in_zone_elapsed_seconds[track_id])}"
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        for zone in self.zones:
            if len(zone.polygon) < 2:
                continue
            points = np.asarray(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
            color = self._bgr(zone.color)
            cv2.polylines(annotated, [points], zone.closed and len(zone.polygon) >= 3, color, 2)
            x, y = points[0, 0]
            cv2.putText(
                annotated,
                zone.name,
                (int(x), max(20, int(y) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        return annotated

    @staticmethod
    def _bgr(color: str) -> tuple[int, int, int]:
        value = color.lstrip("#")
        if len(value) != 6:
            return 0, 0, 255
        red, green, blue = (int(value[offset : offset + 2], 16) for offset in (0, 2, 4))
        return blue, green, red
