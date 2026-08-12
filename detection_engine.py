from __future__ import annotations

import logging
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO

from models import AlarmEvent, ZoneDefinition

logger = logging.getLogger(__name__)


class DetectionEngine:
    def __init__(
        self,
        model_path: str,
        zones: list[ZoneDefinition],
        device: str,
    ) -> None:
        logger.info("Loading model: %s on device: %s", model_path, device)
        self.model = YOLO(model_path)
        self.device = device
        self.tracker = self._new_tracker()
        self.zones = [ZoneDefinition.from_dict(zone.to_dict()) for zone in zones]
        self.entry_times: dict[tuple[str, Hashable], float] = {}
        self.last_alarm_times: dict[tuple[str, Hashable], float] = {}

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
        names = {zone.name for zone in self.zones}
        self.entry_times = {
            key: value for key, value in self.entry_times.items() if key[0] in names
        }
        self.last_alarm_times = {
            key: value for key, value in self.last_alarm_times.items() if key[0] in names
        }

    def reset_tracking(self) -> None:
        self.tracker = self._new_tracker()
        self.entry_times.clear()
        self.last_alarm_times.clear()

    def process(
        self,
        frame: np.ndarray,
        video_time: float,
        source: str,
        operation_mode: str = "unknown",
    ) -> tuple[np.ndarray, list[AlarmEvent]]:
        result = self.model(
            frame,
            classes=[0],
            verbose=False,
            device=self.device,
        )[0]
        detections = self.tracker.update(sv.Detections.from_ultralytics(result))
        track_ids = self._track_ids(detections)
        in_zone_ids: dict[str, set[Hashable]] = {zone.name: set() for zone in self.zones}
        in_zone_elapsed_seconds: dict[Hashable, float] = {}
        events: list[AlarmEvent] = []

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
                entered_at = self.entry_times.setdefault(key, video_time)
                elapsed_seconds = max(0.0, video_time - entered_at)
                in_zone_elapsed_seconds[track_id] = max(
                    in_zone_elapsed_seconds.get(track_id, 0.0), elapsed_seconds
                )
                last_alarm_at = self.last_alarm_times.get(key)
                if elapsed_seconds < zone.dwell_seconds:
                    continue
                if last_alarm_at is not None and video_time - last_alarm_at < zone.cooldown_seconds:
                    continue
                self.last_alarm_times[key] = video_time
                events.append(
                    AlarmEvent(
                        source=source,
                        operation_mode=operation_mode,
                        zone_name=zone.name,
                        track_id=str(track_id),
                        entered_at_seconds=entered_at,
                        alarm_at_seconds=video_time,
                        wall_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                )

        active_keys = {
            (zone_name, track_id)
            for zone_name, ids in in_zone_ids.items()
            for track_id in ids
        }
        self.entry_times = {
            key: value for key, value in self.entry_times.items() if key in active_keys
        }

        annotated = self._annotate(
            frame,
            detections,
            in_zone_ids,
            in_zone_elapsed_seconds,
        )
        return annotated, events

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
