from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from models import AlarmEvent

logger = logging.getLogger(__name__)


class AlarmPlayer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._is_playing = False

    def trigger(self) -> None:
        with self._lock:
            if self._is_playing:
                return
            self._is_playing = True
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self) -> None:
        try:
            import winsound

            winsound.Beep(1000, 400)
        except (ImportError, RuntimeError) as error:
            logger.warning("Unable to play alarm: %s", error)
        finally:
            with self._lock:
                self._is_playing = False


class EventStore:
    def __init__(self, root: str | Path = "events") -> None:
        self.root = Path(root)
        self.screenshot_dir = self.root / "screenshots"
        self.log_path = self.root / "alarm_events.jsonl"
        self._lock = threading.Lock()

    def record(self, event: AlarmEvent, frame: np.ndarray) -> AlarmEvent:
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        safe_zone = "".join(character if character.isalnum() else "_" for character in event.zone_name)
        filename = f"{event.wall_time.replace(':', '-').replace(' ', '_')}_{safe_zone}_id{event.track_id}.jpg"
        screenshot_path = self.screenshot_dir / filename
        cv2.imwrite(str(screenshot_path), frame)
        event.screenshot_path = str(screenshot_path)
        payload = {
            "time": event.wall_time,
            "video_source": event.source,
            "zone_name": event.zone_name,
            "track_id": event.track_id,
            "entered_at_seconds": event.entered_at_seconds,
            "alarm_at_seconds": event.alarm_at_seconds,
            "screenshot_path": event.screenshot_path,
        }
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as event_file:
                event_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return event

    def load_recent(self, limit: int = 200) -> list[AlarmEvent]:
        if not self.log_path.exists():
            return []
        events: list[AlarmEvent] = []
        with self._lock, self.log_path.open("r", encoding="utf-8") as event_file:
            for line in event_file:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events.append(
                    AlarmEvent(
                        source=str(payload.get("video_source", "")),
                        zone_name=str(payload.get("zone_name", "")),
                        track_id=str(payload.get("track_id", "")),
                        entered_at_seconds=float(payload.get("entered_at_seconds", 0.0)),
                        alarm_at_seconds=float(payload.get("alarm_at_seconds", 0.0)),
                        wall_time=str(payload.get("time", "")),
                        screenshot_path=str(payload.get("screenshot_path", "")),
                    )
                )
        return events[-limit:]

    def clear(self) -> None:
        with self._lock:
            if self.log_path.exists():
                self.log_path.unlink()
