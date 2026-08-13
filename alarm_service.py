from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from uuid import uuid4

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

    @staticmethod
    def _event_payload(event: AlarmEvent) -> dict[str, object]:
        return {
            "session_id": event.session_id,
            "time": event.wall_time,
            "video_source": event.source,
            "operation_mode": event.operation_mode,
            "zone_name": event.zone_name,
            "track_id": event.track_id,
            "entered_at_seconds": event.entered_at_seconds,
            "alarm_at_seconds": event.alarm_at_seconds,
            "exited_at_seconds": event.exited_at_seconds,
            "duration_seconds": event.duration_seconds,
            "entry_screenshot_path": event.entry_screenshot_path,
            "alarm_screenshot_path": event.alarm_screenshot_path,
            "screenshot_path": event.screenshot_path,
            "status": event.status,
        }

    def _append(self, payload: dict[str, object]) -> None:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as event_file:
                event_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _save_screenshot(self, event: AlarmEvent, frame: np.ndarray, kind: str) -> str:
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{event.session_id}_{kind}.jpg"
        screenshot_path = self.screenshot_dir / filename
        if not cv2.imwrite(str(screenshot_path), frame):
            return ""
        return str(screenshot_path)

    def open_session(self, event: AlarmEvent, frame: np.ndarray) -> AlarmEvent:
        event.entry_screenshot_path = self._save_screenshot(event, frame, "entry")
        self._append({"schema_version": 2, "action": "opened", "event": self._event_payload(event)})
        return event

    def mark_alarmed(self, event: AlarmEvent, frame: np.ndarray) -> AlarmEvent:
        event.alarm_screenshot_path = self._save_screenshot(event, frame, "alarm")
        event.screenshot_path = event.alarm_screenshot_path or event.entry_screenshot_path
        self._append(
            {
                "schema_version": 2,
                "action": "alarmed",
                "session_id": event.session_id,
                "alarm_at_seconds": event.alarm_at_seconds,
                "alarm_screenshot_path": event.alarm_screenshot_path,
                "screenshot_path": event.screenshot_path,
                "status": event.status,
            }
        )
        return event

    def close_session(self, event: AlarmEvent) -> AlarmEvent:
        self._append(
            {
                "schema_version": 2,
                "action": "closed",
                "session_id": event.session_id,
                "exited_at_seconds": event.exited_at_seconds,
                "duration_seconds": event.duration_seconds,
                "status": event.status,
            }
        )
        return event

    def record(self, event: AlarmEvent, frame: np.ndarray) -> AlarmEvent:
        event.alarm_at_seconds = event.alarm_at_seconds or event.entered_at_seconds
        event.status = "alarmed"
        self.open_session(event, frame)
        return self.mark_alarmed(event, frame)

    def load_recent(
        self,
        limit: int = 200,
        operation_mode: str | None = None,
    ) -> list[AlarmEvent]:
        if not self.log_path.exists():
            return []
        events_by_session: dict[str, AlarmEvent] = {}
        order: list[str] = []
        with self._lock, self.log_path.open("r", encoding="utf-8") as event_file:
            for line in event_file:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("schema_version") == 2:
                    self._apply_v2_payload(payload, events_by_session, order)
                    continue
                event = self._event_from_payload(payload)
                events_by_session[event.session_id] = event
                if event.session_id not in order:
                    order.append(event.session_id)
        events = [events_by_session[session_id] for session_id in order]
        if operation_mode is not None:
            events = [event for event in events if event.operation_mode == operation_mode]
        return events[-limit:]

    @staticmethod
    def _event_from_payload(payload: dict[str, object]) -> AlarmEvent:
        screenshot_path = str(payload.get("screenshot_path", ""))
        return AlarmEvent(
            source=str(payload.get("video_source", payload.get("source", ""))),
            zone_name=str(payload.get("zone_name", "")),
            track_id=str(payload.get("track_id", "")),
            entered_at_seconds=float(payload.get("entered_at_seconds", 0.0)),
            alarm_at_seconds=(
                float(payload["alarm_at_seconds"])
                if payload.get("alarm_at_seconds") is not None
                else None
            ),
            wall_time=str(payload.get("time", payload.get("wall_time", ""))),
            operation_mode=str(payload.get("operation_mode", "unknown")),
            screenshot_path=screenshot_path,
            session_id=str(payload.get("session_id", payload.get("event_id", ""))) or uuid4().hex,
            exited_at_seconds=(
                float(payload["exited_at_seconds"])
                if payload.get("exited_at_seconds") is not None
                else None
            ),
            duration_seconds=(
                float(payload["duration_seconds"])
                if payload.get("duration_seconds") is not None
                else None
            ),
            entry_screenshot_path=str(payload.get("entry_screenshot_path", "")),
            alarm_screenshot_path=str(payload.get("alarm_screenshot_path", screenshot_path)),
            status=str(payload.get("status", "completed" if payload.get("exited_at_seconds") is not None else "alarmed")),
        )

    @classmethod
    def _apply_v2_payload(
        cls,
        payload: dict[str, object],
        events_by_session: dict[str, AlarmEvent],
        order: list[str],
    ) -> None:
        action = str(payload.get("action", ""))
        if action == "opened":
            event_payload = payload.get("event", {})
            if not isinstance(event_payload, dict):
                return
            event = cls._event_from_payload(event_payload)
            events_by_session[event.session_id] = event
            order.append(event.session_id)
            return
        session_id = str(payload.get("session_id", ""))
        event = events_by_session.get(session_id)
        if event is None:
            return
        if action == "alarmed":
            event.alarm_at_seconds = float(payload["alarm_at_seconds"])
            event.alarm_screenshot_path = str(payload.get("alarm_screenshot_path", ""))
            event.screenshot_path = str(payload.get("screenshot_path", event.alarm_screenshot_path))
            event.status = "alarmed"
        elif action == "closed":
            event.exited_at_seconds = float(payload["exited_at_seconds"])
            event.duration_seconds = float(payload["duration_seconds"])
            event.status = "completed"

    def clear(self, operation_mode: str | None = None) -> None:
        with self._lock:
            if not self.log_path.exists():
                return
            if operation_mode is None:
                self.log_path.unlink()
                return
        retained = [
            event
            for event in self.load_recent(limit=10_000)
            if event.operation_mode != operation_mode
        ]
        with self._lock:
            self.log_path.write_text(
                "".join(
                    json.dumps(
                        {"schema_version": 2, "action": "opened", "event": self._event_payload(event)},
                        ensure_ascii=False,
                    )
                    + "\n"
                    for event in retained
                ),
                encoding="utf-8",
            )
