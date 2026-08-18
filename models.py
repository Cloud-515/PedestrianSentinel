from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from source_history import load_history


@dataclass
class ZoneDefinition:
    name: str
    polygon: list[list[float]] = field(default_factory=list)
    closed: bool = False
    dwell_seconds: float = 2.0
    cooldown_seconds: float = 10.0
    color: str = "#E53935"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZoneDefinition":
        polygon = [list(map(float, point)) for point in data.get("polygon", [])]
        return cls(
            name=str(data.get("name", "未命名区域")),
            polygon=polygon,
            closed=bool(data.get("closed", len(polygon) >= 3)),
            dwell_seconds=float(data.get("dwell_seconds", 2.0)),
            cooldown_seconds=float(data.get("cooldown_seconds", 10.0)),
            color=str(data.get("color", "#E53935")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ZoneProfile:
    name: str
    zones: list[ZoneDefinition] = field(default_factory=list)
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    version: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZoneProfile":
        return cls(
            name=str(data.get("name", "未命名配置组")),
            zones=[ZoneDefinition.from_dict(zone) for zone in data.get("zones", [])],
            description=str(data.get("description", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            version=int(data.get("version", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "zones": [zone.to_dict() for zone in self.zones],
        }


@dataclass
class AppConfig:
    source: str = "0"
    source_type: str = "camera"
    test_mode: bool = False
    operation_mode: str = "monitor"
    video_source: str = ""
    monitor_source: str = "0"
    camera_history: list[str] = field(default_factory=list)
    file_history: list[str] = field(default_factory=list)
    active_profile: str = ""
    loop_playback: bool = True
    playback_speed: float = 1.0
    model_path: str = "yolo11n.pt"
    inference_device: str = "cpu"
    cpu_low_power_preset: bool = False
    zones: list[ZoneDefinition] = field(default_factory=list)
    source_size: list[int] = field(default_factory=list)
    display_to_original_scale: dict[str, float] = field(
        default_factory=lambda: {"x": 1.0, "y": 1.0}
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        source = str(data.get("source", "0"))
        source_type = str(data.get("source_type", "camera"))
        test_mode = bool(data.get("test_mode", False))
        operation_mode = str(data.get("operation_mode", ""))
        if operation_mode not in {"monitor", "video"}:
            operation_mode = "video" if test_mode or source_type == "file" else "monitor"
        video_source = str(
            data.get("video_source", source if operation_mode == "video" else "")
        )
        monitor_source = str(
            data.get("monitor_source", source if operation_mode == "monitor" else "0")
        )
        return cls(
            source=source,
            source_type=source_type,
            test_mode=test_mode,
            operation_mode=operation_mode,
            video_source=video_source,
            monitor_source=monitor_source,
            camera_history=load_history(data.get("camera_history")),
            file_history=load_history(data.get("file_history"), file_source=True),
            active_profile=str(data.get("active_profile", "")),
            loop_playback=bool(data.get("loop_playback", True)),
            playback_speed=float(data.get("playback_speed", 1.0)),
            model_path=str(data.get("model_path", "yolo11n.pt")),
            inference_device=str(data.get("inference_device", "cpu")),
            cpu_low_power_preset=bool(data.get("cpu_low_power_preset", False)),
            zones=[ZoneDefinition.from_dict(zone) for zone in data.get("zones", [])],
            source_size=[int(value) for value in data.get("source_size", [])[:2]],
            display_to_original_scale={
                "x": float(data.get("display_to_original_scale", {}).get("x", 1.0)),
                "y": float(data.get("display_to_original_scale", {}).get("y", 1.0)),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_type": self.source_type,
            "test_mode": self.test_mode,
            "operation_mode": self.operation_mode,
            "video_source": self.video_source,
            "monitor_source": self.monitor_source,
            "camera_history": self.camera_history,
            "file_history": self.file_history,
            "active_profile": self.active_profile,
            "loop_playback": self.loop_playback,
            "playback_speed": self.playback_speed,
            "model_path": self.model_path,
            "inference_device": self.inference_device,
            "cpu_low_power_preset": self.cpu_low_power_preset,
            "source_size": self.source_size,
            "polygon_coordinate_space": "original_video_pixels",
            "display_to_original_scale": self.display_to_original_scale,
            "zones": [zone.to_dict() for zone in self.zones],
        }


@dataclass
class AlarmEvent:
    source: str
    zone_name: str
    track_id: str
    entered_at_seconds: float
    alarm_at_seconds: float | None
    wall_time: str
    operation_mode: str = "unknown"
    screenshot_path: str = ""
    session_id: str = field(default_factory=lambda: uuid4().hex)
    exited_at_seconds: float | None = None
    duration_seconds: float | None = None
    entry_screenshot_path: str = ""
    alarm_screenshot_path: str = ""
    status: Literal["active", "alarmed", "completed"] = "active"

    def __post_init__(self) -> None:
        if self.alarm_screenshot_path and not self.screenshot_path:
            self.screenshot_path = self.alarm_screenshot_path
        elif self.screenshot_path and not self.alarm_screenshot_path:
            self.alarm_screenshot_path = self.screenshot_path

    @property
    def alarmed(self) -> bool:
        return self.alarm_at_seconds is not None

    @staticmethod
    def _format_timestamp(value: float) -> str:
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def format_event_time(self, value: float | None, precision: int = 2) -> str:
        if value is None:
            return "未记录"
        if self.operation_mode == "monitor":
            return self._format_timestamp(value)
        return f"{value:.{precision}f}s"

    def to_row(self) -> list[str]:
        return [
            self.wall_time,
            self.source,
            self.zone_name,
            self.track_id,
            self.format_event_time(self.entered_at_seconds),
            self.format_event_time(self.alarm_at_seconds),
            self.format_event_time(self.exited_at_seconds),
            "未触发报警" if not self.alarmed else self.status,
            f"{self.duration_seconds:.2f}s" if self.duration_seconds is not None else "未结算",
            self.entry_screenshot_path or self.screenshot_path,
            self.alarm_screenshot_path,
        ]


@dataclass
class SessionTransition:
    kind: Literal["entered", "alarmed", "exited"]
    event: AlarmEvent
