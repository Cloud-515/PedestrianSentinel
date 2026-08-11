from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
class AppConfig:
    source: str = "0"
    source_type: str = "camera"
    test_mode: bool = False
    loop_playback: bool = True
    playback_speed: float = 1.0
    model_path: str = "yolo11n.pt"
    inference_device: str = "cpu"
    zones: list[ZoneDefinition] = field(default_factory=list)
    source_size: list[int] = field(default_factory=list)
    display_to_original_scale: dict[str, float] = field(
        default_factory=lambda: {"x": 1.0, "y": 1.0}
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        return cls(
            source=str(data.get("source", "0")),
            source_type=str(data.get("source_type", "camera")),
            test_mode=bool(data.get("test_mode", False)),
            loop_playback=bool(data.get("loop_playback", True)),
            playback_speed=float(data.get("playback_speed", 1.0)),
            model_path=str(data.get("model_path", "yolo11n.pt")),
            inference_device=str(data.get("inference_device", "cpu")),
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
            "loop_playback": self.loop_playback,
            "playback_speed": self.playback_speed,
            "model_path": self.model_path,
            "inference_device": self.inference_device,
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
    alarm_at_seconds: float
    wall_time: str
    screenshot_path: str = ""

    def to_row(self) -> list[str]:
        return [
            self.wall_time,
            self.source,
            self.zone_name,
            self.track_id,
            f"{self.entered_at_seconds:.2f}s",
            f"{self.alarm_at_seconds:.2f}s",
            self.screenshot_path,
        ]
