from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from source_history import load_history

logger = logging.getLogger(__name__)

# config.json 的结构版本。目前只有 1，先占个位：以后要改字段含义时，靠它就知道
# 手上这份文件该怎么迁移，而不必去猜字段的有无。
CONFIG_VERSION = 1

DEFAULT_ZONE_NAME = "未命名区域"
DEFAULT_ZONE_COLOR = "#E53935"

# 取证截图的留存默认值，也是设置页里那两个框的初始值。
# 天数按现场经验给 15 天：一次闯入的两张图，15 天足够覆盖事后追溯的窗口，
# 而再往后基本不会有人回头看。容量上限是第二道闸 —— 天数管不住「一天闯进来几百次」
# 那种现场，到了上限就从最旧的删起。
DEFAULT_RETENTION_DAYS = 15
DEFAULT_RETENTION_MB = 2048
MAX_RETENTION_DAYS = 3650
MAX_RETENTION_MB = 102400  # 100 GB

# 远程通知的请求体形状。常量放在这里是因为它们属于配置 schema（要写进 config.json
# 并被校验），notifications.py 再从这里取。
NOTIFICATION_FORMAT_GENERIC = "generic"
NOTIFICATION_FORMAT_TEXT_BOT = "text_bot"
NOTIFICATION_FORMATS = (NOTIFICATION_FORMAT_GENERIC, NOTIFICATION_FORMAT_TEXT_BOT)

_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")

# 界面上显示的中文状态。JSONL 里的 status 字段仍然写英文（机器读的那一份），
# 这里只管给人看的那一份。
_STATUS_LABELS = {
    "active": "进行中",
    "alarmed": "已报警",
    "completed": "已结束",
}


def _warn_field(key: str, value: object, default: object) -> None:
    """坏字段回落默认值，但要出声。

    静默吞掉的话，用户只会看到「我的区域/速度莫名变了」，却没有任何线索；而配置
    是能被手改的文本文件，改错一个字符完全不奇怪。
    """
    logger.warning("配置字段 %s 的值无法解析(%r)，已回落为默认值 %r", key, value, default)


def _coerce_str(value: object, default: str, *, key: str) -> str:
    if value is None:
        return default
    text = value.strip() if isinstance(value, str) else str(value)
    if not text:
        _warn_field(key, value, default)
        return default
    return text


def _coerce_optional_str(value: object, default: str = "") -> str:
    """描述、时间戳这类自由文本：空着是合法的，不算坏数据。"""
    if value is None:
        return default
    return value if isinstance(value, str) else str(value)


def _coerce_bool(value: object, default: bool, *, key: str) -> bool:
    # 字段缺失（None）不是坏数据，不该刷警告。
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # JSON 里 1/0 也是常见的写法，但 true/false 之外的类型仍要留痕。
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    _warn_field(key, value, default)
    return default


def _coerce_float(value: object, default: float, *, key: str) -> float:
    # 字段缺失（None）不是坏数据，不该刷警告。
    if value is None:
        return default
    # bool 是 int 的子类，但 JSON 里的 true 落到阈值字段上只能是坏数据。
    if isinstance(value, bool):
        _warn_field(key, value, default)
        return default
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        _warn_field(key, value, default)
        return default
    if not math.isfinite(number):
        _warn_field(key, value, default)
        return default
    return number


def _coerce_int(value: object, default: int, *, key: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        _warn_field(key, value, default)
        return default
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        _warn_field(key, value, default)
        return default


def _coerce_retention_enabled(data: dict[str, Any]) -> bool:
    """取证截图的自动清理开关。缺这个键时一律按**关闭**处理。

    这个功能第一版是默认开启的（15 天 / 2048 MB），所以升级上来的 config.json 里可能
    只有天数与上限、没有这个开关。此时**不能**按「填了数值就是同意」来推断 —— 那些
    数值是程序自己写进去的默认值，用户从没点过头，而代价是删他的取证材料。
    只把这件事记进日志，让它有迹可循。
    """
    if "screenshot_retention_enabled" in data:
        return _coerce_bool(
            data.get("screenshot_retention_enabled"),
            False,
            key="screenshot_retention_enabled",
        )
    legacy = _coerce_int(data.get("screenshot_retention_days"), 0, key="screenshot_retention_days")
    if legacy > 0 or _coerce_int(data.get("screenshot_retention_mb"), 0, key="screenshot_retention_mb") > 0:
        logger.info(
            "配置里有截图留存的天数/上限但没有开关，按「未启用自动清理」处理"
            "（旧版本默认开启，升级后需要用户自己确认）"
        )
    return False


def _coerce_choice(value: object, default: str, allowed: tuple[str, ...], *, key: str) -> str:
    """枚举型字段：只认白名单里的值。

    不认识的值一律回落默认——保留原值反而更危险，下游会按「不是我认识的那一种」
    走到没想过的分支上去。
    """
    if value is None:
        return default
    text = value if isinstance(value, str) else str(value)
    if text in allowed:
        return text
    _warn_field(key, value, default)
    return default


def _coerce_color(value: object) -> str:
    if value is None:
        return DEFAULT_ZONE_COLOR
    if isinstance(value, str) and _COLOR_PATTERN.match(value.strip()):
        return value.strip()
    _warn_field("zones.color", value, DEFAULT_ZONE_COLOR)
    return DEFAULT_ZONE_COLOR


def _coerce_size(value: object) -> list[int]:
    """画面尺寸只认「至少两个整数」。"""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        if value:
            _warn_field("source_size", value, [])
        return []
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError):
        _warn_field("source_size", value, [])
        return []


def _coerce_polygon(value: object) -> list[list[float]]:
    """只留下真正的顶点。

    config.json 是给人看、也能手改的文本，而 cv2.pointPolygonTest 对形状很挑：
    一个只有 1 个坐标的「点」会让它抛断言错误，异常从检测线程里冒出来，整个检测
    就停了（见 detection_engine.DetectionEngine._contains）。坏点在这里丢掉，
    剩下的顶点仍然可用 —— 比整份配置回落默认值温和得多。
    """
    if not isinstance(value, (list, tuple)):
        _warn_field("zones.polygon", value, [])
        return []
    points: list[list[float]] = []
    rejected: list[object] = []
    for point in value:
        coordinates: list[float] = []
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                coordinates = [float(point[0]), float(point[1])]
            except (TypeError, ValueError):
                coordinates = []
        if len(coordinates) == 2 and all(math.isfinite(item) for item in coordinates):
            points.append(coordinates)
        else:
            rejected.append(point)
    if rejected:
        # 汇总成一条：一个手改坏的多边形可能有几十个坏点，逐条刷屏反而淹掉别的日志。
        logger.warning(
            "配置里有 %d 个多边形顶点无法解析，已丢弃（前几个: %r）",
            len(rejected),
            rejected[:3],
        )
    return points


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _coerce_zone_list(value: object) -> list["ZoneDefinition"]:
    if value is None:
        return []
    if not isinstance(value, list):
        # 手改成 {"区域名": {...}} 这种映射形式时，遍历出来的会是字符串键，
        # 进到 ZoneDefinition.from_dict 就是 'str' object has no attribute 'get'。
        _warn_field("zones", value, [])
        return []
    zones = [ZoneDefinition.from_dict(zone) for zone in value if isinstance(zone, dict)]
    duplicates = sorted(
        name for name, count in Counter(zone.name for zone in zones).items() if count > 1
    )
    if duplicates:
        # 判定与会话都按 (区域名, 目标ID) 记录，重名会让两个区域的会话与冷却互相
        # 串台。界面上建不出重名区域，但 config.json 能手改出来，所以要说一声。
        logger.warning("警戒区域重名（%s），它们的闯入判定会互相干扰", "、".join(duplicates))
    return zones


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
        if not isinstance(data, dict):
            data = {}
        polygon = _coerce_polygon(data.get("polygon", []))
        # 少于三个顶点的多边形圈不出面积，无论文件里怎么写都不能算闭合 ——
        # 让「closed 蕴含 len(polygon) >= 3」成为构造时就成立的约束。
        closed = len(polygon) >= 3 and _coerce_bool(
            data.get("closed", True), True, key="zones.closed"
        )
        return cls(
            name=_coerce_str(data.get("name"), DEFAULT_ZONE_NAME, key="zones.name"),
            polygon=polygon,
            closed=closed,
            dwell_seconds=max(
                0.0,
                _coerce_float(data.get("dwell_seconds"), 2.0, key="zones.dwell_seconds"),
            ),
            cooldown_seconds=max(
                0.0,
                _coerce_float(
                    data.get("cooldown_seconds"), 10.0, key="zones.cooldown_seconds"
                ),
            ),
            color=_coerce_color(data.get("color", DEFAULT_ZONE_COLOR)),
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
        if not isinstance(data, dict):
            data = {}
        return cls(
            name=_coerce_str(data.get("name"), "未命名配置组", key="name"),
            zones=_coerce_zone_list(data.get("zones")),
            description=_coerce_optional_str(data.get("description")),
            created_at=_coerce_optional_str(data.get("created_at")),
            updated_at=_coerce_optional_str(data.get("updated_at")),
            version=_coerce_int(data.get("version"), 1, key="version"),
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
    # 取证截图留存。**默认关闭**：自动删文件这件事必须由用户显式打开，第一版把它
    # 默认开着，等于让程序在用户还没看过设置的情况下就动他的取证材料。
    # 下面两个数值只在开关打开时生效，所以关掉开关不会丢掉用户填过的天数与上限。
    screenshot_retention_enabled: bool = False
    screenshot_retention_days: int = DEFAULT_RETENTION_DAYS
    screenshot_retention_mb: int = DEFAULT_RETENTION_MB
    # 远程通知。默认关闭：往外部地址发数据必须由用户显式打开。
    notification_enabled: bool = False
    notification_url: str = ""
    notification_format: str = NOTIFICATION_FORMAT_GENERIC
    notification_include_screenshot: bool = False
    version: int = CONFIG_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        # 逐字段兜底：一个坏字段只废掉它自己。以前是整份 from_dict 一路裸转换，
        # 任何一个字段类型不对都会抛异常，调用方只能整份回落默认值 —— 用户手改错
        # 一个字符，代价是所有警戒区域一起消失，而退出时那份默认配置还会被写回
        # config.json 覆盖掉原文件。
        if not isinstance(data, dict):
            logger.warning("配置根节点不是对象(%r)，已按空配置处理", type(data).__name__)
            data = {}

        source = _coerce_optional_str(data.get("source"), "0")
        source_type = _coerce_str(data.get("source_type"), "camera", key="source_type")
        test_mode = _coerce_bool(data.get("test_mode"), False, key="test_mode")
        operation_mode = _coerce_str(data.get("operation_mode"), "", key="operation_mode")
        if operation_mode not in {"monitor", "video"}:
            operation_mode = "video" if test_mode or source_type == "file" else "monitor"
        video_source = _coerce_optional_str(
            data.get("video_source"), source if operation_mode == "video" else ""
        )
        monitor_source = _coerce_optional_str(
            data.get("monitor_source"), source if operation_mode == "monitor" else "0"
        )
        scale = data.get("display_to_original_scale")
        if not isinstance(scale, dict):
            scale = {}
        return cls(
            source=source,
            source_type=source_type,
            test_mode=test_mode,
            operation_mode=operation_mode,
            video_source=video_source,
            monitor_source=monitor_source,
            camera_history=load_history(data.get("camera_history")),
            file_history=load_history(data.get("file_history"), file_source=True),
            active_profile=_coerce_optional_str(data.get("active_profile")),
            loop_playback=_coerce_bool(data.get("loop_playback"), True, key="loop_playback"),
            # 取值区间与播放面板的滑块一致(0.25× ~ 4×)，免得载入一个滑块表达不了的值。
            playback_speed=min(
                4.0,
                max(
                    0.25,
                    _coerce_float(data.get("playback_speed"), 1.0, key="playback_speed"),
                ),
            ),
            model_path=_coerce_str(data.get("model_path"), "yolo11n.pt", key="model_path"),
            inference_device=_coerce_str(
                data.get("inference_device"), "cpu", key="inference_device"
            ),
            cpu_low_power_preset=_coerce_bool(
                data.get("cpu_low_power_preset"), False, key="cpu_low_power_preset"
            ),
            zones=_coerce_zone_list(data.get("zones")),
            source_size=_coerce_size(data.get("source_size")),
            display_to_original_scale={
                "x": _coerce_float(scale.get("x"), 1.0, key="display_to_original_scale.x"),
                "y": _coerce_float(scale.get("y"), 1.0, key="display_to_original_scale.y"),
            },
            screenshot_retention_enabled=_coerce_retention_enabled(data),
            screenshot_retention_days=_clamp(
                _coerce_int(
                    data.get("screenshot_retention_days"),
                    DEFAULT_RETENTION_DAYS,
                    key="screenshot_retention_days",
                ),
                0,
                MAX_RETENTION_DAYS,
            ),
            screenshot_retention_mb=_clamp(
                _coerce_int(
                    data.get("screenshot_retention_mb"),
                    DEFAULT_RETENTION_MB,
                    key="screenshot_retention_mb",
                ),
                0,
                MAX_RETENTION_MB,
            ),
            notification_enabled=_coerce_bool(
                data.get("notification_enabled"), False, key="notification_enabled"
            ),
            # 地址允许为空（就等于没配），所以用 optional：空字符串不是坏数据。
            notification_url=_coerce_optional_str(data.get("notification_url")).strip(),
            notification_format=_coerce_choice(
                data.get("notification_format"),
                NOTIFICATION_FORMAT_GENERIC,
                NOTIFICATION_FORMATS,
                key="notification_format",
            ),
            notification_include_screenshot=_coerce_bool(
                data.get("notification_include_screenshot"),
                False,
                key="notification_include_screenshot",
            ),
            # 读进来的就是当前结构了（迁移在上面就地做掉），所以版本号按当前值写，
            # 而不是照抄文件里的旧值。
            version=CONFIG_VERSION,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
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
            "screenshot_retention_enabled": self.screenshot_retention_enabled,
            "screenshot_retention_days": self.screenshot_retention_days,
            "screenshot_retention_mb": self.screenshot_retention_mb,
            "notification_enabled": self.notification_enabled,
            "notification_url": self.notification_url,
            "notification_format": self.notification_format,
            "notification_include_screenshot": self.notification_include_screenshot,
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

    @property
    def status_label(self) -> str:
        """界面上的状态文字。

        ``status`` 本身是要写进 JSONL 的机器字段，保持英文；这里是给人看的那一份，
        所以「没报过警」和「报过警」必须分开说 —— 只照抄 status 的话，界面上会出现
        active/completed 这种英文词，或者把没报警的会话说成「已结束」。
        """
        if not self.alarmed:
            return "未触发报警"
        return _STATUS_LABELS.get(self.status, self.status)

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
            self.status_label,
            f"{self.duration_seconds:.2f}s" if self.duration_seconds is not None else "未结算",
            self.entry_screenshot_path or self.screenshot_path,
            self.alarm_screenshot_path,
        ]


@dataclass
class SessionTransition:
    kind: Literal["entered", "alarmed", "exited"]
    event: AlarmEvent
