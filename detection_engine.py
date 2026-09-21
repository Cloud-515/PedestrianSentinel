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
        """按「地面 → 行人 → 信息」的层次画这一帧。

        顺序不是随意的，它决定了观感：

        1. **警戒区先落在地上**（贴地渲染：半透明填充 + 近宽远窄的边带）；
        2. **再把行人的像素贴回区域之上** —— 这样边界看起来在人后面，而不是浮在人身上；
        3. **最后才是检测框与标签**，信息层永远在最上面。

        原来是把区域用一条不透明实线画在最后，于是它横穿人身上、又硬又平，看着像贴
        在屏幕上的一条红线。
        """
        annotated = frame.copy()
        track_ids = self._track_ids(detections)
        boxes = [
            tuple(int(value) for value in box) for box in detections.xyxy
        ]

        self._draw_ground_zones(annotated, in_zone_ids)
        self._restore_people(annotated, frame, boxes)

        for zone in self.zones:
            if len(zone.polygon) >= 2:
                # 区域名属于信息层，画在遮挡之后 —— 画在前面的话，路过的人会把名字
                # 擦掉一块，看着像渲染出错。
                self._draw_zone_name(
                    annotated, zone, np.asarray(zone.polygon, dtype=np.float32)
                )

        for index, (x1, y1, x2, y2) in enumerate(boxes):
            track_id = track_ids[index] if index < len(track_ids) else None
            inside = any(track_id in ids for ids in in_zone_ids.values())
            color = (0, 0, 255) if inside else (0, 200, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
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
        return annotated

    # 贴地渲染的参数。
    #
    # 立体感的来源是**透视一致性**：屏幕宽度恒定的线看起来贴在屏幕上，而地面上一段
    # 固定宽度的警戒带，靠近相机的部分看起来更宽。所以边带的宽度随 y 线性增长（画面
    # 底部最宽），不需要相机标定就能读成"躺在地上"。
    #
    # 层次也要分明，否则一片半透明色块只会像蒙在画面上的一层膜：面很淡（SOFT_ALPHA），
    # 边带很实（HARD_ALPHA），边带上面再压一条亮线当作"贴在地面的胶带"，边带下面留
    # 一道接地暗带当接触阴影。
    BAND_WIDTH_RATIO = 0.013
    BAND_MIN_WIDTH = 1.0
    BAND_MAX_WIDTH = 26.0
    SOFT_ALPHA = 0.16
    HARD_ALPHA = 0.92
    ALERT_ALPHA_BONUS = 0.06
    SHADOW_EXTRA_WIDTH = 3.0

    def _band_width(self, y: float) -> float:
        return min(self.BAND_MAX_WIDTH, max(self.BAND_MIN_WIDTH, self.BAND_WIDTH_RATIO * y))

    @staticmethod
    def _lighten(color: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
        return tuple(int(channel + (255 - channel) * ratio) for channel in color)  # type: ignore[return-value]

    def _draw_ground_zones(
        self,
        annotated: np.ndarray,
        in_zone_ids: dict[str, set[Hashable]],
    ) -> None:
        for zone in self.zones:
            if len(zone.polygon) < 2:
                continue
            polygon = np.asarray(zone.polygon, dtype=np.float32)
            if len(polygon) >= 3 and zone.closed:
                self._draw_ground_zone(annotated, zone, polygon, in_zone_ids)
            else:
                # 还没闭合的区域是编辑中的折线，按普通细线画（它还没有"面"可以贴地）
                points = polygon.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    annotated, [points], False, self._bgr(zone.color), 2, cv2.LINE_AA
                )

    def _draw_ground_zone(
        self,
        annotated: np.ndarray,
        zone: ZoneDefinition,
        polygon: np.ndarray,
        in_zone_ids: dict[str, set[Hashable]],
    ) -> None:
        height, width = annotated.shape[:2]
        # 区域可能有一部分在画面外（顶点拖到边缘之外是允许的），所以范围要夹住。
        left = max(0, int(polygon[:, 0].min()) - 4)
        right = min(width, int(polygon[:, 0].max()) + 8)
        top = max(0, int(polygon[:, 1].min()) - 4)
        bottom = min(height, int(polygon[:, 1].max()) + int(self.BAND_MAX_WIDTH) + 6)
        if right - left < 2 or bottom - top < 2:
            return

        roi = annotated[top:bottom, left:right]
        local = polygon - np.array([left, top], dtype=np.float32)
        local_int = local.astype(np.int32)
        color = self._bgr(zone.color)
        alert = bool(in_zone_ids.get(zone.name))
        edges = list(zip(local, np.roll(local, -1, axis=0)))

        # 第一层：很淡的底色 + 边带下方的接地暗带。混一次就够，别为每个元素各混一次
        # —— 每次 addWeighted 都要遍历整个区域，那是白给的成本。
        soft = roi.copy()
        cv2.fillPoly(soft, [local_int], color)
        for start, end in edges:
            shadow = self._band_quad(start, end, extra=self.SHADOW_EXTRA_WIDTH)
            if shadow is not None:
                cv2.fillPoly(soft, [shadow], (48, 48, 48))
        cv2.addWeighted(soft, self.SOFT_ALPHA, roi, 1 - self.SOFT_ALPHA, 0, roi)

        # 第二层：边带（实）+ 上沿的亮线（胶带的上边缘）。
        hard = roi.copy()
        band = color if alert else tuple(int(channel * 0.92) for channel in color)
        highlight = self._lighten(color, 0.35)
        for start, end in edges:
            quad = self._band_quad(start, end)
            if quad is not None:
                cv2.fillPoly(hard, [quad], band)
            start_lip = start[1] - (self._band_width(float(start[1])) / 2)
            end_lip = end[1] - (self._band_width(float(end[1])) / 2)
            cv2.line(
                hard,
                (int(start[0]), int(start_lip)),
                (int(end[0]), int(end_lip)),
                highlight,
                2,
                cv2.LINE_AA,
            )
        alpha = min(0.98, self.HARD_ALPHA + (self.ALERT_ALPHA_BONUS if alert else 0.0))
        cv2.addWeighted(hard, alpha, roi, 1 - alpha, 0, roi)

    def _band_quad(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        extra: float = 0.0,
        offset: float = 0.0,
    ) -> np.ndarray | None:
        """一条边对应的贴地带四边形。

        带子以边界线为**中心**上下各半：地面上的胶带就是这样——它的中心线是区域边界，
        画面里则因为透视而"下宽上窄"。只向下偏移会让上边界往区域内偏、下边界往区域外
        偏，四条边的观感就不一致了。
        """
        start_half = (self._band_width(float(start[1])) + extra) / 2 + offset
        end_half = (self._band_width(float(end[1])) + extra) / 2 + offset
        points = np.array(
            [
                [start[0], start[1] - start_half],
                [end[0], end[1] - end_half],
                [end[0], end[1] + end_half],
                [start[0], start[1] + start_half],
            ],
            dtype=np.int32,
        )
        if points[:, 1].max() - points[:, 1].min() < 1:
            return None
        return points

    def _draw_zone_name(
        self,
        annotated: np.ndarray,
        zone: ZoneDefinition,
        polygon: np.ndarray,
    ) -> None:
        height, width = annotated.shape[:2]
        # 顶点可能在画面外（区域允许拖出边缘），名字要夹回画面内，否则整行字都看不见。
        x = min(max(4, int(polygon[0][0])), max(4, width - 8))
        y = min(max(18, int(polygon[0][1]) - 8), max(18, height - 6))
        # 先画一遍深色粗体当描边：名字会落在各种背景上（深色沥青、浅色地砖），
        # 只有一种颜色时总有一半场合看不清。
        for color, thickness in (((0, 0, 0), 4), (self._bgr(zone.color), 2)):
            cv2.putText(
                annotated,
                zone.name,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                thickness,
                cv2.LINE_AA,
            )

    def _restore_people(
        self,
        annotated: np.ndarray,
        original: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
    ) -> None:
        """把行人的像素贴回区域绘制之上，让边界看起来在人后面。

        只用检测框内接的椭圆近似人体轮廓：上分割模型能拿到真实掩码，但推理会慢一倍
        左右、低功耗预设的 INT8 模型还要重新导出，而椭圆已经足够让边界"断在人身前"。

        只处理与区域包围盒有交集的框：没人靠近区域时这一步一分钱不花。
        """
        bounds = self._zones_bounds(annotated.shape[1], annotated.shape[0])
        if bounds is None:
            return
        left, top, right, bottom = bounds
        height, width = annotated.shape[:2]
        for box_left, box_top, box_right, box_bottom in boxes:
            if (
                box_right <= left
                or box_left >= right
                or box_bottom <= top
                or box_top >= bottom
            ):
                continue
            x0 = max(0, box_left)
            y0 = max(0, box_top)
            x1 = min(width, box_right)
            y1 = min(height, box_bottom)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.ellipse(
                mask,
                ((x1 - x0) // 2, (y1 - y0) // 2),
                (max(1, int((x1 - x0) * 0.46)), max(1, int((y1 - y0) * 0.5))),
                0,
                0,
                360,
                255,
                -1,
            )
            cv2.copyTo(original[y0:y1, x0:x1], mask, annotated[y0:y1, x0:x1])

    def _zones_bounds(self, width: int, height: int) -> tuple[int, int, int, int] | None:
        """所有闭合区域的并集包围盒（夹在画面内）。没有闭合区域时返回 None。"""
        closed = [
            np.asarray(zone.polygon, dtype=np.float32)
            for zone in self.zones
            if zone.closed and len(zone.polygon) >= 3
        ]
        if not closed:
            return None
        points = np.concatenate(closed)
        return (
            max(0, int(points[:, 0].min()) - 8),
            max(0, int(points[:, 1].min()) - 8),
            min(width, int(points[:, 0].max()) + 8),
            min(height, int(points[:, 1].max()) + 8),
        )

    @staticmethod
    def _bgr(color: str) -> tuple[int, int, int]:
        value = color.lstrip("#")
        if len(value) != 6:
            return 0, 0, 255
        red, green, blue = (int(value[offset : offset + 2], 16) for offset in (0, 2, 4))
        return blue, green, red
