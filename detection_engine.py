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
    # 最后一次**真实检出**发生在第几个检测轮次（预测帧不算）。见 PRESENCE_GRACE_CYCLES。
    last_detected_cycle: int = 0


class DetectionEngine:
    def __init__(
        self,
        model_path: str,
        zones: list[ZoneDefinition],
        device: str,
        policy: InferencePolicy | None = None,
        show_detection_details: bool = False,
    ) -> None:
        self.policy = policy or InferencePolicy(model_path=model_path, device=device)
        # 画面上的标签要不要带上目标编号与置信度。默认不带 —— 见 _box_label。
        # 这个属性可以在运行中改（检测线程每帧都会读它），所以设置页勾上就立刻生效。
        self.show_detection_details = show_detection_details
        logger.info(
            "Loading model: %s on device: %s (%s)",
            self.policy.model_path,
            self.policy.device,
            self.policy.label,
        )
        # task 必须显式声明。不声明时 ultralytics 会**按文件名猜**任务
        # （见 nn/tasks.py 的 guess_model_task），而低功耗那个模型是个目录
        # `yolo11n_int8_openvino_model` —— 名字里既没有 "detect" 也没有 "-seg" 之类
        # 的字样，于是它猜不出来，在控制台打一条 "Unable to automatically guess model
        # task" 的 WARNING。那条警告本身无害（它假设的 detect 恰好是对的），但它看着
        # 像程序出了问题，而现场排查的人得先花时间解释一遍。本程序只做行人检测框，
        # 声明 detect 比让库去猜更准确。
        self.model = YOLO(self.policy.model_path, task="detect")
        self.device = self.policy.device
        self.tracker = self._new_tracker()
        self.zones = [ZoneDefinition.from_dict(zone.to_dict()) for zone in zones]
        # 所有闭合区域的并集包围盒（画面坐标），用于「只检测警戒区」这一档；见
        # InferencePolicy.detect_region_margin 与 _detect_crop。
        self._zone_bounds = self._zone_union(self.zones)
        self.active_sessions: dict[tuple[str, Hashable], ActiveIntrusion] = {}
        # 上次报警时刻，按 (区域名, 目标ID) 记。必须活得比会话长 —— 冷却要压制的
        # 恰恰是「离开又立刻回来」产生的下一个会话，而会话本身在离开时就没了。
        self._last_alarm_at: dict[tuple[str, Hashable], float] = {}
        # 每条轨迹最后一次**真实检出**的置信度。预测帧（tracked_objects）不带置信度，
        # 而标签和报警记录都需要一个数，所以按 ID 记着。只保留本帧仍在的 ID，见
        # _remember_confidences：ByteTrack 的 ID 只增不减，不筛就会一直长。
        self._last_confidence: dict[Hashable, float] = {}
        # 每条轨迹最后一次被**真正匹配上**的位置，以及那是第几个检测轮次。跳过的帧要用
        # 它把「已经跟丢的轨迹」钉住，见 _stabilise_predicted_boxes。
        self._last_matched_box: dict[Hashable, np.ndarray] = {}
        self._last_matched_cycle: dict[Hashable, int] = {}
        # 检测轮次计数：每跑一次推理算一轮（低功耗模式下 4 帧一轮）。驻留判据用的是
        # 它，而不是墙钟秒数 —— 见 PRESENCE_GRACE_CYCLES。
        self._detection_cycles = 0
        self._processed_frames = 0

    # 轨迹短暂消失的容忍时间（秒）。半身被遮挡的人在低功耗预设下检测会断断续续，
    # 一帧没检到就关闭会话的话，下一个会话会重新计时，停留时间永远涨不到阈值 ——
    # 表现就是"人一直在区域里，却一条报警都没有"。取值参考下面 lost_track_buffer
    # 对应的时长（60 帧 ≈ 2 秒），略小一点，让"轨迹还活着"与"会话还留着"基本同步。
    SESSION_GAP_TOLERANCE_SECONDS = 1.5
    # 检测置信度下限。ultralytics 默认 0.25，而半身被遮挡的人置信度常掉到 0.1~0.2 ——
    # 那些检测在默认阈值下会被直接丢掉，ByteTrack 也就没有机会用它维持住已有轨迹
    # （它的第二段关联本来就是专门吃低分检测来扛遮挡的）。放低到这里不会凭空多出新
    # 轨迹：新建轨迹另有 NEW_TRACK_CONFIDENCE 这道闸门，低分检测只能延续已有轨迹。
    DETECTION_CONFIDENCE = 0.1
    # 新建一条轨迹所需的置信度（tracker 的 high_conf_det_threshold）。
    #
    # 这是「画面里出现一个没有人形的框」的第一道闸门。检出本身是模型给的结果，而追踪器
    # 只给**够自信的检出**发 ID；原来的 0.6 太松 —— 栏杆、橱窗模特、广告牌、反光这些
    # 东西被认成人时经常正好落在 0.6~0.7，一旦拿到 ID，这个框就会一直画下去（每帧都被
    # 检出，或由预测维持）。这类弱误检仍然参与关联（能维持已有人物轨迹），但不再自己
    # 开一条新轨迹。
    #
    # 0.65 是实测选的，不是拍的。在 8月11日-1.mp4 的 90 帧（标准与低功耗两条路径）上
    # 扫过 0.60/0.65/0.70：
    #
    #     门槛  标准: 轨迹/强检出没跟上   低功耗: 轨迹/强检出没跟上
    #     0.60      12 条 /  7 个            12 条 / 4 个
    #     0.65      12 条 /  9 个            10 条 / 5 个
    #     0.70      11 条 / 11 个             9 条 / 9 个
    #
    # 0.70 挡掉的垃圾比 0.65 只多一点点（标准 475 vs 464 个弱检出没人跟），代价却是
    # 「真人被检出却没跟上」翻倍 —— 低功耗预设的 INT8 模型给人打的分本来就偏低
    # （真人常见 0.6~0.7），0.70 会把一部分真人挡在门外。换现场素材后这个数可以重扫。
    NEW_TRACK_CONFIDENCE = 0.65
    # 轨迹与检出的关联 IoU 门槛。原来 0.1 太松：真人走开之后，原地或附近的弱检出
    # （0.1~0.69 都只能用于关联）很容易接管他的轨迹 ID，于是框留在原地、而那里没有
    # 任何东西。0.25 仍然容得下相邻帧的正常位移（低功耗模式下预测框是钉住的，人走动
    # 一个检测周期后框会偏一点，实测 IoU 远高于它）。
    TRACK_IOU_THRESHOLD = 0.25
    # 「这个目标现在真的在那里」的判据：检测**连续漏掉几轮**之后，框就只算预测 ——
    # 不推进驻留、也不报警。
    #
    # 用「检测轮次」而不是墙钟秒数：低功耗模式每 4 帧才检测一次，标准模式每帧都检测，
    # 同一个秒数在两种模式下的含义完全不同（1 秒 = 检测 7 次，或检测 30 次），而
    # 「检测器看了一轮、没看见它」在哪种模式下的含义都一样。
    #
    # 挡的是「一次误检靠预测框撑满驻留阈值」：驻留时钟按「距首次进入的时间」算，而
    # 预测帧同样计入，于是低功耗下每 4 帧有 3 帧都在替一次误检说话。挡掉之后，框停留
    # 在原地、而检测器连着两轮都没看见它的那段时间里，不会产生报警。真在场的人每个
    # 检测轮次都会被检出，不受影响；被遮挡的人最多只是晚一个检测周期报警。
    PRESENCE_GRACE_CYCLES = 1
    # 画框时区分强弱检出的分界：低于它的框画细线、暗色，并且标签里带上置信度 ——
    # 让操作员一眼能分清「模型很确定」和「模型在猜」。0.4 以下基本是噪声区。
    WEAK_DETECTION_CONFIDENCE = 0.4
    # 轨迹缓存（置信度、最后一次被匹配的位置）保留多少个**检测轮次**。
    # 比追踪器的丢失预算宽得多（60 帧 ≈ 2 秒，低功耗下才 15 轮），所以只用来兜住内存，
    # 不会把还活着的轨迹清掉。
    ANCHOR_TTL_CYCLES = 90
    # 「只检测警戒区」时，裁片面积超过整帧这个比例就不裁了 —— 省不了多少，反而丢掉区外
    # 的框。多个区域分散在画面两角时就会走到这一步。
    DETECT_REGION_MAX_AREA_RATIO = 0.75

    @staticmethod
    def _new_tracker() -> ByteTrackTracker:
        return ByteTrackTracker(
            track_activation_threshold=0.25,
            # 60 帧 ≈ 2 秒（按 30fps 折算）。要 ≥ SESSION_GAP_TOLERANCE_SECONDS，
            # 否则轨迹先被追踪器丢掉、人再出现时拿到新 ID，会话照样会重新计时 ——
            # 那样"会话容忍"就形同虚设了。
            #
            # 注意追踪器实际用的是**时间预算**（lost_track_buffer / 30 秒）：低功耗模式
            # 下 update 只每 4 帧调一次，帧数预算会被拉长成 8 秒，时间预算才是这 2 秒。
            lost_track_buffer=60,
            frame_rate=30,
            # 0 = 新轨迹一建立就算"成熟"，不会因为下一帧没匹配上就被剪掉。
            #
            # 必须这么设，因为低功耗模式下跳过的帧现在也会调一次 update（见 process：
            # 推进滤波器，让预测框跟着人走）。新轨迹是在某个检测帧上建出来的，紧接着
            # 的 3 个跳过帧里它必然"没有匹配"——按最小连续帧数 1 的规则，它当场就被
            # 剪掉，于是整条轨迹永远立不起来（实测：所有目标都不再有框）。
            # 拿到 ID 的时机没有变化：ID 仍然在它**下一次被匹配上**时才分配。
            minimum_consecutive_frames=0,
            minimum_iou_threshold=DetectionEngine.TRACK_IOU_THRESHOLD,
            high_conf_det_threshold=DetectionEngine.NEW_TRACK_CONFIDENCE,
        )

    def update_zones(self, zones: list[ZoneDefinition]) -> None:
        self.zones = [ZoneDefinition.from_dict(zone.to_dict()) for zone in zones]
        # 区域是运行中可改的：用户画完新区域，检测范围要跟着变（见 _detect_crop）。
        self._zone_bounds = self._zone_union(self.zones)

    def reset_tracking(self) -> None:
        self.tracker = self._new_tracker()
        self.active_sessions.clear()
        # 冷却记录也要清：循环播放重头开始时 video_time 会跳回 0，留着旧时刻会让
        # 差值变成负数，把报警一直压住直到播放位置追上来。
        self._last_alarm_at.clear()
        self._last_confidence.clear()
        self._detection_cycles = 0
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
            self._detection_cycles += 1
            inference_kwargs: dict[str, Any] = {
                "classes": [0],
                "verbose": False,
                "device": self.device,
                "conf": self.DETECTION_CONFIDENCE,
            }
            if self.policy.imgsz is not None:
                inference_kwargs["imgsz"] = self.policy.imgsz
            crop = self._detect_crop(frame)
            # 注意别把这个变量叫 source：source 是本函数的参数（视频源标识），
            # 事件记录要用它。曾经因为这里遮住了它，报警记录里的 source 变成了一张
            # 1080p 图像数组 —— json.dumps 直接抛异常，检测线程当场停掉。
            model_input = frame if crop is None else frame[crop[1] : crop[3], crop[0] : crop[2]]
            result = self.model(model_input, **inference_kwargs)[0]
            raw_detections = sv.Detections.from_ultralytics(result)
            if crop is not None:
                raw_detections = self._to_frame_coordinates(raw_detections, crop)
            detections = (
                self.tracker.update(raw_detections, timestamp=video_time)
                if self.policy.uses_tracker_prediction
                else self.tracker.update(raw_detections)
            )
        else:
            # 跳过的帧也要把追踪器的滤波器推进一格：喂一个空检测，让卡尔曼按真实时间
            # 步往前走。不推进的话框会**冻在**上一次检测的位置（滤波器压根没动），人却
            # 还在走 —— 低功耗模式下看着就是"框跟不上人、一跳一跳"。推进之后预测框按
            # 估计速度继续移动，下一轮检测时它与真实框的重叠大得多，轨迹因此不容易断
            # （断一次就是一个新 ID，而新 ID 会让驻留重新计时）。
            self.tracker.update(
                sv.Detections.empty(), timestamp=video_time
            )
            detections = self._stabilise_predicted_boxes(self.tracker.tracked_objects)
        track_ids = self._track_ids(detections)
        confidences = self._frame_confidences(detections)
        in_zone_ids: dict[str, set[Hashable]] = {zone.name: set() for zone in self.zones}
        in_zone_elapsed_seconds: dict[Hashable, float] = {}
        transitions: list[SessionTransition] = []

        for index, track_id in enumerate(track_ids):
            if track_id is None or index >= len(detections.xyxy):
                continue
            confidence = confidences[index] if index < len(confidences) else None
            x1, y1, x2, y2 = detections.xyxy[index]
            anchor = ((float(x1) + float(x2)) / 2, float(y2))
            for zone in self.zones:
                if not zone.closed or len(zone.polygon) < 3 or not self._contains(zone, anchor):
                    continue
                in_zone_ids[zone.name].add(track_id)
                key = (zone.name, track_id)
                session = self.active_sessions.get(key)
                if session is None:
                    if not detect_this_frame:
                        # 这一帧没有真检出（低功耗模式下画的是预测框）：不能凭一个预测
                        # 出来的位置就宣布「有人进入了区域」。等下一帧真检出再说，而
                        # 低功耗下一个检测周期只有 4 帧，晚不了多久。
                        continue
                    # 进入那一刻：事件就是在这里建出来的，所以 wall_time（记录时间）与
                    # entered_wall_time（进入时间）指的是同一个时刻，两个都记下来。
                    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    event = AlarmEvent(
                        source=source,
                        operation_mode=operation_mode,
                        zone_name=zone.name,
                        track_id=str(track_id),
                        entered_at_seconds=video_time,
                        alarm_at_seconds=None,
                        wall_time=started_at,
                        entered_wall_time=started_at,
                    )
                    session = ActiveIntrusion(event, video_time)
                    self.active_sessions[key] = session
                    transitions.append(SessionTransition("entered", event))
                session.last_seen_at_seconds = video_time
                if detect_this_frame:
                    session.last_detected_cycle = self._detection_cycles
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
                    #
                    # 驻留够了还要求「检测器最近真的看见过它」：框停留在原地不等于人还在
                    # 那里（见 PRESENCE_GRACE_CYCLES）。真在场的人每个检测轮次都会被检出，
                    # 所以这条只在「目标已经不在、框由预测维持」时才起作用。
                    if (
                        self._presence_is_current(session)
                        and self._cooldown_elapsed(key, zone, video_time)
                    ):
                        session.event.alarm_at_seconds = video_time
                        session.event.status = "alarmed"
                        session.event.alarm_confidence = (
                            confidence if confidence is not None
                            else self._last_confidence.get(track_id)
                        )
                        self._last_alarm_at[key] = video_time
                        transitions.append(SessionTransition("alarmed", session.event))
            if confidence is not None:
                self._last_confidence[track_id] = confidence
            if detect_this_frame:
                # 这一帧的框是**匹配上的真实观测**，记下位置与本轮次编号：跳过的帧上要用
                # 它判断这条轨迹是不是「刚看见过」，见 _stabilise_predicted_boxes。
                self._last_matched_box[track_id] = np.asarray(
                    (x1, y1, x2, y2), dtype=np.float32
                )
                self._last_matched_cycle[track_id] = self._detection_cycles

        active_keys = {
            (zone_name, track_id)
            for zone_name, ids in in_zone_ids.items()
            for track_id in ids
        }
        for key in list(self.active_sessions):
            if key in active_keys:
                continue
            session = self.active_sessions[key]
            # 短暂丢失先留着：一帧没检到就关会话的话，下一个会话会重新计时，停留时间
            # 永远涨不到阈值。留着的这段时间**不累加驻留**（上面只按看到的帧累加），
            # 所以不会给一个已经离开的人补一次报警。
            if video_time - session.last_seen_at_seconds <= self.SESSION_GAP_TOLERANCE_SECONDS:
                continue
            transitions.append(self._close_session(key, video_time))

        self._prune_alarm_history(video_time)
        self._forget_stale_tracks()

        if render:
            # 预测帧（tracked_objects）不带分数，用该轨迹最后一次真实检出的分数顶上：
            # 否则低功耗模式下 3/4 的帧会显示一堆没有数字的框，看起来像"全都不识别了"。
            display_confidences = [
                value if value is not None else self._last_confidence.get(track_id)
                for track_id, value in zip(track_ids, confidences)
            ]
            annotated = self._annotate(
                frame,
                detections,
                in_zone_ids,
                in_zone_elapsed_seconds,
                confidences=display_confidences,
                # 「闯入」= 这个目标的会话已经真的报过警（响过铃、写过记录）。这是操作员
                # 最需要一眼看出的分界，所以由引擎从会话状态里取，而不是让标注去猜。
                alarmed_ids={
                    track_id
                    for (_, track_id), session in self.active_sessions.items()
                    if session.event.alarmed
                },
            )
        else:
            annotated = frame
        return annotated, transitions

    @staticmethod
    def _zone_union(zones: list[ZoneDefinition]) -> tuple[float, float, float, float] | None:
        """所有**闭合**区域的并集包围盒；一个都没有时返回 None。"""
        polygons = [
            np.asarray(zone.polygon, dtype=np.float32)
            for zone in zones
            if zone.closed and len(zone.polygon) >= 3
        ]
        if not polygons:
            return None
        points = np.concatenate(polygons)
        return (
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        )

    def _detect_crop(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """这一帧要送进网络的裁片（区域外扩）；None = 送整帧。

        只在策略要求「只检测警戒区」时才裁（见 ``InferencePolicy.detect_region_margin``）。
        退化成整帧的三种情况：策略没开、还没有闭合区域（用户正在画，这时整帧检测让
        他看得见框）、区域并集已经占了整帧的大半（裁了也不省）。
        """
        margin = self.policy.detect_region_margin
        if margin is None or self._zone_bounds is None:
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self._zone_bounds
        pad_x, pad_y = (x2 - x1) * margin, (y2 - y1) * margin
        crop = (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(width, int(x2 + pad_x)),
            min(height, int(y2 + pad_y)),
        )
        crop_width, crop_height = crop[2] - crop[0], crop[3] - crop[1]
        if crop_width < 32 or crop_height < 32:
            return None
        if crop_width * crop_height > width * height * self.DETECT_REGION_MAX_AREA_RATIO:
            return None
        return crop

    @staticmethod
    def _to_frame_coordinates(
        detections: sv.Detections, crop: tuple[int, int, int, int]
    ) -> sv.Detections:
        """把裁片上的框换算回整帧坐标。

        区域判定、追踪、标注、取证截图全都按整帧坐标走，所以换算必须发生在
        **送进追踪器之前**（在裁片坐标里追踪，框会整体偏移一个裁片原点）。
        """
        offset = np.asarray([crop[0], crop[1], crop[0], crop[1]], dtype=np.float32)
        shifted = np.asarray(detections.xyxy, dtype=np.float32) + offset
        return sv.Detections(
            xyxy=shifted,
            confidence=detections.confidence,
            class_id=detections.class_id,
        )

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
    def _frame_confidences(detections: sv.Detections) -> list[float | None]:
        """这一帧每个框的检测置信度；没有的那一项是 None。

        预测帧（``tracked_objects``）**不带置信度** —— 那是卡尔曼预测出来的框，没有
        对应的检出分数。所以这里逐项取 None，而不是假装有个数。

        保留 3 位小数：模型给的是 float32，``0.83`` 落到这里会变成 0.8299999833…，
        直接写进报警记录既难看又不好比较。
        """
        values = getattr(detections, "confidence", None)
        if values is None:
            return [None] * len(detections)
        return [None if value is None else round(float(value), 3) for value in values]

    def _presence_is_current(self, session: ActiveIntrusion) -> bool:
        """这个会话代表的目标，检测器最近是否真的看见过。

        用来挡住「框还在原地、人早就不在了」的那种报警：预测框能把驻留时钟一路推过
        阈值（低功耗模式下每 4 帧只检测一次，其余 3 帧全靠预测），所以驻留够了之后
        还必须有一次不那么久远的真实检出。容忍 ``PRESENCE_GRACE_CYCLES`` 个被漏掉的
        检测轮次 —— 半身被遮挡的人本来就常常一两轮检不出来。
        """
        return (
            self._detection_cycles - session.last_detected_cycle
            <= self.PRESENCE_GRACE_CYCLES
        )

    def _stabilise_predicted_boxes(self, detections: sv.Detections) -> sv.Detections:
        """把**已经跟丢**的轨迹钉在它最后一次被看见的位置上。

        跳过的帧上卡尔曼会按估计速度继续外推，所以目标走出画面之后，框还会以同样的
        速度往前滑 —— 实测 225px/秒、最多滑出 375px，观感是「空地上有个框自己在走」。
        这与我们要的东西只差一步：**两次真实观测之间**的插值正是「框跟着人走」，而
        观测已经断掉之后的外推就只是幻觉。

        判据是「这条轨迹在最近一次检测里还匹配上了吗」：是 → 用预测位置（插值）；
        否 → 用最后一次真实观测的位置。判据用检测轮次编号比较，所以与帧率无关。
        """
        track_ids = self._track_ids(detections)
        boxes = np.asarray(detections.xyxy, dtype=np.float32).copy()
        frozen_any = False
        for index, track_id in enumerate(track_ids):
            if track_id is None or index >= len(boxes):
                continue
            if self._last_matched_cycle.get(track_id) == self._detection_cycles:
                continue  # 最近一轮还看见过：预测位置是可信的插值
            frozen = self._last_matched_box.get(track_id)
            if frozen is not None:
                boxes[index] = frozen
                frozen_any = True
        if not frozen_any:
            return detections
        return sv.Detections(
            xyxy=boxes,
            tracker_id=np.asarray(
                [-1 if track_id is None else int(track_id) for track_id in track_ids],
                dtype=int,
            ),
        )

    def _forget_stale_tracks(self) -> None:
        """丢掉很久没被匹配上的轨迹的缓存（置信度、最后一次位置）。

        ByteTrack 的 ID 只增不减，不清理这些 dict 会一直长。但**不能按「这一帧还在不在」
        清**：检测帧上没匹配上的那些轨迹，恰恰是跳过的帧要用锚点钉住位置的 —— 一帧不在
        就删掉，跟丢的预测框当场就会开始外推。所以按「轮次年龄」清，上限取得比追踪器的
        丢失预算（``lost_track_buffer`` 折算约 2 秒）宽得多，活着/刚丢的轨迹一定还在。
        """
        for track_id, cycle in list(self._last_matched_cycle.items()):
            if self._detection_cycles - cycle > self.ANCHOR_TTL_CYCLES:
                del self._last_matched_cycle[track_id]
                self._last_matched_box.pop(track_id, None)
                self._last_confidence.pop(track_id, None)

    @staticmethod
    def _format_dwell(seconds: float) -> str:
        """区域内已经待了多久：``1.5秒``。

        只到 0.1 秒。报警记录里的时长是三位小数 —— 那是存档，要能跟取证截图对上；而
        画在框上的数字是给人瞄一眼的，小数点后三位在活动画面上根本来不及看，只是让
        标签更长、更容易和别人的撞上。
        """
        return f"{max(0.0, seconds):.1f}秒"

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
        *,
        confidences: list[float | None] | None = None,
        alarmed_ids: set[Hashable] | None = None,
    ) -> np.ndarray:
        """按「地面 → 行人 → 信息」的层次画这一帧。

        顺序不是随意的，它决定了观感：

        1. **警戒区先落在地上**（贴地渲染：半透明填充 + 近宽远窄的边带）；
        2. **再把行人的像素贴回区域之上** —— 这样边界看起来在人后面，而不是浮在人身上；
        3. **最后才是检测框与标签**，信息层永远在最上面。

        原来是把区域用一条不透明实线画在最后，于是它横穿人身上、又硬又平，看着像贴
        在屏幕上的一条红线。

        框只有两种画法，由**置信度**决定：实线 + 数字是模型比较确定的，细线暗色 +
        低分是模型在猜。``confidences`` 可以传一份「已经补过最后一次真实检出分数」的
        列表（见 ``process``）—— 低功耗模式下每 4 帧里只有 1 帧真跑推理，其余 3 帧
        画的是卡尔曼预测框、本身不带分数，如果不补，屏幕上就会有 3/4 的时间在显示
        一堆没有数字的框，看上去像"全都不识别了"。

        也正因如此，这里**不再按帧区分「预测 / 检出」的画法**：那种样式会在 4 帧里
        闪 3 次，而它想表达的信息（这一帧是否真检出）对看画面的人没什么用 —— 判断
        真假靠的是分数高低，判断该不该报警的活儿在会话逻辑里。

        标签是**先全部排好版、再一起画**的（``_layout_labels``）：一行字挨着一行字时，
        按框的顺序各自往自己框上一贴，几行字就会叠成一团黑。先算位置才能让它们互相
        让开，而只有先知道所有标签的尺寸才谈得上让位。

        标签写什么由 ``show_detection_details`` 决定（见 ``_box_label``）：默认只写
        给人看的那三种状态，打开开关才带上目标编号与置信度。
        """
        annotated = frame.copy()
        track_ids = self._track_ids(detections)
        if confidences is None:
            confidences = self._frame_confidences(detections)
        boxes = [
            tuple(int(value) for value in box) for box in detections.xyxy
        ]

        self._draw_ground_zones(annotated, in_zone_ids)
        self._restore_people(annotated, frame, boxes)

        # 区域名先画、并占住自己的位置：它是画面里固定的地标，而框标签是每帧都在动的
        # 东西 —— 让动的躲开静的，反过来做的话名字会被来回路过的人反复盖住。
        occupied: list[tuple[int, int, int, int]] = []
        for zone in self.zones:
            if len(zone.polygon) >= 2:
                # 区域名属于信息层，画在遮挡之后 —— 画在前面的话，路过的人会把名字
                # 擦掉一块，看着像渲染出错。
                rect = self._draw_zone_name(
                    annotated, zone, np.asarray(zone.polygon, dtype=np.float32)
                )
                if rect is not None:
                    occupied.append(rect)

        scale = self._label_scale(annotated.shape[0])
        labels: list[
            tuple[tuple[int, int, int, int], str, tuple[int, int], tuple[int, int, int]]
        ] = []
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            track_id = track_ids[index] if index < len(track_ids) else None
            confidence = confidences[index] if index < len(confidences) else None
            inside = any(track_id in ids for ids in in_zone_ids.values())
            color = (0, 0, 255) if inside else (0, 200, 0)
            weak = (
                confidence is not None
                and confidence < self.WEAK_DETECTION_CONFIDENCE
            )
            if weak:
                # 暗一档、细一档：弱检出不该和确定的检出一样抢眼。
                color = tuple(int(channel * 0.7) for channel in color)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            else:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            text = self._box_label(
                track_id,
                confidence,
                in_zone_elapsed_seconds.get(track_id) if inside else None,
                alarmed=track_id in (alarmed_ids or ()),
                show_details=self.show_detection_details,
            )
            labels.append(((x1, y1, x2, y2), text, self._label_plate_size(text, scale), color))

        # 大的框（离相机近的人）先选位置：一个近处的人被挤掉标签，比远处的人被挤更难看。
        labels.sort(
            key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]),
            reverse=True,
        )
        placements = self._layout_labels(
            [(box, size) for box, _, size, _ in labels],
            (annotated.shape[1], annotated.shape[0]),
            occupied,
        )
        for (_, text, _, color), rect in zip(labels, placements):
            self._draw_label(annotated, text, rect, color, scale)
        return annotated

    @staticmethod
    def _box_label(
        track_id: Hashable | None,
        confidence: float | None,
        in_zone_seconds: float | None,
        *,
        alarmed: bool = False,
        show_details: bool = False,
    ) -> str:
        """框上那行字。默认是给**看画面的人**看的：``行人``／``目标9 区域内 1.5秒``／``目标9 闯入 2.0秒``。

        看画面的人要判断的只有三件事 —— 是不是人、进没进警戒区、待了多久（该不该
        抬头看一眼）。置信度回答不了其中任何一个，所以默认不显示。

        三种状态的差别就是操作员要不要管：

        * ``行人``：在区域外，只是画面里的一个人；
        * ``目标9 区域内 1.5秒``：进了警戒区、还在计时，没到停留阈值（还没响过铃）；
        * ``目标9 闯入 2.0秒``：已经越过阈值、**真的报过警**（响铃 + 写记录 + 存取证截图）。

        **进了区域就带上编号**：报警记录里写的是「目标 9」，取证截图上如果同时有几个人
        在区域内，只有编号能说明哪一行字对应记录里的那个目标 —— 没有编号的框（追踪器
        还没认下来的）本来就不会被判进区域，所以区域内的编号一定拿得到。

        ``show_details``（设置页里的「显示目标编号与置信度」）打开后回到工程视角：
        ``目标9 0.86 区内1.5秒``。排查「为什么老误报」时要的就是这几个数 —— 编号能
        和报警记录对上，而置信度直接说明模型有多确定（低于 0.4 的框还会画得又细又暗）。
        """
        if not show_details:
            if in_zone_seconds is None or track_id is None:
                return "行人"
            state = "闯入" if alarmed else "区域内"
            return f"目标{track_id} {state} {DetectionEngine._format_dwell(in_zone_seconds)}"
        label = f"目标{track_id}" if track_id is not None else "人物"
        if confidence is not None:
            label += f" {confidence:.2f}"
        if in_zone_seconds is not None:
            label += f" 区内{DetectionEngine._format_dwell(in_zone_seconds)}"
        return label

    # 标签的画法。这几条都是为了"一眼看清"，不是装饰：
    #
    # * **中文靠 cv2 内置的 Unicode 字体**（opencv 5.0 的 `Built-in Unicode font: YES`）。
    #   原来的 Hershey 字体只有 ASCII，所以标签只能是英文；换成中文不需要字体文件、
    #   PIL 或新依赖。**降级 opencv 之前先确认这一条**（降下去中文会变成乱码方块），
    #   tests/test_label_overlay.py 里有测试钉住它。
    # * **字号跟着画面高度走**：标注是烧进原始分辨率的帧里的，1080p 上 0.55 的话，画面
    #   一缩小到窗口里这行字就只有几个像素高。按帧高算，换个分辨率看到的字一样大。
    # * **文字垫一块深色底**：标签会落在沥青、地砖、人身上，只靠字的颜色总有一半场合
    #   读不出来（区域名用黑描边解决同一件事，这里用的是垫底）。
    # * **相互不重叠**：见 _layout_labels。
    LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
    LABEL_THICKNESS = 2
    LABEL_PADDING_X = 5
    LABEL_PADDING_Y = 3
    # 标签与它所属的框之间的间距，也是让位时一步的高度增量。
    LABEL_GAP = 4
    # 两个标签之间至少留出的空隙。
    LABEL_MARGIN = 2
    # 上下让位最多走几步；再多说明这块地方实在挤不下，接受重叠比让标签飞远更可读。
    LABEL_MAX_STEPS = 8
    LABEL_PLATE_ALPHA = 0.62
    LABEL_MIN_SCALE = 0.55
    LABEL_MAX_SCALE = 1.0
    # 字号 = 帧高 / 这个数（夹在 MIN/MAX 之间）。1080p 上是 0.9，720p 上是 0.6。
    LABEL_SCALE_REFERENCE_HEIGHT = 1200
    # 区域名比框标签小一点、也固定不缩放：它是画面里的地标，不该跟人争注意力。
    ZONE_NAME_SCALE = 0.65
    # 区域名在顶点上方留出的距离。
    ZONE_NAME_GAP = 8

    @classmethod
    def _label_scale(cls, frame_height: int) -> float:
        scale = frame_height / cls.LABEL_SCALE_REFERENCE_HEIGHT
        return min(cls.LABEL_MAX_SCALE, max(cls.LABEL_MIN_SCALE, scale))

    @classmethod
    def _label_plate_size(cls, text: str, scale: float) -> tuple[int, int]:
        """标签垫底块的尺寸（已经含内边距）。

        文字画在 ``(x + LABEL_PADDING_X, y + LABEL_PADDING_Y + 字身高)`` —— cv2 的
        y 是**基线**，不是上边，所以基线要往下让出字身高，否则字会飘到垫底块外面。
        """
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cls.LABEL_FONT, scale, cls.LABEL_THICKNESS
        )
        return (
            text_width + 2 * cls.LABEL_PADDING_X,
            text_height + baseline + 2 * cls.LABEL_PADDING_Y,
        )

    @staticmethod
    def _rects_overlap(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
        margin: int = 0,
    ) -> bool:
        """两个矩形是否相交（``margin`` 是要求的最小间隔）。"""
        first_x, first_y, first_width, first_height = first
        second_x, second_y, second_width, second_height = second
        return (
            first_x - margin < second_x + second_width
            and second_x - margin < first_x + first_width
            and first_y - margin < second_y + second_height
            and second_y - margin < first_y + first_height
        )

    @classmethod
    def _collisions(
        cls,
        rect: tuple[int, int, int, int],
        taken: list[tuple[int, int, int, int]],
    ) -> int:
        """这块地方压住了几个已经放好的标签（0 = 完全空着）。"""
        return sum(1 for other in taken if cls._rects_overlap(rect, other, cls.LABEL_MARGIN))

    @classmethod
    def _layout_labels(
        cls,
        entries: list[tuple[tuple[int, int, int, int], tuple[int, int]]],
        frame_size: tuple[int, int],
        occupied: list[tuple[int, int, int, int]] | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """给每个标签找一块地方，返回各自的垫底矩形（左上角 + 宽高），顺序与入参一致。

        ``entries`` 是 ``(框, 标签尺寸)``；``occupied`` 是**先到先得**的既有矩形（区域名
        就是按这个先占住的 —— 名字被框标签盖住一次，那个区域在哪就说不清了）。

        位置优先「框正上方」（那里最空），其次框内上沿、框内下沿、框正下方；四处都撞上
        别人时，就在框上方那条线上**上下交替让位**，一步一个行高。

        让位只动纵向、横向一律不挪：标签横向离开自己的框以后，看的人就得猜这行字是谁
        的，那是比重叠更糟的事。最坏情况（画面里挤了太多人）会放弃让位、接受重叠 ——
        信息少一行也比让某个人完全没有标签强，但即便如此也要挑**压到别人最少**的那个
        位置，而不是所有人叠在同一处。
        """
        frame_width, frame_height = frame_size
        taken = list(occupied or [])
        placements: list[tuple[int, int, int, int]] = []

        for box, (label_width, label_height) in entries:
            x1, y1, _, y2 = box
            x = min(max(0, x1), max(0, frame_width - label_width))
            step = label_height + cls.LABEL_GAP

            def clamp_y(value: int) -> int:
                return min(max(0, value), max(0, frame_height - label_height))

            base = clamp_y(y1 - cls.LABEL_GAP - label_height)
            wanted = [
                base,
                clamp_y(y1 + cls.LABEL_GAP),
                clamp_y(y2 - label_height - cls.LABEL_GAP),
                clamp_y(y2 + cls.LABEL_GAP),
            ]
            # 让位：从首选位置上下交替走，越走越远。
            for hop in range(1, cls.LABEL_MAX_STEPS + 1):
                wanted.append(clamp_y(base - hop * step))
                wanted.append(clamp_y(base + hop * step))
            candidates: list[int] = []
            for y in wanted:
                if y not in candidates:
                    candidates.append(y)

            def rect_at(y: int) -> tuple[int, int, int, int]:
                return (x, y, label_width, label_height)

            placed: int | None = None
            for y in candidates:
                if cls._collisions(rect_at(y), taken) == 0:
                    placed = y
                    break
            if placed is None:
                # 一个空位都没有：挑压得最少的，同样少时挑离首选位置最近的 ——
                # 重叠已经不可避免，至少别让所有标签叠在同一处。
                placed = min(
                    candidates,
                    key=lambda y: (cls._collisions(rect_at(y), taken), abs(y - base)),
                )
            rect = rect_at(placed)
            taken.append(rect)
            placements.append(rect)
        return placements

    @classmethod
    def _draw_label(
        cls,
        annotated: np.ndarray,
        text: str,
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int],
        scale: float,
    ) -> None:
        """画一块垫底 + 一行标签。rect 由 _layout_labels 定好，这里只管画。"""
        x, y, width, height = rect
        region = annotated[y : y + height, x : x + width]
        if region.size:
            plate = np.zeros_like(region)
            cv2.addWeighted(
                plate, cls.LABEL_PLATE_ALPHA, region, 1.0 - cls.LABEL_PLATE_ALPHA, 0, region
            )
        (_, text_height), _ = cv2.getTextSize(
            text, cls.LABEL_FONT, scale, cls.LABEL_THICKNESS
        )
        cv2.putText(
            annotated,
            text,
            (x + cls.LABEL_PADDING_X, y + cls.LABEL_PADDING_Y + text_height),
            cls.LABEL_FONT,
            scale,
            color,
            cls.LABEL_THICKNESS,
            cv2.LINE_AA,
        )

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
    ) -> tuple[int, int, int, int] | None:
        """画区域名，并返回它占住的矩形（框标签据此让位）。

        名字用的是和框标签同一套画法（深色垫底 + 区域色文字），不是原来那种「黑描边 +
        彩色字」：名字落在自己区域的贴地面和边带上 —— 红色区域里一个红描黑边的名字，
        实地看就是一团红，而垫底在任何背景上都一样清楚。
        """
        if not zone.name:
            return None
        height, width = annotated.shape[:2]
        (text_width, text_height), baseline = cv2.getTextSize(
            zone.name, self.LABEL_FONT, self.ZONE_NAME_SCALE, self.LABEL_THICKNESS
        )
        plate_width = text_width + 2 * self.LABEL_PADDING_X
        plate_height = text_height + baseline + 2 * self.LABEL_PADDING_Y
        # 名字挂在区域的第一个顶点上（略高一点）。顶点可能在画面外（区域允许拖出边缘），
        # 所以整块垫底要夹回画面内，否则整行字都看不见。
        x = min(max(0, int(polygon[0][0])), max(0, width - plate_width))
        y = min(
            max(0, int(polygon[0][1]) - self.ZONE_NAME_GAP - text_height - self.LABEL_PADDING_Y),
            max(0, height - plate_height),
        )
        rect = (x, y, plate_width, plate_height)
        self._draw_label(annotated, zone.name, rect, self._bgr(zone.color), self.ZONE_NAME_SCALE)
        return rect

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
