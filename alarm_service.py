from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

import app_paths
from models import AlarmEvent

logger = logging.getLogger(__name__)

# 倒读时用来识别「这一行开启了一条新会话」的结构标记，见 EventStore._starts_session。
_OPENED_MARKER = re.compile(r'"action"\s*:\s*"opened"')
_SCHEMA_MARKER = re.compile(r'"schema_version"\s*:')
_LEGACY_MARKER = re.compile(r'"entered_at_seconds"\s*:')
# 取证截图文件名里的落盘钟点：20260921-143610_<会话ID>_alarm.jpg
#
# 只认这一种命名。更早的命名是 2026-08-12_11-20-07_区域_1_id0.jpg，那里面的时间戳**等于
# 会话开始那一刻**、不是报警那一刻（拿现有 1755 条记录对过，全部差 0 秒），所以它不能
# 用来还原报警钟点 —— 认它反而会把进入时刻说成报警时刻。
_SCREENSHOT_STAMP = re.compile(r"(?<!\d)(\d{8})-(\d{6})_")


def _clock_now() -> str:
    """落盘那一刻的真实钟点，格式与记录里的 wall_time 一致。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class AlarmPlayer:
    def __init__(self, warning_audio_path: str | Path | None = None) -> None:
        self.warning_audio_path = (
            Path(warning_audio_path)
            if warning_audio_path is not None
            # 分组式发布布局优先，扁平式（资源直接放 exe 同级）作为兼容回退。
            else app_paths.resource(
                "assets/warming_converted.wav", "warming_converted.wav"
            )
        )
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

            winsound.PlaySound(
                str(self.warning_audio_path),
                winsound.SND_FILENAME | winsound.SND_NODEFAULT,
            )
        except (ImportError, OSError, RuntimeError) as error:
            logger.warning("Unable to play alarm: %s", error)
        finally:
            with self._lock:
                self._is_playing = False


class EventStore:
    # 从尾部倒读时的块大小。一次会话在 JSONL 里大约 600 字节，200 条也就 120 KB，
    # 两三次读取就够；块再大就浪费在「读了却不看」的字节上。
    READ_CHUNK_BYTES = 64 * 1024

    @staticmethod
    def unavailable_reason(event: AlarmEvent, path: str) -> str:
        """截图显示不出来时，给用户一句能看懂的解释。

        原来一律显示"不可用"，而三种完全不同的情况（本来就没报警、记录里没有路径、文件
        被清理或删掉了）看起来一模一样 —— 于是"记录里经常出现报警取证不可用"成了一个
        说不清的现象，也没法判断要不要处理。
        """
        if not event.alarmed:
            return "未触发报警（本就没有报警截图）"
        if not path:
            return "截图未写入（记录里没有路径，请看 logs\\app.log 里的取证截图告警）"
        return "截图文件缺失（可能已被留存策略清理，或被人手动删除）"

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
            # 三个时刻各自的真实钟点（视频模式下 *_at_seconds 只是视频里的位置）。
            "entered_wall_time": event.entered_wall_time,
            "alarmed_wall_time": event.alarmed_wall_time,
            "exited_wall_time": event.exited_wall_time,
            # 报警那一刻的检测置信度（误报排查用）。
            "alarm_confidence": event.alarm_confidence,
        }

    def _append(self, payload: dict[str, object]) -> None:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as event_file:
                event_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _save_screenshot(self, event: AlarmEvent, frame: np.ndarray, kind: str) -> str:
        """保存一张取证截图，返回**相对 events 根目录**的路径（失败返回空串）。

        两处都是踩过坑才这么写的：

        * 用 ``imencode`` + ``write_bytes`` 而不是 ``cv2.imwrite``。**cv2.imwrite 在非
          ASCII 路径下会静默失败**（返回 False，文件根本没写），而发布包的目录名是中文
          （行人警戒区域监控），于是打包版里每一次取证截图都写不进去 —— 记录写着"已报警"、
          截图路径却是空的，界面上显示成"报警取证不可用"。开发目录是纯 ASCII，所以这个
          问题在开发机上一直没暴露。
        * 存相对路径。绿色版的说明书写着"整个文件夹要一起拷贝"，而绝对路径一旦被拷到
          别处就全部失效 —— 相对路径让记录跟着 events 目录走。
        """
        if frame.size == 0 or frame.ndim != 3:
            # 空帧（比如会话收尾时用占位帧）不该写成一张坏图，也不该悄悄吞掉。
            logger.warning("取证截图(%s)跳过：帧无效 %s", kind, getattr(frame, "shape", None))
            return ""
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        # 文件名：时间戳 + 会话ID + 角色。
        #
        # 时间戳取**落盘那一刻**，而不是事件里的 entered/alarm_at_seconds —— 后者在视频
        # 模式下是播放位置（秒数），拿它当时间会写出 1970 年。落盘时刻既真实，又和文件的
        # 修改时间一致（按修改时间排序与按文件名排序结果相同，留存清理用的也是修改时间）。
        # 会话ID 留着是为了能从文件名一眼回溯到 JSONL 里的那条记录。
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        screenshot_path = self.screenshot_dir / f"{stamp}_{event.session_id}_{kind}.jpg"
        encoded, buffer = cv2.imencode(".jpg", frame)
        if not encoded:
            logger.warning("取证截图(%s)编码失败: %s", kind, screenshot_path.name)
            return ""
        try:
            screenshot_path.write_bytes(buffer.tobytes())
        except OSError as error:
            logger.warning("取证截图(%s)写入失败 %s: %s", kind, screenshot_path, error)
            return ""
        return self.relative_path(screenshot_path)

    def relative_path(self, path: str | Path) -> str:
        """把绝对路径换算成相对 events 根目录的路径（已经是相对的则原样返回）。"""
        candidate = Path(path)
        if not candidate.is_absolute():
            return candidate.as_posix()
        try:
            return candidate.relative_to(self.root).as_posix()
        except ValueError:
            # 不在 events 目录下（用户手改过配置之类）：保留绝对路径，总比丢了强。
            logger.warning("截图不在 events 目录内，按绝对路径记录: %s", candidate)
            return str(candidate)

    def resolve_screenshot(self, path: str) -> Path | None:
        """把记录里的截图路径解析成实际文件；找不到返回 None。

        新记录存的是相对路径（跟着 events 目录走）。旧记录存的是绝对路径，所以还要兜一层：
        绝对路径找不到文件时，按**文件名**到当前截图目录里再找一次 —— 换过安装目录、把
        绿色版拷到别处的旧记录因此还能恢复。
        """
        if not path:
            return None
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        if candidate.is_file():
            return candidate
        fallback = self.screenshot_dir / Path(path).name
        return fallback if fallback.is_file() else None

    def open_session(self, event: AlarmEvent, frame: np.ndarray) -> AlarmEvent:
        event.entry_screenshot_path = self._save_screenshot(event, frame, "entry")
        # 进入这一刻的钟点：事件就是在有人进入区域时建出来的，所以它和 wall_time 是同一个
        # 时刻。两个都写下来 —— wall_time 是「记录时间」那一列，这个字段是「进入时间」。
        event.entered_wall_time = event.entered_wall_time or _clock_now()
        self._append({"schema_version": 2, "action": "opened", "event": self._event_payload(event)})
        return event

    def mark_alarmed(self, event: AlarmEvent, frame: np.ndarray) -> AlarmEvent:
        event.alarm_screenshot_path = self._save_screenshot(event, frame, "alarm")
        event.screenshot_path = event.alarm_screenshot_path or event.entry_screenshot_path
        # 报警那一刻的真实钟点。视频模式下 alarm_at_seconds 只是视频里的位置，没有它
        # 界面上就只能显示「2.00s」。
        event.alarmed_wall_time = _clock_now()
        self._append(
            {
                "schema_version": 2,
                "action": "alarmed",
                "session_id": event.session_id,
                "alarm_at_seconds": event.alarm_at_seconds,
                "alarmed_wall_time": event.alarmed_wall_time,
                "alarm_screenshot_path": event.alarm_screenshot_path,
                "screenshot_path": event.screenshot_path,
                "status": event.status,
                "alarm_confidence": event.alarm_confidence,
            }
        )
        return event

    def close_session(self, event: AlarmEvent) -> AlarmEvent:
        event.exited_wall_time = _clock_now()
        self._append(
            {
                "schema_version": 2,
                "action": "closed",
                "session_id": event.session_id,
                "exited_at_seconds": event.exited_at_seconds,
                "exited_wall_time": event.exited_wall_time,
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
        with self._lock:
            return self._load_recent_locked(limit, operation_mode)

    def load_all(self) -> list[AlarmEvent]:
        """读出全部会话，不设条数上限、也不按运行模式过滤。

        「查看记录」弹窗要回答的是「这个点位到底发生过什么」，所以它读全量。而
        ``load_recent`` 的 200 条上限是留给常驻面板的 —— 那张表每次启动都要填一遍，
        不能因为记录文件长到几十 MB 就拖慢启动。报警记录是纯文本，量级可以忽略。
        """
        with self._lock:
            events, _ = self._scan_locked()
            return events

    def _load_recent_locked(
        self,
        limit: int,
        operation_mode: str | None,
    ) -> list[AlarmEvent]:
        """**调用方必须已经持有 self._lock。**"""
        if operation_mode is None:
            events, _ = self._scan_locked(limit)
            return events[-limit:]
        # 按模式过滤时，尾部那一屏里可能没几条是这个模式的，所以逐级加大预算往回
        # 读，读到文件开头为止。这样既不用每次全量扫，也不会因为只看了尾巴而少给
        # 用户几条记录。
        budget = max(limit, 1)
        while True:
            events, read_everything = self._scan_locked(budget)
            matching = [event for event in events if event.operation_mode == operation_mode]
            if len(matching) >= limit or read_everything:
                return matching[-limit:]
            budget *= 4

    def _scan_locked(
        self,
        minimum_sessions: int | None = None,
    ) -> tuple[list[AlarmEvent], bool]:
        """扫出会话列表，返回 (会话, 是否读了整份文件)。

        **调用方必须已经持有 self._lock。**

        ``minimum_sessions`` 给了就从文件尾部倒着读，攒够这么多个会话起点就停：
        报警记录是永久保留的，一台有人流量的点位一年能写几十 MB，而界面一次只看
        得着最后 200 条。给 None 则整份读完（清理要重写整个文件，必须读全）。
        """
        if not self.log_path.exists():
            return [], True
        if minimum_sessions is None:
            with self.log_path.open("r", encoding="utf-8") as event_file:
                return self._fold_lines(event_file), True
        lines, read_everything = self._tail_lines(minimum_sessions)
        return self._fold_lines(lines), read_everything

    def _tail_lines(self, minimum_sessions: int) -> tuple[list[str], bool]:
        """从文件尾部倒着读，返回 (按文件顺序排列的行, 是否读到了文件开头)。

        按字节块倒读而不是「读全量再切片」：调用方要的是最后几百条，而文件可能有
        几十 MB。用二进制模式读、按 ``\\n`` 切、把首段残行留到下一块拼接 —— 这样
        跨块的多字节字符（中文区域名很常见）也只在拼完整之后才解码，不会被从中间
        截断成乱码。
        """
        chunk_size = max(1, int(self.READ_CHUNK_BYTES))
        collected: list[str] = []  # 倒序累积
        sessions_seen = 0
        with self.log_path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            position = stream.tell()
            pending = b""
            while position > 0 and sessions_seen <= minimum_sessions:
                size = min(chunk_size, position)
                position -= size
                stream.seek(position)
                block = stream.read(size) + pending
                pieces = block.split(b"\n")
                # pieces[0] 是本块开头那半行，它的前半截在更靠前的块里，留给下一轮
                # 拼；其余都是完整行，从后往前收。
                pending = pieces[0]
                for raw in reversed(pieces[1:]):
                    line = raw.decode("utf-8", errors="replace")
                    collected.append(line)
                    if self._starts_session(line):
                        sessions_seen += 1
            if pending:
                collected.append(pending.decode("utf-8", errors="replace"))
        collected.reverse()
        return collected, position == 0

    @staticmethod
    def _starts_session(line: str) -> bool:
        """这一行是否开启了一个新会话。

        会话是 opened → alarmed → closed 三行追加出来的（legacy 记录则是自成一行的
        一条完整会话），所以「新会话」＝「opened 那一行」。

        这里刻意用结构标记而不是 ``json.loads``：倒读时每收一行都要判断一次，而
        真正需要解析的只有窗口里那几百行 —— 用标记判断能省掉一半解析。标记写了
        ``\\s*`` 以容忍手工重排过的空白，但仍要求引号与冒号，避免把区域名里恰好
        出现「opened」这种字样误判成会话起点（误判会让窗口提前收窄）。
        """
        if _OPENED_MARKER.search(line):
            return True
        return not _SCHEMA_MARKER.search(line) and bool(_LEGACY_MARKER.search(line))

    def _fold_lines(self, lines: Iterable[str]) -> list[AlarmEvent]:
        """把逐行记录折叠成一份会话列表（保留首次出现的顺序）。"""
        events_by_session: dict[str, AlarmEvent] = {}
        order: list[str] = []
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            # 这个文件是纯文本，被手工编辑过就会出现非对象的行；撞上它不该让
            # 整个报警记录打不开。
            if not isinstance(payload, dict):
                continue
            if payload.get("schema_version") == 2:
                self._apply_v2_payload(payload, events_by_session, order)
                continue
            event = self._event_from_payload(payload)
            # 早期版本的单行记录：当时是**通过驻留判定、决定报警那一刻**才建出事件的
            # （见当时 detection_engine 的建法），所以那条记录的 wall_time 是报警钟点、
            # 不是进入钟点。这里把它认成报警钟点，进入钟点则如实留空 —— 反过来按
            # 「进入 = wall_time」显示，会把报警时间说成进入时间。
            event.alarmed_wall_time = event.alarmed_wall_time or event.wall_time
            events_by_session[event.session_id] = event
            if event.session_id not in order:
                order.append(event.session_id)
        return [self._recover_clock(events_by_session[session_id]) for session_id in order]

    @staticmethod
    def _optional_float(payload: dict[str, object], key: str) -> float | None:
        """可选的浮点字段：没记过、或者被手改成读不出来的东西，都当没记。

        报警记录是纯文本，用户能打开也能改。新加的可选字段不该让整条记录读不出来 ——
        而 ``float("abc")`` 抛的 ValueError 会一路冒出 ``_fold_lines``，把整块报警记录
        一起废掉。
        """
        value = payload.get(key)
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

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
            entered_wall_time=str(payload.get("entered_wall_time", "")),
            alarmed_wall_time=str(payload.get("alarmed_wall_time", "")),
            exited_wall_time=str(payload.get("exited_wall_time", "")),
            alarm_confidence=EventStore._optional_float(payload, "alarm_confidence"),
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
            # v2 的 opened 行是**有人进入区域时**写下的（见 detection_engine），所以它的
            # wall_time 就是进入钟点 —— 但只对「那时还没有 entered_wall_time 字段」的记录
            # 这么补（键都不存在）。键在、值为空，是清空记录把老记录重写成了 v2 形态，那种
            # 情况下进入钟点确实没记过，不能拿 wall_time 顶上（它是报警钟点）。
            if "entered_wall_time" not in event_payload:
                event.entered_wall_time = event.wall_time
            events_by_session[event.session_id] = event
            order.append(event.session_id)
            return
        session_id = str(payload.get("session_id", ""))
        event = events_by_session.get(session_id)
        if event is None:
            return
        if action == "alarmed":
            event.alarm_at_seconds = float(payload["alarm_at_seconds"])
            event.alarmed_wall_time = str(payload.get("alarmed_wall_time", ""))
            event.alarm_screenshot_path = str(payload.get("alarm_screenshot_path", ""))
            event.screenshot_path = str(payload.get("screenshot_path", event.alarm_screenshot_path))
            event.status = "alarmed"
            event.alarm_confidence = cls._optional_float(payload, "alarm_confidence")
        elif action == "closed":
            event.exited_at_seconds = float(payload["exited_at_seconds"])
            event.exited_wall_time = str(payload.get("exited_wall_time", ""))
            event.duration_seconds = float(payload["duration_seconds"])
            event.status = "completed"

    @staticmethod
    def _clock_from_screenshot(path: str) -> str:
        """从取证截图的文件名里读出落盘钟点（"2026-09-21 14:36:10"）；读不出返回空串。

        文件名是 ``{时间戳}_{会话ID}_{角色}.jpg``，时间戳取的就是**落盘那一刻**（见
        ``_save_screenshot``）。所以「记录里没写报警钟点」的那批记录，钟点其实一直躺在
        文件名里 —— 加这两个字段之前写下的记录，靠它就能显示出真实时间。
        """
        match = _SCREENSHOT_STAMP.search(Path(path).name)
        if match is None:
            return ""
        try:
            stamp = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            return ""
        return stamp.strftime("%Y-%m-%d %H:%M:%S")

    def _recover_clock(self, event: AlarmEvent) -> AlarmEvent:
        """给「加字段之前写下」的记录补上报警钟点。

        只补报警那一刻：进入那一刻的钟点记在 ``wall_time`` 里；退出那一刻没有截图，
        文件名里也就没有它，补不了 —— 界面会如实说没记，而不是拿视频位置冒充时间。
        """
        if event.operation_mode != "video" or event.alarmed_wall_time:
            return event
        event.alarmed_wall_time = self._clock_from_screenshot(event.alarm_screenshot_path)
        return event

    def clear(self, operation_mode: str | None = None) -> None:
        with self._lock:
            if not self.log_path.exists():
                return
            if operation_mode is None:
                self.log_path.unlink()
                return
            # 读-改-写全程待在同一把锁里。检测线程随时可能往这个文件追加一条，而
            # 「清空记录」按钮在检测运行中是可以点的 —— 中间松开锁的话，那条刚写
            # 下的报警会连同别的记录一起被这份重写覆盖掉，日志里查不到、截图却还在。
            # 这里也不再限制条数：为了清一个模式而丢掉另一个模式的更早记录，没有道理。
            events, _ = self._scan_locked()
            retained = [
                event for event in events if event.operation_mode != operation_mode
            ]
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
