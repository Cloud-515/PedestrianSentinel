"""报警的远程通知。

本机蜂鸣解决不了这个产品的核心问题：没人守在现场时，「报警」必须能离开这台机器。
这不只是图方便 —— 取证材料同样不该只存在于被监控现场的那一台机器上，机器被搬走、
盘坏掉、断电，证据就一起没了。

几处刻意的取舍：

* **只用标准库**（``urllib``）。这个项目靠 PyInstaller 打包发布，多一个 requests 就多
  一份要跟着打进包、跟着升级、跟着排查的东西，而这里只需要发一个 POST。
* **发送在后台线程里做**。检测线程和界面都不该被网络超时拖住 —— 一次连接超时 5 秒，
  在检测循环里就是 5 秒没有任何帧被处理。
* **队列有上限**。网络长时间不通时宁可丢弃最旧的待发通知，也不要把内存堆起来；本地
  的报警记录与截图早就是落盘的，通知丢了不会丢事实。
* **默认关闭**。往外部地址发数据必须由用户显式打开，所以配置默认是关的。
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from models import (
    NOTIFICATION_FORMAT_GENERIC,
    NOTIFICATION_FORMAT_TEXT_BOT,
    AlarmEvent,
)

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 5.0
MAX_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 3.0

# 待发队列的上限。一次报警一条，32 条已经足够覆盖「网络抖了一会儿」这种情况。
MAX_PENDING = 32

# 附带的截图超过这个大小就不附带了。企业微信机器人的请求体上限是 2 MB，
# base64 还要再涨三分之一，所以这里卡在 1 MB。
MAX_SCREENSHOT_BYTES = 1024 * 1024


@dataclass(frozen=True)
class NotificationSettings:
    enabled: bool = False
    url: str = ""
    payload_format: str = NOTIFICATION_FORMAT_GENERIC
    include_screenshot: bool = False

    @property
    def usable(self) -> bool:
        """开关打开、地址填了、并且是个 http(s) 地址。"""
        return (
            self.enabled
            and bool(self.url.strip())
            and self.url.strip().lower().startswith(("http://", "https://"))
        )


@dataclass(frozen=True)
class NotificationJob:
    url: str
    body: bytes
    label: str


@dataclass(frozen=True)
class NotificationResult:
    ok: bool
    message: str
    attempts: int = 0


def _event_payload(event: AlarmEvent) -> dict[str, object]:
    return {
        "session_id": event.session_id,
        "wall_time": event.wall_time,
        "zone_name": event.zone_name,
        "track_id": event.track_id,
        "video_source": event.source,
        "operation_mode": event.operation_mode,
        "entered_at_seconds": event.entered_at_seconds,
        "alarm_at_seconds": event.alarm_at_seconds,
        "screenshot_path": event.alarm_screenshot_path or event.entry_screenshot_path,
    }


def build_text(event: AlarmEvent) -> str:
    """给人看的一句话。企业微信/钉钉的文本消息直接用这个。"""
    return "\n".join(
        [
            f"【闯入报警】{event.zone_name}",
            f"时间：{event.wall_time}",
            f"目标 ID：{event.track_id}",
            f"视频源：{event.source}",
            f"报警时刻：{event.format_moment('alarmed')}",
        ]
    )


def build_body(
    event: AlarmEvent,
    settings: NotificationSettings,
    *,
    screenshot_reader: Callable[[str], bytes | None] | None = None,
) -> bytes:
    """拼出请求体。

    两种形态：``generic`` 是自描述的 JSON（本程序自己的字段都在 ``event`` 里），
    ``text_bot`` 是 ``{"msgtype":"text","text":{"content":...}}`` —— 企业微信与钉钉
    的群机器人都是这个形状，也是这类通知最常见的落点。
    """
    text = build_text(event)
    if settings.payload_format == NOTIFICATION_FORMAT_TEXT_BOT:
        # 文本机器人只能收文本，附不了图；这一点在界面上也写明了。
        return json.dumps(
            {"msgtype": "text", "text": {"content": text}}, ensure_ascii=False
        ).encode("utf-8")

    payload: dict[str, object] = {"text": text, "event": _event_payload(event)}
    if settings.include_screenshot:
        path = event.alarm_screenshot_path or event.entry_screenshot_path
        image = (screenshot_reader or _read_screenshot)(path) if path else None
        if image is not None:
            payload["screenshot_name"] = Path(path).name
            payload["screenshot_base64"] = base64.b64encode(image).decode("ascii")
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _read_screenshot(path: str) -> bytes | None:
    try:
        size = Path(path).stat().st_size
    except OSError as error:
        logger.warning("通知附带截图失败，读不到 %s: %s", path, error)
        return None
    if size > MAX_SCREENSHOT_BYTES:
        logger.warning(
            "通知未附带截图：%s 有 %.1f MB，超过 %.0f MB 的上限",
            path,
            size / 1024**2,
            MAX_SCREENSHOT_BYTES / 1024**2,
        )
        return None
    try:
        return Path(path).read_bytes()
    except OSError as error:
        logger.warning("通知附带截图失败，读不到 %s: %s", path, error)
        return None


def build_job(event: AlarmEvent, settings: NotificationSettings) -> NotificationJob | None:
    """配置不可用时返回 None —— 调用方据此决定要不要提示用户。"""
    if not settings.usable:
        return None
    return NotificationJob(
        url=settings.url.strip(),
        body=build_body(event, settings),
        label=f"{event.zone_name} / 目标 {event.track_id}",
    )


def post_notification(
    job: NotificationJob,
    *,
    opener: Callable[..., object] = urllib.request.urlopen,
    sleeper: Callable[[float], None] = time.sleep,
) -> NotificationResult:
    """发一条通知，失败重试 MAX_ATTEMPTS 次。

    异常一律吃掉并转成 NotificationResult：报警本身已经落盘了，通知发不出去不该
    把检测线程或界面带崩，只该留下一条能查的日志。
    """
    request = urllib.request.Request(
        job.url,
        data=job.body,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "PedestrianZoneMonitor",
        },
        method="POST",
    )
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with opener(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", 200)
            if not 200 <= int(status) < 300:
                last_error = f"HTTP {status}"
            else:
                logger.info("已发送报警通知（%s）", job.label)
                return NotificationResult(True, "通知已发送", attempt)
        except urllib.error.HTTPError as error:
            last_error = f"HTTP {error.code}"
        except urllib.error.URLError as error:
            last_error = f"连接失败: {error.reason}"
        except Exception as error:  # noqa: BLE001 - 见下面那句
            # 刻意接住所有异常，而不是只接 OSError/ValueError：这条链路横跨 DNS、
            # TLS、代理与操作系统，出什么怪事都有可能（SSL 库自身抛 RuntimeError
            # 就是真实遇到过的一种）。它是「发送」这件事的边界，让异常逃出去只会
            # 把后台发送线程带走，而报警本身早就落盘了。
            last_error = f"{type(error).__name__}: {error}"
        if attempt < MAX_ATTEMPTS:
            sleeper(RETRY_DELAY_SECONDS)
    logger.warning("报警通知发送失败（%s）：%s", job.label, last_error)
    return NotificationResult(False, f"通知发送失败：{last_error}", MAX_ATTEMPTS)


@dataclass(frozen=True)
class _Pending:
    job: NotificationJob
    on_result: Callable[[NotificationResult], None] | None


class NotificationDispatcher:
    """把通知丢进队列，由一条后台线程依次发送。

    界面和检测线程调用的只有 :meth:`notify`，它只做「拼好的一次性数据入队」，不碰
    网络、不碰磁盘。``on_result`` 在发送线程里被调用，界面侧要自己转到 GUI 线程
    （MainWindow 用的是 Qt 信号，跨线程投递由 Qt 排队完成）。
    """

    def __init__(
        self,
        *,
        sender: Callable[[NotificationJob], NotificationResult] = post_notification,
    ) -> None:
        self._sender = sender
        self._queue: queue.Queue[_Pending | None] = queue.Queue(maxsize=MAX_PENDING)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._sent = 0
        self._failed = 0
        self._dropped = 0

    def notify(
        self,
        job: NotificationJob,
        on_result: Callable[[NotificationResult], None] | None = None,
    ) -> bool:
        """入队。队列满时丢掉这一条并返回 False（本地记录不受影响）。"""
        self._ensure_thread()
        try:
            self._queue.put_nowait(_Pending(job, on_result))
        except queue.Full:
            with self._lock:
                self._dropped += 1
                dropped = self._dropped
            # 只在第一条和每 10 条被丢时各说一次：断网一晚上不该刷出几千行日志。
            if dropped == 1 or dropped % 10 == 0:
                logger.warning(
                    "通知队列已满(%d)，已丢弃 %d 条待发通知（报警记录与截图不受影响）",
                    MAX_PENDING,
                    dropped,
                )
            return False
        return True

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"sent": self._sent, "failed": self._failed, "dropped": self._dropped}

    def close(self, timeout: float = 2.0) -> None:
        """收尾：让孩子线程把已经入队的通知发完再退出。"""
        thread = self._thread
        if thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread.join(timeout)
        self._thread = None

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run, name="notification-sender", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            pending = self._queue.get()
            try:
                if pending is None:
                    return
                try:
                    result = self._sender(pending.job)
                except Exception as error:  # noqa: BLE001 - 发送端是注入点，别让它带走整条线程
                    logger.exception("通知发送端抛出异常")
                    result = NotificationResult(False, f"通知发送失败：{error}")
                with self._lock:
                    if result.ok:
                        self._sent += 1
                    else:
                        self._failed += 1
                if pending.on_result is not None:
                    try:
                        pending.on_result(result)
                    except Exception:  # noqa: BLE001 - 回调是界面的事，别把发送线程炸掉
                        logger.exception("通知结果回调失败")
            finally:
                self._queue.task_done()


def synthetic_event(*, zone_name: str = "测试区域") -> AlarmEvent:
    """给「发送测试」用的一条假事件。

    刻意不复用真实报警事件：用户在没人的时候按这个按钮，不该让界面上多出一条
    看起来像真事的报警记录，也不该往 events\\ 里写东西。

    名字不叫 ``test_*``：那个前缀会被 pytest 当成测试函数收集（它一被 import 进
    测试模块就会触发）。
    """
    now = time.time()
    return AlarmEvent(
        source="（界面测试）",
        zone_name=zone_name,
        track_id="-",
        entered_at_seconds=now,
        alarm_at_seconds=now,
        wall_time=time.strftime("%Y-%m-%d %H:%M:%S"),
        operation_mode="monitor",
    )
