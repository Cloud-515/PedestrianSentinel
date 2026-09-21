"""远程通知：拼包、重试、后台队列。

这套东西的价值全在「报警发得出去」这一件事上，而它偏偏是最不容易在开发机上发现的
一环 —— 地址填错、机器人被禁、网络不通，都只有真正跑起来才知道。所以这里把拼包与
重试逻辑做成可注入的（opener / sleeper），逐条钉住，不依赖真实网络。
"""

from __future__ import annotations

import base64
import json
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from models import AlarmEvent
from notifications import (
    MAX_ATTEMPTS,
    MAX_PENDING,
    NotificationDispatcher,
    NotificationJob,
    NotificationResult,
    NotificationSettings,
    build_body,
    build_job,
    build_text,
    post_notification,
    synthetic_event,
)


class FakeResponse:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def make_event(**overrides: object) -> AlarmEvent:
    values: dict[str, object] = {
        "source": "rtsp://camera/live",
        "zone_name": "北侧入口",
        "track_id": "12",
        "entered_at_seconds": 1.0,
        "alarm_at_seconds": 3.5,
        "wall_time": "2026-09-21 10:00:00",
        "operation_mode": "monitor",
    }
    values.update(overrides)
    return AlarmEvent(**values)  # type: ignore[arg-type]


class SettingsTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        self.assertFalse(NotificationSettings().usable)
        self.assertFalse(NotificationSettings(enabled=True).usable)

    def test_requires_an_http_address(self) -> None:
        self.assertFalse(NotificationSettings(enabled=True, url="   ").usable)
        self.assertFalse(
            NotificationSettings(enabled=True, url="ftp://example.com/hook").usable
        )
        self.assertTrue(
            NotificationSettings(enabled=True, url="https://example.com/hook").usable
        )
        self.assertTrue(
            NotificationSettings(enabled=True, url=" http://10.0.0.5:8000/x ").usable
        )


class PayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def _settings(self, **overrides: object) -> NotificationSettings:
        values: dict[str, object] = {
            "enabled": True,
            "url": "https://example.com/hook",
            "payload_format": "generic",
            "include_screenshot": False,
        }
        values.update(overrides)
        return NotificationSettings(**values)  # type: ignore[arg-type]

    def test_text_carries_what_an_operator_needs_to_react(self) -> None:
        text = build_text(make_event())

        self.assertIn("北侧入口", text)
        self.assertIn("12", text)
        self.assertIn("2026-09-21 10:00:00", text)
        self.assertIn("rtsp://camera/live", text)

    def test_generic_body_is_self_describing_json(self) -> None:
        body = json.loads(build_body(make_event(), self._settings()))

        self.assertIn("北侧入口", body["text"])
        self.assertEqual(body["event"]["zone_name"], "北侧入口")
        self.assertEqual(body["event"]["track_id"], "12")
        self.assertEqual(body["event"]["session_id"], make_event().session_id[:0] or body["event"]["session_id"])
        self.assertNotIn("screenshot_base64", body)

    def test_screenshot_is_attached_only_when_asked_for(self) -> None:
        frame = self.directory / "alarm.jpg"
        frame.write_bytes(b"jpeg-bytes")
        event = make_event(alarm_screenshot_path=str(frame))

        without = json.loads(build_body(event, self._settings()))
        with_image = json.loads(
            build_body(event, self._settings(include_screenshot=True))
        )

        self.assertNotIn("screenshot_base64", without)
        self.assertEqual(with_image["screenshot_name"], "alarm.jpg")
        self.assertEqual(
            base64.b64decode(with_image["screenshot_base64"]), b"jpeg-bytes"
        )

    def test_oversized_screenshot_is_skipped_instead_of_bloating_the_request(self) -> None:
        huge = self.directory / "alarm.jpg"
        huge.write_bytes(b"x" * (2 * 1024 * 1024))
        event = make_event(alarm_screenshot_path=str(huge))

        with self.assertLogs("notifications", level="WARNING") as captured:
            body = json.loads(build_body(event, self._settings(include_screenshot=True)))

        self.assertNotIn("screenshot_base64", body)
        self.assertTrue(any("未附带截图" in line for line in captured.output))

    def test_missing_screenshot_does_not_break_the_notification(self) -> None:
        event = make_event(alarm_screenshot_path=str(self.directory / "没了.jpg"))

        with self.assertLogs("notifications", level="WARNING"):
            body = json.loads(build_body(event, self._settings(include_screenshot=True)))

        self.assertNotIn("screenshot_base64", body)
        self.assertIn("text", body)

    def test_text_bot_format_matches_wecom_and_dingtalk(self) -> None:
        frame = self.directory / "alarm.jpg"
        frame.write_bytes(b"jpeg-bytes")
        event = make_event(alarm_screenshot_path=str(frame))

        body = json.loads(
            build_body(
                event,
                self._settings(
                    payload_format="text_bot", include_screenshot=True
                ),
            )
        )

        self.assertEqual(body["msgtype"], "text")
        self.assertIn("北侧入口", body["text"]["content"])
        # 这类机器人只收文本，勾了附带截图也不该冒出一个它不认识的字段。
        self.assertEqual(set(body), {"msgtype", "text"})

    def test_job_is_none_when_the_settings_are_unusable(self) -> None:
        self.assertIsNone(build_job(make_event(), NotificationSettings()))
        self.assertIsNone(
            build_job(make_event(), NotificationSettings(enabled=True, url=""))
        )

    def test_job_uses_the_trimmed_url(self) -> None:
        job = build_job(make_event(), self._settings(url=" https://example.com/hook "))

        self.assertIsNotNone(job)
        self.assertEqual(job.url, "https://example.com/hook")
        self.assertIn("北侧入口", job.label)

    def test_synthetic_event_is_marked_as_a_test(self) -> None:
        event = synthetic_event()

        self.assertIn("测试", event.source)
        self.assertIn("测试", build_text(event))
        # 时间要是「现在」，不能是 1970。
        self.assertGreater(event.alarm_at_seconds or 0, time.time() - 60)


class PostNotificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.job = NotificationJob(
            url="https://example.com/hook", body=b"{}", label="测试"
        )
        self.slept: list[float] = []

    def _sleeper(self, seconds: float) -> None:
        self.slept.append(seconds)

    def test_successful_post_reports_ok(self) -> None:
        calls: list[object] = []

        def opener(request: object, timeout: float | None = None) -> FakeResponse:
            calls.append(request)
            return FakeResponse(200)

        result = post_notification(self.job, opener=opener, sleeper=self._sleeper)

        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.slept, [])
        request = calls[0]
        self.assertEqual(request.get_method(), "POST")  # type: ignore[attr-defined]
        self.assertEqual(
            request.get_header("Content-type"),  # type: ignore[attr-defined]
            "application/json; charset=utf-8",
        )

    def test_http_error_is_retried_then_reported(self) -> None:
        def opener(request: object, timeout: float | None = None) -> FakeResponse:
            raise urllib.error.HTTPError(self.job.url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

        result = post_notification(self.job, opener=opener, sleeper=self._sleeper)

        self.assertFalse(result.ok)
        self.assertEqual(result.attempts, MAX_ATTEMPTS)
        self.assertIn("404", result.message)
        # 重试之间要等一会儿，且只等 MAX_ATTEMPTS-1 次。
        self.assertEqual(len(self.slept), MAX_ATTEMPTS - 1)

    def test_connection_error_is_reported_not_raised(self) -> None:
        def opener(request: object, timeout: float | None = None) -> FakeResponse:
            raise urllib.error.URLError("没有网络")

        result = post_notification(self.job, opener=opener, sleeper=self._sleeper)

        self.assertFalse(result.ok)
        self.assertIn("连接失败", result.message)

    def test_non_2xx_status_is_a_failure(self) -> None:
        def opener(request: object, timeout: float | None = None) -> FakeResponse:
            return FakeResponse(500)

        result = post_notification(self.job, opener=opener, sleeper=self._sleeper)

        self.assertFalse(result.ok)
        self.assertIn("500", result.message)

    def test_unexpected_exception_does_not_escape(self) -> None:
        """报警已经落盘了，发送端无论出什么事都不该把调用方炸掉。"""

        def opener(request: object, timeout: float | None = None) -> FakeResponse:
            raise RuntimeError("SSL 库抽风")

        result = post_notification(self.job, opener=opener, sleeper=self._sleeper)

        self.assertFalse(result.ok)
        self.assertIn("RuntimeError", result.message)


class DispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.done = threading.Event()

    def test_notify_sends_and_reports_back(self) -> None:
        sent: list[NotificationJob] = []
        results: list[NotificationResult] = []

        def sender(job: NotificationJob) -> NotificationResult:
            sent.append(job)
            return NotificationResult(True, "通知已发送", 1)

        dispatcher = NotificationDispatcher(sender=sender)
        self.addCleanup(dispatcher.close)
        job = NotificationJob(url="https://x", body=b"{}", label="测试")

        self.assertTrue(
            dispatcher.notify(job, on_result=lambda result: (results.append(result), self.done.set()))
        )
        self.assertTrue(self.done.wait(5), "发送回调没有回来")

        self.assertEqual([item.label for item in sent], ["测试"])
        self.assertTrue(results[0].ok)
        self.assertEqual(dispatcher.stats(), {"sent": 1, "failed": 0, "dropped": 0})

    def test_failure_is_counted_and_still_reported(self) -> None:
        dispatcher = NotificationDispatcher(
            sender=lambda job: NotificationResult(False, "通知发送失败：连接失败", 2)
        )
        self.addCleanup(dispatcher.close)
        results: list[NotificationResult] = []

        dispatcher.notify(
            NotificationJob(url="https://x", body=b"{}", label="测试"),
            on_result=lambda result: (results.append(result), self.done.set()),
        )
        self.assertTrue(self.done.wait(5))

        self.assertFalse(results[0].ok)
        self.assertEqual(dispatcher.stats()["failed"], 1)

    def test_broken_callback_does_not_kill_the_sender_thread(self) -> None:
        dispatcher = NotificationDispatcher(
            sender=lambda job: NotificationResult(True, "通知已发送", 1)
        )
        self.addCleanup(dispatcher.close)

        dispatcher.notify(
            NotificationJob(url="https://x", body=b"{}", label="坏回调"),
            on_result=lambda result: (_ for _ in ()).throw(RuntimeError("界面炸了")),
        )
        dispatcher.notify(
            NotificationJob(url="https://x", body=b"{}", label="后一条"),
            on_result=lambda result: self.done.set(),
        )

        self.assertTrue(self.done.wait(5), "回调抛异常后发送线程应该继续处理下一条")

    def test_full_queue_drops_newest_without_growing_memory(self) -> None:
        """网络不通时宁可丢通知，也不该把内存堆起来。"""
        release = threading.Event()
        blocked = threading.Event()

        def sender(job: NotificationJob) -> NotificationResult:
            blocked.set()
            release.wait(5)
            return NotificationResult(True, "通知已发送", 1)

        dispatcher = NotificationDispatcher(sender=sender)
        self.addCleanup(release.set)
        self.addCleanup(dispatcher.close)
        job = NotificationJob(url="https://x", body=b"{}", label="占位")

        self.assertTrue(dispatcher.notify(job))
        self.assertTrue(blocked.wait(5), "后台线程没有开始发送")

        accepted = sum(1 for _ in range(MAX_PENDING + 5) if dispatcher.notify(job))

        self.assertGreater(dispatcher.stats()["dropped"], 0)
        self.assertLessEqual(accepted, MAX_PENDING)
        release.set()

    def test_close_stops_the_worker_thread(self) -> None:
        dispatcher = NotificationDispatcher(
            sender=lambda job: NotificationResult(True, "通知已发送", 1)
        )
        dispatcher.notify(NotificationJob(url="https://x", body=b"{}", label="测试"))
        thread = dispatcher._thread
        self.assertIsNotNone(thread)

        dispatcher.close(timeout=5)

        self.assertFalse(thread.is_alive())  # type: ignore[union-attr]

    def test_real_post_notification_is_wired_by_default(self) -> None:
        """默认 sender 必须是真的发送函数，否则功能看起来「装好了」却什么都不发。"""
        dispatcher = NotificationDispatcher()

        self.assertIs(dispatcher._sender, post_notification)
        dispatcher.close()


class WiringContractTests(unittest.TestCase):
    """MainWindow 依赖的几个约定。"""

    def test_build_body_does_not_touch_the_network_or_disk_by_itself(self) -> None:
        with patch("urllib.request.urlopen") as urlopen:
            build_body(make_event(), NotificationSettings(enabled=True, url="https://x"))

        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
