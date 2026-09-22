"""配置读取的容错。

config.json 是个纯文本文件，用户能打开、能手改，改坏一个字符完全不奇怪。原来的
``AppConfig.from_dict`` 是一路裸转换：任何一个字段类型不对都会抛异常，于是

* 调用方只能整份回落默认值 —— 用户手改错一个字符，代价是所有警戒区域一起消失，
  而且退出时那份默认配置还会被写回 config.json 把原文件覆盖掉；
* ``"zones": {...}`` 这种写法抛的是 AttributeError，而 ``_load_config`` 只接
  (OSError, ValueError, TypeError)，异常一路冒到 main() 之外，程序每次启动都只弹
  一个「程序出错」，用户只能自己想到去删 config.json。

这里锁住的行为是：坏字段只废掉它自己，好字段照旧，且**任何**输入都不抛异常。
"""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from config_store import ConfigStore
from models import (
    MAX_COLUMN_WIDTH,
    MIN_COLUMN_WIDTH,
    AlarmEvent,
    AppConfig,
    ZoneDefinition,
    ZoneProfile,
)

GOOD_ZONES = [
    {
        "name": "卸货区",
        "polygon": [[10, 20], [30, 40], [50, 60]],
        "closed": True,
        "dwell_seconds": 3.0,
        "cooldown_seconds": 30.0,
        "color": "#1E88E5",
    }
]


def base_config() -> dict:
    return {
        "version": 1,
        "source": "D:/素材/demo.mp4",
        "source_type": "file",
        "operation_mode": "video",
        "video_source": "D:/素材/demo.mp4",
        "monitor_source": "0",
        "active_profile": "夜间值守",
        "loop_playback": True,
        "playback_speed": 2.0,
        "model_path": "yolo11n.pt",
        "inference_device": "cpu",
        "cpu_low_power_preset": True,
        "source_size": [1920, 1080],
        "display_to_original_scale": {"x": 2.5, "y": 2.5},
        "zones": GOOD_ZONES,
    }


class BrokenFieldIsolationTests(unittest.TestCase):
    """一个坏字段不该连累其它字段。"""

    def test_broken_scalar_field_keeps_zones_and_the_rest(self) -> None:
        for field_name, broken in (
            ("playback_speed", "fast"),
            ("source_size", ["a", 1080]),
            ("loop_playback", "也许"),
            ("display_to_original_scale", "不是对象"),
            ("test_mode", []),
        ):
            with self.subTest(field=field_name):
                payload = {**base_config(), field_name: broken}
                config = AppConfig.from_dict(payload)

                # 坏字段回落，其余照旧 —— 尤其是区域，它是用户唯一没法重来的东西。
                self.assertEqual(len(config.zones), 1)
                self.assertEqual(config.zones[0].name, "卸货区")
                self.assertEqual(config.active_profile, "夜间值守")
                self.assertEqual(config.video_source, "D:/素材/demo.mp4")

    def test_single_broken_zone_does_not_discard_the_others(self) -> None:
        payload = {
            **base_config(),
            "zones": [
                GOOD_ZONES[0],
                {"name": "坏区域", "polygon": "123", "dwell_seconds": "x"},
                "字符串不是区域对象",
                {"name": "第二个好区域", "polygon": [[1, 2], [3, 4], [5, 6]]},
            ],
        }

        config = AppConfig.from_dict(payload)

        self.assertEqual(
            [zone.name for zone in config.zones],
            ["卸货区", "坏区域", "第二个好区域"],
        )
        # 坏顶点被丢掉，不是把整份配置丢掉。
        self.assertEqual(config.zones[1].polygon, [])
        self.assertEqual(config.zones[1].dwell_seconds, 2.0)
        self.assertEqual(len(config.zones[2].polygon), 3)

    def test_zones_written_as_object_does_not_raise_attribute_error(self) -> None:
        """这曾经是启动崩溃的来源：'str' object has no attribute 'get'。"""
        config = AppConfig.from_dict({**base_config(), "zones": {"区域1": {}}})

        self.assertEqual(config.zones, [])

    def test_non_dict_root_is_treated_as_empty(self) -> None:
        for payload in ([], "config", 3, None):
            with self.subTest(payload=payload):
                config = AppConfig.from_dict(payload)  # type: ignore[arg-type]
                self.assertEqual(config.zones, [])
                self.assertEqual(config.inference_device, "cpu")

    def test_malformed_config_never_raises(self) -> None:
        """一次性铺开各种坏数据，确认 from_dict 一个都不抛。"""
        payloads = [
            {},
            {"zones": None},
            {"zones": [None, 1, [], "x"]},
            {"zones": [{"polygon": [None, [1], [1, "a"], [float("nan"), 2]]}]},
            {"source_size": []},
            {"source_size": None},
            {"display_to_original_scale": {"x": None, "y": "z"}},
            {"playback_speed": None},
            {"active_profile": 0},
            {"camera_history": "0"},
            {"file_history": [None, 1, "a.mp4"]},
            {"operation_mode": 3},
            {"cpu_low_power_preset": "true"},
        ]

        for payload in payloads:
            with self.subTest(payload=payload):
                AppConfig.from_dict(payload)


class FieldCoercionTests(unittest.TestCase):
    def test_polygon_keeps_only_real_vertices(self) -> None:
        zone = ZoneDefinition.from_dict(
            {
                "name": "区域",
                "polygon": [
                    [1, 2],
                    [3.5, 4.5, 6],  # 多出来的坐标忽略
                    [7],  # 只有 1 个坐标 —— 正是会让 cv2.pointPolygonTest 断言失败的那种点
                    "字符串",
                    None,
                    ["a", "b"],
                    [float("inf"), 0],
                    [8, 9],
                ],
            }
        )

        self.assertEqual(zone.polygon, [[1.0, 2.0], [3.5, 4.5], [8.0, 9.0]])

    def test_closed_requires_at_least_three_vertices(self) -> None:
        two_points = ZoneDefinition.from_dict({"name": "两点", "polygon": [[1, 2], [3, 4]], "closed": True})
        self.assertFalse(two_points.closed)

        three_points = ZoneDefinition.from_dict(
            {"name": "三点", "polygon": [[1, 2], [3, 4], [5, 6]], "closed": True}
        )
        self.assertTrue(three_points.closed)

    def test_thresholds_are_clamped_to_sane_values(self) -> None:
        zone = ZoneDefinition.from_dict(
            {"name": "区域", "polygon": [], "dwell_seconds": -5, "cooldown_seconds": -1}
        )

        # 负数停留阈值等于「一进区就报警」，不能让它从配置里溜进来。
        self.assertEqual(zone.dwell_seconds, 0.0)
        self.assertEqual(zone.cooldown_seconds, 0.0)

    def test_invalid_color_falls_back_to_default(self) -> None:
        for color in ("red", "#GGG", "#12345", 123, None):
            with self.subTest(color=color):
                zone = ZoneDefinition.from_dict({"name": "区域", "color": color})
                self.assertEqual(zone.color, "#E53935")

    def test_playback_speed_is_clamped_to_the_slider_range(self) -> None:
        self.assertEqual(AppConfig.from_dict({"playback_speed": 100}).playback_speed, 4.0)
        self.assertEqual(AppConfig.from_dict({"playback_speed": 0}).playback_speed, 0.25)

    def test_blank_required_strings_fall_back(self) -> None:
        config = AppConfig.from_dict({"model_path": "   ", "inference_device": ""})

        self.assertEqual(config.model_path, "yolo11n.pt")
        self.assertEqual(config.inference_device, "cpu")

    def test_version_is_written_and_restored(self) -> None:
        config = AppConfig.from_dict({**base_config(), "version": 0})

        self.assertEqual(config.version, 1)
        self.assertEqual(config.to_dict()["version"], 1)
        self.assertEqual(AppConfig.from_dict(config.to_dict()).version, 1)

    def test_screenshot_cleanup_is_disabled_by_default(self) -> None:
        """自动删除取证材料这件事必须由用户显式打开。"""
        config = AppConfig.from_dict({})

        self.assertFalse(config.screenshot_retention_enabled)
        # 数值仍然预填成建议值，但没打开开关就不会生效。
        self.assertEqual(config.screenshot_retention_days, 15)
        self.assertEqual(config.screenshot_retention_mb, 2048)

    def test_legacy_retention_numbers_do_not_enable_cleanup(self) -> None:
        """旧版本的 config.json 里只有天数与上限（当时默认开启）。

        那些数值是程序自己写进去的默认值，用户从没点过头 —— 升级时不能按「填了数值
        就是同意」推断，否则等于替他同意删自己的取证材料。
        """
        with self.assertLogs("models", level=logging.INFO) as captured:
            config = AppConfig.from_dict(
                {"screenshot_retention_days": 15, "screenshot_retention_mb": 2048}
            )

        self.assertFalse(config.screenshot_retention_enabled)
        self.assertTrue(
            any("未启用自动清理" in message for message in captured.output), captured.output
        )

    def test_retention_switch_round_trips(self) -> None:
        config = AppConfig.from_dict(
            {
                "screenshot_retention_enabled": True,
                "screenshot_retention_days": 3,
                "screenshot_retention_mb": 256,
            }
        )

        self.assertTrue(config.screenshot_retention_enabled)
        restored = AppConfig.from_dict(config.to_dict())
        self.assertTrue(restored.screenshot_retention_enabled)
        self.assertEqual(restored.screenshot_retention_days, 3)
        self.assertEqual(restored.screenshot_retention_mb, 256)

    def test_column_widths_round_trip(self) -> None:
        config = AppConfig.from_dict(
            {**base_config(), "history_column_widths": {"区域": 260, "记录时间": 180}}
        )

        self.assertEqual(config.history_column_widths, {"区域": 260, "记录时间": 180})
        restored = AppConfig.from_dict(config.to_dict())
        self.assertEqual(restored.history_column_widths, {"区域": 260, "记录时间": 180})

    def test_detection_details_are_off_by_default_and_round_trip(self) -> None:
        """画面上那行「目标编号 / 置信度」默认不显示。

        默认开着的话，看画面的人第一眼看到的就是一串他不需要的数字（而且行更长、
        更容易和别人的标签挤在一起）—— 那是工程视角，不是监控视角。
        """
        self.assertFalse(AppConfig.from_dict({}).show_detection_details)
        self.assertFalse(AppConfig.from_dict(base_config()).show_detection_details)

        config = AppConfig.from_dict({**base_config(), "show_detection_details": True})

        self.assertTrue(config.show_detection_details)
        self.assertTrue(AppConfig.from_dict(config.to_dict()).show_detection_details)

    def test_a_broken_details_switch_falls_back_to_off(self) -> None:
        config = AppConfig.from_dict(
            {**base_config(), "show_detection_details": "是的"}
        )

        self.assertFalse(config.show_detection_details)

    def test_column_widths_are_dropped_one_by_one_when_broken(self) -> None:
        """这是程序自己写的界面状态，坏条目丢掉就行 —— 宽度随时还能拖回来。"""
        config = AppConfig.from_dict(
            {
                **base_config(),
                "history_column_widths": {
                    "区域": 260,
                    "写成了文字": "很宽",
                    "负的": -5,
                    "布尔值": True,
                    5: 100,  # JSON 里键都是字符串，这里模拟手改出来的非字符串键
                },
            }
        )

        self.assertEqual(config.history_column_widths, {"区域": 260})

    def test_column_widths_are_clamped(self) -> None:
        config = AppConfig.from_dict(
            {**base_config(), "history_column_widths": {"太窄": 1, "太宽": 999999}}
        )

        self.assertEqual(
            config.history_column_widths,
            {"太窄": MIN_COLUMN_WIDTH, "太宽": MAX_COLUMN_WIDTH},
        )

    def test_column_widths_written_as_a_list_fall_back_to_empty(self) -> None:
        config = AppConfig.from_dict(
            {**base_config(), "history_column_widths": [100, 200]}
        )

        self.assertEqual(config.history_column_widths, {})

    def test_broken_fields_are_reported_in_the_log(self) -> None:
        """回落要出声，否则用户只会觉得「我的设置莫名变了」。"""
        with self.assertLogs("models", level=logging.WARNING) as captured:
            AppConfig.from_dict({**base_config(), "playback_speed": "fast"})

        self.assertTrue(
            any("playback_speed" in message for message in captured.output),
            captured.output,
        )

    def test_broken_polygon_is_reported_once_not_per_vertex(self) -> None:
        """一个手改坏的多边形可能有几十个坏点，逐条刷屏会淹掉别的日志。"""
        with self.assertLogs("models", level=logging.WARNING) as captured:
            AppConfig.from_dict(
                {
                    **base_config(),
                    "zones": [
                        {
                            "name": "坏区域",
                            "polygon": [[1], "x", None, ["a", "b"], [float("inf"), 0]],
                        }
                    ],
                }
            )

        polygon_warnings = [
            message for message in captured.output if "多边形顶点" in message
        ]
        self.assertEqual(len(polygon_warnings), 1, captured.output)
        self.assertIn("5 个", polygon_warnings[0])

    def test_duplicate_zone_names_are_reported(self) -> None:
        """重名会让两个区域的会话与冷却按 (区域名, 目标ID) 串台。"""
        with self.assertLogs("models", level=logging.WARNING) as captured:
            config = AppConfig.from_dict(
                {
                    **base_config(),
                    "zones": [
                        {"name": "入口", "polygon": [[1, 2], [3, 4], [5, 6]]},
                        {"name": "入口", "polygon": [[7, 8], [9, 10], [11, 12]]},
                        {"name": "通道", "polygon": [[1, 2], [3, 4], [5, 6]]},
                    ],
                }
            )

        self.assertEqual(len(config.zones), 3)
        self.assertTrue(
            any("重名" in message for message in captured.output), captured.output
        )
        # 只报警告，不改用户起的名字。
        self.assertEqual([zone.name for zone in config.zones], ["入口", "入口", "通道"])

    def test_healthy_config_logs_no_warning(self) -> None:
        """兜底逻辑不该在健康配置上刷警告 —— 缺字段是常态，不是坏数据。"""
        records: list[logging.LogRecord] = []

        class Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        collector = Collector(level=logging.WARNING)
        models_logger = logging.getLogger("models")
        models_logger.addHandler(collector)
        self.addCleanup(models_logger.removeHandler, collector)

        AppConfig.from_dict(base_config())
        ZoneProfile.from_dict(ZoneProfile(name="空的").to_dict())

        self.assertEqual([record.getMessage() for record in records], [])


class ZoneProfileResilienceTests(unittest.TestCase):
    def test_profile_survives_broken_fields(self) -> None:
        profile = ZoneProfile.from_dict(
            {
                "name": "",
                "zones": [GOOD_ZONES[0], "不是对象", {"name": "坏", "polygon": "12"}],
                "description": 5,
                "version": "二",
            }
        )

        self.assertEqual(profile.name, "未命名配置组")
        self.assertEqual(profile.version, 1)
        self.assertEqual(profile.description, "5")
        self.assertEqual([zone.name for zone in profile.zones], ["卸货区", "坏"])

    def test_profile_round_trips_healthy_data(self) -> None:
        profile = ZoneProfile(
            name="夜间值守",
            description="入口",
            zones=[ZoneDefinition(name="入口", polygon=[[1, 2], [3, 4], [5, 6]], closed=True)],
        )

        restored = ZoneProfile.from_dict(profile.to_dict())

        self.assertEqual(restored.name, "夜间值守")
        self.assertEqual(restored.description, "入口")
        self.assertTrue(restored.zones[0].closed)
        self.assertEqual(restored.zones[0].polygon, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


class ConfigStoreResilienceTests(unittest.TestCase):
    def store(self) -> tuple[ConfigStore, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "config.json"
        return ConfigStore(path), path

    def test_missing_file_yields_defaults(self) -> None:
        store, _ = self.store()

        config = store.load()

        self.assertEqual(config.zones, [])
        self.assertEqual(config.inference_device, "cpu")

    def test_non_object_root_is_reported_as_value_error(self) -> None:
        """必须是个 ValueError：调用方接的是 ValueError，不是 AttributeError。"""
        store, path = self.store()
        path.write_text("[1, 2, 3]", encoding="utf-8")

        with self.assertRaises(ValueError):
            store.load()

    def test_unparsable_json_is_reported_as_value_error(self) -> None:
        store, path = self.store()
        path.write_text("{不是 json", encoding="utf-8")

        with self.assertRaises(ValueError):
            store.load()

    def test_quarantine_renames_the_file_so_the_next_start_succeeds(self) -> None:
        store, path = self.store()
        path.write_text("[1, 2, 3]", encoding="utf-8")

        backup = store.quarantine()

        self.assertIsNotNone(backup)
        self.assertFalse(path.exists())
        self.assertEqual(json.loads(backup.read_text(encoding="utf-8")), [1, 2, 3])
        # 坏文件挪走之后，同一份配置就能正常读出来了。
        self.assertEqual(store.load().zones, [])

    def test_quarantine_without_a_file_is_a_no_op(self) -> None:
        store, _ = self.store()

        self.assertIsNone(store.quarantine())

    def test_quarantine_keeps_the_original_content_for_manual_recovery(self) -> None:
        """备份不是删掉：用户很可能只是改错了一个字符，区域坐标还得能捞回来。"""
        store, path = self.store()
        path.write_text(json.dumps({**base_config(), "playback_speed": "fast"}), encoding="utf-8")

        backup = store.quarantine()

        self.assertIsNotNone(backup)
        restored = json.loads(backup.read_text(encoding="utf-8"))
        self.assertEqual(restored["active_profile"], "夜间值守")

    def test_partially_broken_config_still_loads_its_zones(self) -> None:
        """整份读得下来时不该走隔离路径：区域必须留在内存里，而不是回落默认值。"""
        store, path = self.store()
        path.write_text(
            json.dumps({**base_config(), "source_size": ["宽", "高"]}),
            encoding="utf-8",
        )

        config = store.load()

        self.assertEqual(len(config.zones), 1)
        self.assertEqual(config.zones[0].name, "卸货区")
        self.assertEqual(config.source_size, [])

    def test_round_trip_keeps_alarm_relevant_settings(self) -> None:
        store, _ = self.store()
        store.save(AppConfig.from_dict(base_config()))

        restored = store.load()

        self.assertEqual(restored.operation_mode, "video")
        self.assertEqual(restored.zones[0].cooldown_seconds, 30.0)
        self.assertEqual(restored.zones[0].polygon, [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        self.assertTrue(restored.cpu_low_power_preset)


class HealthCheckTests(unittest.TestCase):
    def test_a_healthy_config_round_trips_through_the_store(self) -> None:
        """上面全在测坏数据，这里确认好数据没被兜底逻辑改样。"""
        config = AppConfig.from_dict(base_config())

        restored = AppConfig.from_dict(config.to_dict())

        self.assertEqual(restored.to_dict(), config.to_dict())


class EventStatusLabelTests(unittest.TestCase):
    def test_status_label_is_chinese_for_every_state(self) -> None:
        event = AlarmEvent(
            source="0",
            zone_name="区域",
            track_id="1",
            entered_at_seconds=1.0,
            alarm_at_seconds=None,
            wall_time="2026-09-20 10:00:00",
        )

        self.assertEqual(event.status_label, "未触发报警")
        event.alarm_at_seconds = 2.0
        event.status = "alarmed"
        self.assertEqual(event.status_label, "已报警")
        event.status = "completed"
        self.assertEqual(event.status_label, "已结束")


if __name__ == "__main__":
    unittest.main()
