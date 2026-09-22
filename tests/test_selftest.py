"""`main.--selftest` 的回归测试。

自检是打包版唯一的自动验收手段，所以它自己**不能崩**：任何一步抛出异常都会让
报告一个字都写不出来，构建脚本只剩一个空的 selftest.log 和一个 PyInstaller 模态
对话框 —— 恰恰在最需要诊断信息的时候什么都没有。真实事故：ultralytics 的
models/yolo/semantic/train.py 顶层 `import matplotlib.pyplot`，被 spec 排掉后
`import main_window` 当场炸掉，弹窗还把 -Wait 的构建脚本堵死了。

这里用假模块替掉整条重型依赖链（真货要 800 MB、二十秒），只验自检自己的行为。
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import app_paths
import app_version
import asset_manifest
import main

# 自检会 import 的重型模块。测试里一律替成假货，跑得快，也和真环境解耦。
_HEAVY_MODULES = (
    "numpy",
    "cv2",
    "torch",
    "PySide6",
    "PySide6.QtWidgets",
    "ultralytics",
    "supervision",
    "trackers",
    "scipy",
    "scipy.interpolate",
    "openvino",
    "matplotlib",
)

# 自检的探针步骤各自需要的模块与属性。
_PROBE_MODULES = ("main_window", "alarm_service", "inference_profiles", "compute_devices")


class SelfTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

        # data_dir() 有模块级缓存，不清会把上一个测试的临时目录带进来。
        app_paths.data_dir.cache_clear()
        self.addCleanup(app_paths.data_dir.cache_clear)
        patcher = patch.object(app_paths, "APP_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    @property
    def report_path(self) -> Path:
        return self.root / "logs" / "selftest.log"

    def run_selftest(
        self, modules: dict[str, object], argv: list[str] | None = None
    ) -> tuple[int, str]:
        # _run_selftest 会把报告 print 一份到 stdout（打包版靠它，这里只是噪音）。
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            exit_code = main._run_selftest(argv or [])
        return exit_code, self.report_path.read_text(encoding="utf-8")


def _healthy_modules(root: Path) -> dict[str, object]:
    """一整套「什么都正常」的假模块。"""
    modules: dict[str, object] = {name: types.ModuleType(name) for name in _HEAVY_MODULES}

    openvino = modules["openvino"]
    openvino.__version__ = "2024.6.0-fake"  # type: ignore[attr-defined]

    # 取证截图探针会真编码一张 JPEG 并写盘（它要验的正是"写不写得进去"），
    # 所以假 cv2/numpy 得提供它用到的那两个接口。
    class _EncodedBuffer:
        def tobytes(self) -> bytes:
            return b"\xff\xd8fake-jpeg"

    modules["numpy"].zeros = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
    modules["numpy"].uint8 = "uint8"  # type: ignore[attr-defined]
    modules["cv2"].imencode = lambda extension, image: (True, _EncodedBuffer())  # type: ignore[attr-defined]

    # 自检会核对缓存目录是否落在程序数据目录里，假货得给出那个位置。
    mpl_cache = root / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    modules["matplotlib"].get_cachedir = lambda: str(mpl_cache)  # type: ignore[attr-defined]

    main_window = types.ModuleType("main_window")
    main_window.CONFIG_PATH = root / "config.json"  # type: ignore[attr-defined]
    main_window.EVENTS_DIR = root / "events"  # type: ignore[attr-defined]
    main_window.PROFILES_DIR = root / "profiles"  # type: ignore[attr-defined]
    modules["main_window"] = main_window

    wav = root / "assets" / "warming_converted.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(b"RIFF")

    class _AlarmPlayer:
        warning_audio_path = wav

    alarm_service = types.ModuleType("alarm_service")
    alarm_service.AlarmPlayer = _AlarmPlayer  # type: ignore[attr-defined]
    modules["alarm_service"] = alarm_service

    model_dir = root / "models" / "yolo11n_int8_openvino_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = root / "models" / "yolo11n.pt"
    weights.write_bytes(b"weights")

    # 资源清单要和这两个假文件对得上，否则「一切正常」的假环境自己就不健康了。
    write_manifest(root, {"yolo11n.pt": weights, "warming_converted.wav": wav})

    inference_profiles = types.ModuleType("inference_profiles")
    inference_profiles.LOW_POWER_MODEL_DIR = model_dir  # type: ignore[attr-defined]
    modules["inference_profiles"] = inference_profiles

    compute_devices = types.ModuleType("compute_devices")
    compute_devices.enumerate_inference_devices = lambda: [("cpu", "CPU")]  # type: ignore[attr-defined]
    modules["compute_devices"] = compute_devices

    return modules


def write_manifest(root: Path, assets: dict[str, Path], *, digest: str | None = None) -> Path:
    """按给定资源写出清单。``digest`` 给定时用它当哈希（用来造出「对不上」的情况）。"""
    entries = []
    for name, path in assets.items():
        candidates = (
            [f"models/{name}", name] if name.endswith(".pt") else [f"assets/{name}", name]
        )
        entries.append(
            {
                "name": name,
                "candidates": candidates,
                "bytes": path.stat().st_size,
                "sha256": digest or hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = root / "asset_manifest.json"
    manifest.write_text(
        json.dumps({"version": 1, "assets": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _add_policy_resolution(
    modules: dict[str, object],
    *,
    low_power_available: bool = True,
) -> list[tuple[str, bool]]:
    """给假的 inference_profiles 补上 resolve_inference_policy，并记下它被怎么调的。

    返回的列表按调用顺序收集 ``(device, low_power)`` —— 深度自检必须把标准和低功耗
    两条路都走一遍，只走一条的话「打包版能不能推理」这个问题只答了一半。
    """
    calls: list[tuple[str, bool]] = []

    def resolve(model_path: str, device: str, low_power: bool) -> object:
        calls.append((device, low_power))
        if low_power and not low_power_available:
            return types.SimpleNamespace(policy=None, unavailable_reason="缺少 INT8 模型目录")
        return types.SimpleNamespace(
            policy=types.SimpleNamespace(model_path=model_path, device=device, label="假预设"),
            unavailable_reason=None,
        )

    modules["inference_profiles"].resolve_inference_policy = resolve  # type: ignore[union-attr]
    return calls


class HealthyRunTests(SelfTestBase):
    def test_reports_pass_and_returns_zero(self) -> None:
        exit_code, report = self.run_selftest(_healthy_modules(self.root))

        self.assertEqual(exit_code, 0)
        self.assertIn("RESULT: PASS", report)
        self.assertNotIn("FAIL", report)

    def test_writes_report_under_data_dir(self) -> None:
        self.run_selftest(_healthy_modules(self.root))

        # 构建脚本按这个固定位置去读，挪了它验收就瞎了。
        self.assertTrue(self.report_path.is_file())


class FailingProbeTests(SelfTestBase):
    def test_broken_main_window_is_reported_instead_of_raised(self) -> None:
        # sys.modules 里放 None 会让 `import main_window` 抛 ImportError —— 正是
        # matplotlib 那次事故的形状。自检必须把它记成失败，而不是让它逃出去。
        modules = _healthy_modules(self.root)
        modules["main_window"] = None

        exit_code, report = self.run_selftest(modules)

        self.assertEqual(exit_code, 1)
        self.assertIn("RESULT: FAIL", report)
        self.assertIn("import main_window", report)

    def test_later_probes_still_run_after_an_early_failure(self) -> None:
        # 一步失败不该中断后面的检查，否则一次只能暴露一个问题，排查要来回好几轮。
        modules = _healthy_modules(self.root)
        modules["main_window"] = None

        _, report = self.run_selftest(modules)

        self.assertIn("devices", report)
        self.assertIn("openvino int8 dir", report)

    def test_every_probe_broken_still_produces_a_report(self) -> None:
        modules = _healthy_modules(self.root)
        for name in _PROBE_MODULES:
            modules[name] = None

        exit_code, report = self.run_selftest(modules)

        self.assertEqual(exit_code, 1)
        self.assertIn("RESULT: FAIL", report)
        self.assertTrue(self.report_path.is_file())

    def test_asset_hash_mismatch_is_a_failure_with_a_fix_hint(self) -> None:
        """资源被换掉或拷坏时，低功耗模型之外的这两个文件以前是没人管的。"""
        modules = _healthy_modules(self.root)
        write_manifest(
            self.root,
            {
                "yolo11n.pt": self.root / "models" / "yolo11n.pt",
                "warming_converted.wav": self.root / "assets" / "warming_converted.wav",
            },
            digest="0" * 64,
        )

        exit_code, report = self.run_selftest(modules)

        self.assertEqual(exit_code, 1)
        self.assertIn("与清单不一致", report)
        # 修复建议必须是现场能执行的那条：打包版没有 Python，tools\ 也不随包发布。
        self.assertIn("--write-asset-manifest", report)

    def test_missing_manifest_is_a_failure(self) -> None:
        modules = _healthy_modules(self.root)
        (self.root / "asset_manifest.json").unlink()

        exit_code, report = self.run_selftest(modules)

        self.assertEqual(exit_code, 1)
        self.assertIn("缺少资源清单", report)

    def test_version_drift_between_code_and_version_info_is_a_failure(self) -> None:
        """「exe 属性写着 1.0.0、日志里写着 1.1.0」正是双份版本号最容易出的岔子。"""
        modules = _healthy_modules(self.root)

        with patch.object(app_version, "VERSION", "9.9.9"):
            exit_code, report = self.run_selftest(modules)

        self.assertEqual(exit_code, 1)
        self.assertIn("版本号不一致", report)

    def test_healthy_version_matches_the_packaging_resource(self) -> None:
        """app_version.VERSION 与 execode/version_info.txt 必须一致，这里钉住它。"""
        modules = _healthy_modules(self.root)

        exit_code, report = self.run_selftest(modules)

        self.assertEqual(exit_code, 0, report)
        self.assertIn(app_version.VERSION, report)

    def test_missing_resources_are_reported_as_failures(self) -> None:
        modules = _healthy_modules(self.root)
        (self.root / "models" / "yolo11n.pt").unlink()

        exit_code, report = self.run_selftest(modules)

        self.assertEqual(exit_code, 1)
        self.assertIn("检测模型缺失", report)

    def test_matplotlib_cache_escaping_to_temp_is_a_failure(self) -> None:
        # PyInstaller 自带的 pyi_rth_mplconfig 会把 MPLCONFIGDIR 指向 %TEMP% 下一个
        # 每次都换的目录，覆盖掉我们自己的设置。那次就是这么溜过去的：字体缓存每次
        # 重建，进程被强杀时还在 %TEMP% 留下目录。自检现在必须拦住它。
        modules = _healthy_modules(self.root)
        modules["matplotlib"].get_cachedir = lambda: tempfile.gettempdir()  # type: ignore[attr-defined]

        with patch.object(app_paths, "IS_FROZEN", True):
            exit_code, report = self.run_selftest(modules)

        self.assertEqual(exit_code, 1)
        self.assertIn("matplotlib 缓存目录跑到了", report)

    def test_matplotlib_cache_is_not_checked_in_source_mode(self) -> None:
        # 源码模式用 ~/.matplotlib 是开发机上正常行为，不该报失败。
        modules = _healthy_modules(self.root)
        modules["matplotlib"].get_cachedir = lambda: tempfile.gettempdir()  # type: ignore[attr-defined]

        exit_code, report = self.run_selftest(modules)

        self.assertEqual(exit_code, 0)
        self.assertIn("RESULT: PASS", report)


class WriteAssetManifestTests(unittest.TestCase):
    """``--write-asset-manifest``：现场替换资源后重新生成清单的那条路。

    它必须由 exe 自己能做 —— 说明书里写明权重与告警音可以替换，而打包版没有 Python
    环境、tools\\ 也不随包发布，否则自检给出的修复建议是用户执行不了的。
    """

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        app_paths.data_dir.cache_clear()
        self.addCleanup(app_paths.data_dir.cache_clear)
        patcher = patch.object(app_paths, "APP_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

        (self.root / "models").mkdir(parents=True)
        (self.root / "assets").mkdir(parents=True)
        (self.root / "models" / "yolo11n.pt").write_bytes(b"weights")
        (self.root / "assets" / "warming_converted.wav").write_bytes(b"RIFF")
        # 清单放在 assets\ 下（打包版的布局），这样才验得出「重新生成时覆盖的是
        # 程序实际在用的那一份」，而不是在旁边新写一份、让旧的继续生效。
        self.manifest = self.root / "assets" / "asset_manifest.json"
        asset_manifest.write_manifest(self.manifest)

    def test_regenerates_the_manifest_after_an_asset_is_replaced(self) -> None:
        (self.root / "models" / "yolo11n.pt").write_bytes(b"my-own-weights")
        checks, failures = asset_manifest.verify_manifest()
        self.assertTrue(failures, "替换后应当先判失败")

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = main._write_asset_manifest()

        self.assertEqual(exit_code, 0)
        checks, failures = asset_manifest.verify_manifest()
        self.assertEqual(failures, [])
        self.assertTrue(all(check.ok for check in checks))

    def test_reports_a_missing_asset_instead_of_writing_a_half_manifest(self) -> None:
        (self.root / "models" / "yolo11n.pt").unlink()

        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            exit_code = main._write_asset_manifest()

        self.assertEqual(exit_code, 1)
        self.assertIn("yolo11n.pt", stderr.getvalue())

    def test_writes_back_to_the_manifest_the_program_actually_reads(self) -> None:
        """清单在 assets\\ 下时，重新生成必须覆盖那一份。

        写到旁边会留下一份旧清单，而查找顺序仍然会先读到旧的 —— 于是「重新生成过了」
        和「自检还是失败」同时成立。
        """
        (self.root / "models" / "yolo11n.pt").write_bytes(b"new-weights")

        with contextlib.redirect_stdout(io.StringIO()):
            main._write_asset_manifest()

        checks, failures = asset_manifest.verify_manifest()
        self.assertEqual(failures, [])
        # 写的是校验时实际读的那一份（assets\ 下），不是旁边新写一份。
        self.assertTrue(self.manifest.is_file())
        self.assertEqual(asset_manifest.manifest_path(), self.manifest)
        self.assertTrue(all(check.ok for check in checks))

    def test_flag_is_handled_before_any_gui_starts(self) -> None:
        """现场跑这条命令时不该弹出窗口、也不该去 import PySide6。"""
        # main() 会装一个「未捕获异常就弹窗」的 excepthook，那是**进程级**的全局状态：
        # 不还原的话，后面任何一个在事件循环里抛异常的测试都会走进那个模态弹窗，
        # 然后永远卡在那里（`test_touch_and_wheel` 就是这么挂住的 —— 栈里是
        # main._hook → QMessageBox.exec() 被 processEvents 反复重入）。
        self.addCleanup(setattr, sys, "excepthook", sys.excepthook)
        with (
            patch.object(sys, "argv", ["main.py", "--write-asset-manifest"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exit_code = main.main()

        self.assertEqual(exit_code, 0)


class DeepSelfTestTests(SelfTestBase):
    """``--deep`` 分支 —— 真跑推理那条路。

    这里当然不载真权重（要 800 MB 依赖、几分钟），验的是**外壳**：该跳的时候跳、
    该记成失败的时候记、异常一律不许逃出去。真实推理本身由构建脚本的
    ``-DeepTestVideo`` 在打包版上验，那是唯一能证明冻结环境能推理的地方。
    """

    def test_deep_probes_do_not_run_without_the_flag(self) -> None:
        # 默认那一遍要保持 9 秒量级，每次构建都跑得起。深度探针漏进来就不是这个成本了。
        modules = _healthy_modules(self.root)

        with (
            patch.object(main, "_deep_probe_policy") as policy_probe,
            patch.object(main, "_deep_probe_audio") as audio_probe,
        ):
            exit_code, report = self.run_selftest(modules)

        self.assertEqual(exit_code, 0)
        policy_probe.assert_not_called()
        audio_probe.assert_not_called()
        self.assertNotIn("深度自检", report)

    def test_deep_run_exercises_both_inference_paths(self) -> None:
        modules = _healthy_modules(self.root)
        calls = _add_policy_resolution(modules)
        video = self.root / "clip.mp4"
        video.write_bytes(b"fake")

        with (
            patch.object(main, "_deep_probe_policy") as policy_probe,
            patch.object(main, "_deep_probe_audio") as audio_probe,
        ):
            exit_code, report = self.run_selftest(modules, ["--deep", "--video", str(video)])

        self.assertEqual(exit_code, 0)
        self.assertIn("RESULT: PASS", report)
        audio_probe.assert_called_once()
        self.assertEqual(calls, [("cpu", False), ("cpu", True)])
        self.assertEqual(
            [call.args[0] for call in policy_probe.call_args_list],
            ["standard", "lowpower"],
        )
        # 视频要原样传到底 —— 中途丢掉就悄悄退化成合成帧，告警链路一行没验。
        for call in policy_probe.call_args_list:
            self.assertEqual(call.args[3], video)

    def test_missing_weights_skip_inference_instead_of_downloading(self) -> None:
        # 权重不在时 YOLO(path) 会让 ultralytics **联网抓** yolo11n.pt：自检可能
        # "通过"，但过的是一个刚从网上下来的权重，跟发布包里的无关；而在断网的验收
        # 机上只会挂在超时里。所以这里必须硬停。
        modules = _healthy_modules(self.root)
        _add_policy_resolution(modules)
        (self.root / "models" / "yolo11n.pt").unlink()

        with (
            patch.object(main, "_deep_probe_policy") as policy_probe,
            patch.object(main, "_deep_probe_audio"),
        ):
            exit_code, report = self.run_selftest(modules, ["--deep"])

        self.assertEqual(exit_code, 1)
        policy_probe.assert_not_called()
        self.assertIn("不让 ultralytics 联网下载", report)

    def test_missing_video_is_a_failure_not_a_silent_downgrade(self) -> None:
        # 静默降级成合成帧最坑：报告照样 PASS，而真正想验的告警链路一行没跑，
        # 看报告的人以为验过了。
        modules = _healthy_modules(self.root)
        _add_policy_resolution(modules)

        with (
            patch.object(main, "_deep_probe_policy"),
            patch.object(main, "_deep_probe_audio"),
        ):
            exit_code, report = self.run_selftest(
                modules, ["--deep", "--video", str(self.root / "nope.mp4")]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("nope.mp4", report)

    def test_video_flag_without_a_path_is_a_failure(self) -> None:
        modules = _healthy_modules(self.root)
        _add_policy_resolution(modules)

        with (
            patch.object(main, "_deep_probe_policy"),
            patch.object(main, "_deep_probe_audio"),
        ):
            exit_code, report = self.run_selftest(modules, ["--deep", "--video"])

        self.assertEqual(exit_code, 1)
        self.assertIn("--video 后面没跟路径", report)

    def test_unavailable_low_power_preset_is_recorded_not_failed(self) -> None:
        # 使用说明里写明缺 INT8 模型目录则该模式不可用，那不算构建失败；但原因要
        # 原样打出来，别变成一行沉默的跳过。
        modules = _healthy_modules(self.root)
        _add_policy_resolution(modules, low_power_available=False)

        with (
            patch.object(main, "_deep_probe_policy") as policy_probe,
            patch.object(main, "_deep_probe_audio"),
        ):
            exit_code, report = self.run_selftest(modules, ["--deep"])

        self.assertEqual(exit_code, 0)
        self.assertIn("缺少 INT8 模型目录", report)
        self.assertEqual([call.args[0] for call in policy_probe.call_args_list], ["standard"])

    def test_broken_policy_resolution_is_reported_instead_of_raised(self) -> None:
        # 回归测试：resolve_inference_policy 的 import 与调用一度写在 step() 外面。
        # 它一样会失败（被 spec 的 excludes 误伤、OpenVINO 的 runtime DLL 没收进来
        # 都会），那时异常会直接逃出 _run_selftest —— 一行报告都写不出来，正是本模块
        # 开头那起事故的形状。这里的假 inference_profiles 刻意不带这个函数。
        modules = _healthy_modules(self.root)

        with patch.object(main, "_deep_probe_audio"):
            exit_code, report = self.run_selftest(modules, ["--deep"])

        self.assertEqual(exit_code, 1)
        self.assertIn("深度推理", report)
        self.assertTrue(self.report_path.is_file())

    def test_audio_failure_does_not_stop_the_inference_probes(self) -> None:
        modules = _healthy_modules(self.root)
        _add_policy_resolution(modules)

        with (
            patch.object(main, "_deep_probe_policy") as policy_probe,
            patch.object(main, "_deep_probe_audio", side_effect=RuntimeError("没有音频设备")),
        ):
            exit_code, report = self.run_selftest(modules, ["--deep"])

        self.assertEqual(exit_code, 1)
        self.assertIn("告警音", report)
        # 没声卡的机器上照样要把推理验完，否则一台机器只能得出半个结论。
        self.assertEqual(len(policy_probe.call_args_list), 2)


class ImageWriteProbeTests(unittest.TestCase):
    """取证截图写入探针。

    它要挡住的是最阴的一类故障：**写图失败但没有任何报错**。裸 cv2.imwrite 在非 ASCII
    路径下会返回 False 且什么都不写，而发布包的目录名就是中文 —— 一旦哪天第三方库不再
    打那个 Unicode 补丁，打包版里每一次取证截图都会写不进去，而界面上只显示"报警取证
    不可用"。这个探针在每次构建的浅自检里真写一张 JPEG，把这件事提前暴露。
    """

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_probe(self, target: Path) -> tuple[list[str], str]:
        records: list[str] = []
        failures: list[str] = []
        with patch.object(app_paths, "data", return_value=target):
            main._probe_image_write(
                lambda label, value: records.append(f"{label}={value}"), failures
            )
        return failures, "\n".join(records)

    def test_passes_on_a_plain_path_and_cleans_up_after_itself(self) -> None:
        failures, report = self.run_probe(self.root / "logs" / "probe.jpg")

        self.assertEqual(failures, [])
        self.assertIn("ok", report)
        # 探针不该在数据目录里留下垃圾。
        self.assertEqual(list((self.root / "logs").glob("*.jpg")), [])

    def test_passes_on_a_non_ascii_path(self) -> None:
        """发布包目录名是中文，这条路径必须验到。"""
        failures, report = self.run_probe(
            self.root / "行人警戒区域监控" / "logs" / "probe.jpg"
        )

        self.assertEqual(failures, [], report)
        self.assertIn("ok", report)

    def test_reports_failure_instead_of_raising(self) -> None:
        """写不进去要记成失败，而不是让自检自己炸掉（那样报告一个字都写不出来）。"""
        blocked = self.root / "blocked"
        blocked.write_text("我不是目录", encoding="utf-8")

        failures, report = self.run_probe(blocked / "logs" / "probe.jpg")

        self.assertTrue(failures)
        self.assertIn("FAIL", report)


class StartupBannerTests(unittest.TestCase):
    """启动横幅：现场支持的第一句话就是「你装的是哪一版、配置在哪」。

    它同时钉住一个真发生过的 bug：横幅里用了 app_paths 却没在函数内导入（导入写在
    调用方 main() 里），于是每次启动都会 NameError —— 而自检不走这条路径，跑测试
    也发现不了，是 ruff 的 F821 抓出来的。
    """

    def test_banner_logs_version_and_paths(self) -> None:
        with self.assertLogs("main", level="INFO") as captured:
            main._log_startup_banner()

        text = "\n".join(captured.output)
        self.assertIn(app_version.VERSION, text)
        self.assertIn(str(app_paths.APP_DIR), text)

    def test_banner_does_not_raise_in_source_mode(self) -> None:
        """不校验日志内容，只要求它能跑完：NameError 会在这里现形。"""
        with self.assertLogs("main", level="INFO"):
            main._log_startup_banner()


if __name__ == "__main__":
    unittest.main()
