from __future__ import annotations

import json
import logging
import os
import re
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from logging_config import configure_logging

if TYPE_CHECKING:
    from inference_profiles import InferencePolicy

logger = logging.getLogger(__name__)


def _install_excepthook() -> None:
    """打包版是 --noconsole 的，未捕获异常会静默吞掉，这里补上日志与弹窗。"""

    def _hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical(
            "未捕获的异常", exc_info=(exc_type, exc_value, exc_traceback)
        )
        detail = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            if QApplication.instance() is not None:
                box = QMessageBox()
                box.setIcon(QMessageBox.Icon.Critical)
                box.setWindowTitle("程序出错")
                box.setText(f"发生未处理的错误：{exc_value}")
                box.setInformativeText("详细信息已写入 logs/app.log。")
                box.setDetailedText(detail)
                box.exec()
        except Exception:
            # 弹窗失败不能盖掉原始异常，日志里已经记下了。
            logger.exception("显示错误弹窗失败")

    sys.excepthook = _hook


def _redirect_matplotlib_cache() -> None:
    """把 matplotlib 的字体缓存从每次一换的临时目录挪回程序自己的数据目录。

    matplotlib 是被 ultralytics 硬拖进来的依赖（models/yolo/semantic/train.py 顶层
    `import matplotlib.pyplot`），本程序一张图都不画，但它每次启动都要扫一遍系统字体
    并写 120 KB 的 fontlist 缓存。

    为什么不在 execode/runtime_hook.py 里做：PyInstaller 自带的 pyi_rth_mplconfig
    在自定义钩子**之后**执行，且无条件 `os.environ['MPLCONFIGDIR'] = mkdtemp()`，
    写在钩子里必被覆盖。它那个理由只对 onefile 成立 —— 解包目录 _MEIxxxx 每次都换，
    缓存里的字体路径会指向已删除的目录；而 onedir 的 _internal/matplotlib/mpl-data
    路径是稳定的，隔离掉纯属白扔掉一次字体扫描，还会在 %TEMP% 留下目录（进程被强杀
    时它注册的 atexit 清理不会跑）。

    只在冻结模式下改：源码模式用 ~/.matplotlib 是开发机上正常的行为。
    """
    if not getattr(sys, "frozen", False):
        return
    try:
        import app_paths

        cache_dir = app_paths.data("matplotlib")
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        # 拿不到可写目录就把 PyInstaller 的临时目录留着用，功能不受影响。
        logger.warning("无法准备 matplotlib 缓存目录，沿用默认位置: %s", error)
        return
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


# 深度自检最多推这么多帧。这里验的是「链路通不通」，不是跑完整段视频 —— 命中告警
# 就提前跳出，正常远用不到上限；给上限只是防着「一个人都没检出」时无限跑下去。
_DEEP_MAX_FRAMES = 300
# 自检专用的驻留阈值，比默认的 2 秒短：省几秒推理，而驻留判定本身照样被走到。
_DEEP_DWELL_SECONDS = 1.0


def _parse_video_argument(
    argv: list[str],
    record: Callable[[str, object], None],
    failures: list[str],
) -> Path | None:
    """取 ``--video <路径>``。

    给了却找不到就记成失败 —— 静默降级成合成帧最坑：报告照样 PASS，而真正想验的
    告警链路一行没跑，看报告的人以为验过了。
    """
    if "--video" not in argv:
        return None
    index = argv.index("--video")
    if index + 1 >= len(argv):
        record("--video", "FAIL 后面没跟路径")
        failures.append("--video 后面没跟路径")
        return None
    video_path = Path(argv[index + 1])
    record("视频素材", f"{video_path} | exists={video_path.is_file()}")
    if not video_path.is_file():
        failures.append(f"--video 指定的文件不存在: {video_path}")
        return None
    return video_path


def _deep_probe_policy(
    slug: str,
    label: str,
    policy: "InferencePolicy",
    video_path: Path | None,
    record: Callable[[str, object], None],
    failures: list[str],
) -> None:
    """把一条推理路径**真的跑起来**：载权重、推真帧、让告警落盘。

    为什么非要有这一步：`--selftest` 原本只 import 模块、`exists()` 查文件，那证明
    不了打包版能推理。冻结环境里真正会崩的恰恰是这两件事 ——
      * 载权重：torch 的 C 扩展、OpenVINO 的 plugins.xml 与 runtime DLL，靠
        collect_all 抓来的东西，`import openvino` 成功 ≠ INT8 模型能载起来；
      * 落盘：截图与 JSONL 的路径解析，源码模式下相对 cwd 也能跑通，打包后未必。
    两者都只有真跑一遍才现形，所以这个探针刻意不用 mock。
    """
    import shutil
    import time

    import cv2
    import numpy as np

    import app_paths
    from alarm_service import EventStore
    from detection_engine import DetectionEngine
    from models import ZoneDefinition

    started = time.perf_counter()
    engine = DetectionEngine(policy.model_path, [], policy.device, policy)
    record(f"{label} 载入权重", f"{time.perf_counter() - started:.1f}s | {policy.label}")

    if video_path is None:
        # 没给视频就退到合成帧。这只能证明前向跑得通，证不了告警链路 —— 说清楚，
        # 别让一行 ok 看着像「全都验过了」。
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        started = time.perf_counter()
        engine.process(frame, 0.0, "synthetic", "video")
        record(
            f"{label} 合成帧前向",
            f"{time.perf_counter() - started:.2f}s（未验告警链路：没有视频素材，加 --video 才验）",
        )
        return

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        record(f"{label} 视频", f"FAIL 打不开 {video_path}")
        failures.append(f"{label}: 打不开视频 {video_path}")
        return

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 拿整幅画面当警戒区：这里要验的是链路，不是区域画得准不准。用真实 config.json
    # 里那个区域的话，人没走进去就白跑一趟，还会被误读成「检测不出人」。
    engine.update_zones(
        [
            ZoneDefinition(
                name="自检全画面区域",
                polygon=[
                    [0.0, 0.0],
                    [float(width), 0.0],
                    [float(width), float(height)],
                    [0.0, float(height)],
                ],
                closed=True,
                dwell_seconds=_DEEP_DWELL_SECONDS,
            )
        ]
    )

    # 每条路径一个独立目录，且每次跑之前清空 —— 否则上一次的记录会让「本次告警
    # 落盘了吗」这个判断永远为真。刻意不写进正式的 events\，免得污染取证数据。
    events_root = app_paths.data("selftest-events", slug)
    shutil.rmtree(events_root, ignore_errors=True)
    store = EventStore(events_root)

    counts = {"entered": 0, "alarmed": 0, "exited": 0}
    frames = 0
    frames_with_person = 0
    inference_seconds = 0.0
    try:
        while frames < _DEEP_MAX_FRAMES:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            video_time = frames / fps
            tick = time.perf_counter()
            _, transitions = engine.process(frame, video_time, str(video_path), "video")
            inference_seconds += time.perf_counter() - tick
            if engine.active_sessions:
                frames_with_person += 1
            for transition in transitions:
                counts[transition.kind] = counts.get(transition.kind, 0) + 1
                if transition.kind == "entered":
                    store.open_session(transition.event, frame)
                elif transition.kind == "alarmed":
                    store.mark_alarmed(transition.event, frame)
                else:
                    store.close_session(transition.event)
            if counts["alarmed"]:
                break
    finally:
        capture.release()

    speed = frames / inference_seconds if inference_seconds > 0 else 0.0
    record(f"{label} 推理", f"{frames} 帧 / {inference_seconds:.1f}s = {speed:.1f} fps")
    record(
        f"{label} 闯入判定",
        f"进入 {counts['entered']} 次，告警 {counts['alarmed']} 次，区内帧 {frames_with_person}/{frames}",
    )
    if not counts["entered"]:
        failures.append(f"{label}: {frames} 帧里一个人都没检出，权重或推理后端有问题")
    elif not counts["alarmed"]:
        failures.append(f"{label}: 检出了人却没触发告警，驻留判定有问题")

    screenshots = sorted((events_root / "screenshots").glob("*.jpg"))
    record(
        f"{label} 取证落盘",
        f"{events_root} | jsonl={store.log_path.is_file()} "
        f"会话 {len(store.load_recent())} 条，截图 {len(screenshots)} 张",
    )
    if counts["alarmed"] and (not store.log_path.is_file() or not screenshots):
        failures.append(f"{label}: 告警触发了但取证没落盘（{events_root}）")


def _deep_probe_audio(record: Callable[[str, object], None]) -> None:
    """真的放一次告警音。

    不能走 AlarmPlayer._play() —— 它把 ImportError/OSError/RuntimeError 全吞成一条
    warning（这在运行期是对的，告警放不出来不该弄崩程序），但拿它做探针就等于什么
    都没验。所以这里直连 winsound，让异常冒出来，交给外层的 step() 记成失败。
    """
    import wave

    import winsound

    from alarm_service import AlarmPlayer

    wav = AlarmPlayer().warning_audio_path
    with wave.open(str(wav), "rb") as clip:
        record(
            "告警音格式",
            f"{clip.getnchannels()} 声道 / {clip.getsampwidth() * 8} bit / "
            f"{clip.getframerate()} Hz / {clip.getnframes() / clip.getframerate():.1f}s",
        )
    winsound.PlaySound(str(wav), winsound.SND_FILENAME | winsound.SND_NODEFAULT)
    record("告警音播放", "ok（PlaySound 同步播完未报错）")


def _probe_assets(
    record: Callable[[str, object], None],
    failures: list[str],
) -> None:
    """按清单校验发布资源（权重、告警音）。

    低功耗的 OpenVINO 模型一直有 sha256 校验，而这两个文件没有：被换掉、拷坏、解压
    截断，程序都照样加载，只是「检测不出人」或「报警没声音」—— 都不报错，只让人以为
    程序坏了。这里补上，让「我换了个权重」和「权重被悄悄改坏」都必须被看见。

    资源在说明里是允许用户替换的，替换后需要重新生成清单
    （``tools\\write_asset_manifest.py``），否则这里会失败 —— 那是刻意的。
    """
    import hashlib

    import app_paths

    manifest_path = app_paths.resource("assets/asset_manifest.json", "asset_manifest.json")
    if not manifest_path.is_file():
        failures.append(f"缺少资源清单: {manifest_path}（用 tools\\write_asset_manifest.py 生成）")
        record("资源清单", f"FAIL 不存在 {manifest_path}")
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest["assets"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(f"资源清单无法解析: {error!r}")
        record("资源清单", f"FAIL {error!r}")
        return

    for entry in entries:
        name = str(entry.get("name", "?"))
        path = app_paths.resource(*[str(item) for item in entry.get("candidates", [name])])
        if not path.is_file():
            failures.append(f"发布资源缺失: {name}（期望位置 {path}）")
            record(f"资源 {name}", f"FAIL 不存在 {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = str(entry.get("sha256", ""))
        size = path.stat().st_size
        if digest != expected:
            failures.append(
                f"发布资源与清单不一致: {name}（{path}）"
                "。确认替换无误后运行 tools\\write_asset_manifest.py 重新生成清单"
            )
            record(f"资源 {name}", f"FAIL 哈希不符 {digest[:16]}…")
            continue
        record(f"资源 {name}", f"ok | {size} 字节 | sha256 {digest[:16]}…")


def _probe_version(
    record: Callable[[str, object], None],
    failures: list[str],
) -> None:
    """比对运行期版本号与 exe 版本资源。

    ``app_version.VERSION`` 与 ``execode/version_info.txt`` 是两份东西（后者只在打包
    时被 PyInstaller 读走），所以必须有一处比对，否则「exe 属性写着 1.0.0、日志里写着
    1.1.0」这种岔子只有交付之后才会被发现。
    """
    import app_paths
    import app_version

    record("版本", app_version.VERSION)

    if app_paths.IS_FROZEN:
        packaged = _packaged_version()
        if packaged is None:
            record("exe 版本资源", "无法读取（跳过比对）")
            return
        record("exe 版本资源", packaged)
        if packaged != app_version.VERSION:
            failures.append(
                f"版本号不一致: exe 里是 {packaged}，代码里是 {app_version.VERSION}"
            )
        return

    version_file = Path(__file__).resolve().parent / "execode" / "version_info.txt"
    if not version_file.is_file():
        record("version_info.txt", "缺失（源码模式下跳过比对）")
        return
    match = re.search(r"filevers=\(([^)]*)\)", version_file.read_text(encoding="utf-8"))
    if match is None:
        record("version_info.txt", "FAIL 找不到 filevers")
        failures.append(f"{version_file} 里找不到 filevers")
        return
    parts = [item.strip() for item in match.group(1).split(",")]
    declared = ".".join(parts[:3])
    record("version_info.txt", declared)
    if declared != app_version.VERSION:
        failures.append(
            f"版本号不一致: version_info.txt 里是 {declared}，"
            f"app_version.VERSION 是 {app_version.VERSION}"
        )


def _packaged_version() -> str | None:
    """读 exe 自己的版本资源（FileVersion）。读不到就返回 None，不判失败。"""
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None
    try:
        path = sys.executable
        size = ctypes.windll.version.GetFileVersionInfoSizeW(path, None)  # type: ignore[attr-defined]
        if not size:
            return None
        buffer = ctypes.create_string_buffer(size)
        if not ctypes.windll.version.GetFileVersionInfoW(path, 0, size, buffer):  # type: ignore[attr-defined]
            return None
        value = ctypes.c_void_p()
        length = wintypes.UINT()
        if not ctypes.windll.version.VerQueryValueW(  # type: ignore[attr-defined]
            buffer, "\\", ctypes.byref(value), ctypes.byref(length)
        ):
            return None
        fixed = ctypes.cast(value, ctypes.POINTER(ctypes.c_uint32 * 4)).contents
        return f"{fixed[0]}.{fixed[1]}.{fixed[2]}"
    except Exception:  # noqa: BLE001 - 读不到版本资源不该让自检失败
        logger.exception("读取 exe 版本资源失败")
        return None


def _run_selftest(argv: list[str]) -> int:
    """不建窗口地自检一遍依赖与资源路径，供打包后的构建脚本验收。

    窗口化 exe 没有 stdout 可看，所以结果同时写 logs/selftest.log。

    默认这一遍只 import 模块、查资源在不在 —— 快（约 9 秒）、不需要素材，适合每次
    构建都跑。加 ``--deep`` 会**真的**载权重、推真帧、让告警落盘，两条推理路径
    （标准 与 CPU 低功耗 OpenVINO INT8）各跑一遍，再放一次告警音::

        行人警戒区域监控.exe --selftest --deep --video D:\\素材\\test.mp4

    不给 ``--video`` 也能跑，但那样只验到「前向能跑通」，验不到闯入告警链路。
    """
    import app_paths

    lines: list[str] = []
    failures: list[str] = []

    def record(label: str, value: object) -> None:
        lines.append(f"{label:<22}: {value}")

    record("frozen", app_paths.IS_FROZEN)
    record("executable", sys.executable)
    record("APP_DIR", app_paths.APP_DIR)
    record("data_dir", app_paths.data_dir())

    for label, module_name in (
        ("numpy", "numpy"),
        ("cv2", "cv2"),
        ("torch", "torch"),
        ("PySide6.QtWidgets", "PySide6.QtWidgets"),
        ("ultralytics", "ultralytics"),
        ("supervision", "supervision"),
        ("trackers", "trackers"),
        ("scipy.interpolate", "scipy.interpolate"),
    ):
        try:
            __import__(module_name)
        except Exception as error:  # noqa: BLE001 - 自检要报告任何导入失败
            record(f"import {label}", f"FAIL {error!r}")
            failures.append(f"import {label}: {error!r}")
        else:
            record(f"import {label}", "ok")

    # openvino 只在 CPU 低功耗预设下需要，缺失不算失败。
    try:
        import openvino

        record("import openvino", getattr(openvino, "__version__", "ok"))
    except Exception as error:  # noqa: BLE001
        record("import openvino", f"absent ({error!r}) - 低功耗预设将不可用")

    # 下面这些步骤会顺着 main_window 把整条依赖链拖起来，也正是打包版最容易崩的地方
    # （被排掉的模块、找不到的资源都在这里现形）。**每一步都必须挡住异常**：
    # 让它逃出去的话，报告一个字都写不出来，构建脚本只剩一个空的 selftest.log 和
    # 一个 PyInstaller 模态对话框 —— 恰恰在最需要诊断信息的时候什么都没有。
    # 真实教训：semantic/train.py 顶层的 matplotlib 曾在 `import main_window`
    # 这一步炸掉，弹窗还把 -Wait 的构建脚本堵死了。
    def step(label: str, action: Callable[[], object]) -> object | None:
        try:
            return action()
        except Exception as error:  # noqa: BLE001 - 自检要报告任何失败
            record(label, f"FAIL {error!r}")
            failures.append(f"{label}: {error!r}")
            logger.exception("自检步骤失败: %s", label)
            return None

    def _probe_main_window() -> object:
        import main_window

        record("CONFIG_PATH", main_window.CONFIG_PATH)
        record("EVENTS_DIR", main_window.EVENTS_DIR)
        record("PROFILES_DIR", main_window.PROFILES_DIR)
        return main_window

    def _probe_alarm_wav() -> object:
        from alarm_service import AlarmPlayer

        wav = AlarmPlayer().warning_audio_path
        record("alarm wav", f"{wav} | exists={wav.exists()}")
        if not wav.exists():
            failures.append(f"告警音频缺失: {wav}")
        return wav

    def _probe_model() -> object:
        model = app_paths.resource("models/yolo11n.pt", "yolo11n.pt")
        record("yolo11n.pt", f"{model} | exists={model.exists()}")
        if not model.exists():
            failures.append(f"检测模型缺失: {model}")
        return model

    def _probe_low_power_dir() -> object:
        from inference_profiles import LOW_POWER_MODEL_DIR

        record(
            "openvino int8 dir",
            f"{LOW_POWER_MODEL_DIR} | exists={LOW_POWER_MODEL_DIR.exists()}",
        )
        return LOW_POWER_MODEL_DIR

    def _probe_devices() -> object:
        from compute_devices import enumerate_inference_devices

        devices = enumerate_inference_devices()
        record("devices", devices)
        return devices

    def _probe_matplotlib_cache() -> object:
        # 绿色版不该往用户目录或 %TEMP% 里撒东西。这两个值一起报，是因为
        # PyInstaller 自带的 rthook 会抢着把 MPLCONFIGDIR 指向临时目录（详见
        # _redirect_matplotlib_cache 的说明），只看环境变量看不出最终落点。
        import matplotlib

        cache_dir = Path(matplotlib.get_cachedir())
        record("MPLBACKEND", os.environ.get("MPLBACKEND", "(未设置)"))
        record("mpl cachedir", cache_dir)
        if app_paths.IS_FROZEN and cache_dir != app_paths.data("matplotlib"):
            failures.append(f"matplotlib 缓存目录跑到了 {cache_dir}，应为程序数据目录")
        return cache_dir

    step("import main_window", _probe_main_window)
    step("版本比对", lambda: _probe_version(record, failures))
    step("发布资源", lambda: _probe_assets(record, failures))
    step("alarm wav", _probe_alarm_wav)
    model_path = step("yolo11n.pt", _probe_model)
    step("openvino int8 dir", _probe_low_power_dir)
    step("devices", _probe_devices)
    step("matplotlib cachedir", _probe_matplotlib_cache)

    if "--deep" in argv:
        video_path = _parse_video_argument(argv, record, failures)
        record("深度自检", "已开启" + ("" if video_path else "（无视频素材，仅验前向）"))
        step("告警音", lambda: _deep_probe_audio(record))

        def _probe_all_policies() -> None:
            """标准 与 CPU 低功耗 两条推理路径各跑一遍。

            整段都必须在 step() 里面 —— 连 resolve_inference_policy 的 import 也算。
            它一样会失败（被 excludes 误伤、OpenVINO 的 runtime DLL 没收进来都会），
            而自检的铁律是任何一步都不能把异常放出去：放出去就一行报告都写不出来，
            构建脚本只剩一个空的 selftest.log。
            """
            if model_path is None or not Path(str(model_path)).is_file():
                # 上面那步已经把「模型缺失」记成失败了。这里必须硬停：权重不在时
                # YOLO(path) 会让 ultralytics **联网去下载** yolo11n.pt —— 那样自检
                # 可能"通过"，但通过的是一个刚从网上抓来的权重，跟发布包里的无关，
                # 而且在断网的验收机上只会挂在超时里。
                failures.append("深度自检: 权重文件不在，跳过推理验证（不让 ultralytics 联网下载）")
                return

            from inference_profiles import resolve_inference_policy

            for slug, label, low_power in (
                ("standard", "标准推理", False),
                ("lowpower", "低功耗推理", True),
            ):
                resolution = resolve_inference_policy(str(model_path), "cpu", low_power)
                if resolution.policy is None:
                    # 低功耗预设不可用不算失败（说明书里写明缺模型则该模式不可用），
                    # 但要把原因原样打出来，别让它变成一行沉默的跳过。
                    record(f"{label} 跳过", resolution.unavailable_reason)
                    continue
                step(
                    label,
                    lambda slug=slug, label=label, policy=resolution.policy: (
                        _deep_probe_policy(slug, label, policy, video_path, record, failures)
                    ),
                )

        step("深度推理", _probe_all_policies)

    lines.append("")
    lines.append("RESULT: " + ("FAIL" if failures else "PASS"))
    lines.extend(f"  - {failure}" for failure in failures)
    report = "\n".join(lines)

    print(report)
    report_path = app_paths.data("logs", "selftest.log")
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report + "\n", encoding="utf-8")
    except OSError as error:
        logger.warning("无法写入自检报告 %s: %s", report_path, error)
    return 1 if failures else 0


def _log_startup_banner() -> None:
    """把「哪一版、跑在哪、数据写到哪」记进日志。

    现场支持的第一个问题永远是这两句：你装的是哪一版、配置在哪。以前日志里只有
    「Loading model: …」，版本号只存在于 exe 的属性对话框里，界面和日志都看不到。
    """
    import app_paths
    import app_version

    logger.info(
        "行人警戒区域监控 v%s 启动 | frozen=%s | exe=%s | 程序目录=%s | 数据目录=%s",
        app_version.VERSION,
        app_paths.IS_FROZEN,
        sys.executable,
        app_paths.APP_DIR,
        app_paths.data_dir(),
    )


def main() -> int:
    configure_logging()
    _install_excepthook()
    # 必须在任何可能 import matplotlib 的语句之前 —— 下面的 import main_window 会
    # 顺着 ultralytics 把它拉起来。
    _redirect_matplotlib_cache()

    if "--selftest" in sys.argv[1:]:
        return _run_selftest(sys.argv[1:])

    _log_startup_banner()

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "未安装 PySide6。请运行: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    try:
        from main_window import MainWindow
    except ModuleNotFoundError as error:
        print(
            f"缺少运行依赖 {error.name!r}。请运行: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    application = QApplication(sys.argv)
    application.setApplicationName("行人警戒区域监控")
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
