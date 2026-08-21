# -*- mode: python ; coding: utf-8 -*-
"""行人警戒区域监控 — PyInstaller 单目录（onedir）打包配置。

产物目录结构（构建脚本会把 exe 与外层目录改回中文名）::

    行人警戒区域监控/
    ├── 行人警戒区域监控.exe
    ├── _internal/                          ← Python 解释器 + 全部依赖，勿动
    ├── models/
    │   ├── yolo11n.pt                      ← 可替换
    │   └── yolo11n_int8_openvino_model/    ← CPU 低功耗预设用
    ├── assets/
    │   └── warming_converted.wav           ← 可替换
    └── (config.json / profiles / events / logs 首次运行时生成)

这里刻意用纯 ASCII 的名字构建，避免 Windows 控制台码页把中文名弄成乱码；
改名在 build.ps1 的最后一步做 —— onedir 的引导程序按 exe 所在目录去找
`contents_directory`（默认 "_internal"），与 exe 文件名无关，所以改名是安全的。

模型与音频**不进** datas —— 它们要留在 `_internal/` 外面才能被用户替换，
由 build.ps1 在 PyInstaller 跑完之后拷贝到位。

不要直接调用本文件，走构建脚本::

    powershell -ExecutionPolicy Bypass -File .\execode\build.ps1
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

PROJECT_ROOT = Path(SPECPATH).resolve().parent  # execode 的父目录，即项目根目录
BUILD_NAME = "PedestrianZoneMonitor"            # 纯 ASCII；build.ps1 事后改中文名

_ICON_PATH = Path(SPECPATH) / "app.ico"
_VERSION_PATH = Path(SPECPATH) / "version_info.txt"


def _optional_resource(path: Path, what: str, consequence: str) -> str | None:
    """缺失时**出声**再降级。

    这两项都不该让构建失败（图标和版本信息不影响能不能跑），但也绝不能静默 ——
    悄悄发一个带 PyInstaller 默认图标、没有版本号的 exe，等发现时东西已经在别人
    手里了。之前就是 `... if path.is_file() else None` 一句话吞掉的。
    """
    if path.is_file():
        return str(path)
    print(f"spec: 警告 —— 找不到{what} {path.name}，{consequence}")
    return None


# --- 绕过 CPython 3.10.0 的 bpo-45757 -------------------------------------------
# 3.10.0 的编译器会产出「EXTENDED_ARG 紧跟无参数操作码」的字节码序列，而该版本的
# dis._unpack_opargs 在 else 分支里没把 extended_arg 归零，残留值就串到下一条指令
# 上，把 LOAD_CONST 的常量下标撑爆。PyInstaller 扫描字节码时因此直接崩掉：
#     File ".../PyInstaller/lib/modulegraph/util.py", line 13, in iterate_instructions
#     IndexError: tuple index out of range
# 中招的是所有「够大」的模块 —— torch/nn/functional.py、torch/overrides.py、
# PIL/Image.py、scipy/stats/_stats_py.py、openvino/runtime/opset1/ops.py、
# rich/console.py、sympy/matrices/matrixbase.py …… 全是硬依赖，一个都排除不掉。
# 下面是上游 3.10.1 那个修复的原样搬运（实质就是多一句 extended_arg = 0），因此
# 得到的是**正确**的反汇编结果，而不是把异常吞掉。仅在恰好 3.10.0 上生效；
# 换成 >= 3.10.1 的解释器后此段自动失效，可整段删除。
if sys.version_info[:3] == (3, 10, 0):
    import dis

    def _unpack_opargs_bpo45757(code):
        extended_arg = 0
        for i in range(0, len(code), 2):
            op = code[i]
            if op >= dis.HAVE_ARGUMENT:
                arg = code[i + 1] | extended_arg
                extended_arg = (arg << 8) if op == dis.EXTENDED_ARG else 0
            else:
                arg = None
                extended_arg = 0  # 上游补的就是这一行
            yield (i, op, arg)

    dis._unpack_opargs = _unpack_opargs_bpo45757
    print("spec: 解释器为 3.10.0，已对 dis._unpack_opargs 应用 bpo-45757 修复")



# --- 收集三方依赖 ---------------------------------------------------------------
# 这些包靠动态 import / 数据文件工作，PyInstaller 静态分析抓不全：
#   ultralytics  需要 cfg/*.yaml 等数据文件
#   openvino     需要 runtime DLL 与 plugins.xml
# torch / torchvision 交给 PyInstaller 自带的 hook 处理 —— 千万别再对它们做
# collect_submodules()，那会把整棵开发树拖进来，是上一版 _internal 膨胀到
# 数 GB 的直接原因。
datas: list = []
binaries: list = []
hiddenimports: list = []

for package in ("ultralytics", "supervision", "trackers", "openvino"):
    try:
        package_datas, package_binaries, package_hidden = collect_all(package)
    except Exception:
        continue
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden


# --- 明确排除 -------------------------------------------------------------------
# 这份清单别靠猜，也别靠"改一项、重建五分钟、看崩不崩"。execode/probe_excludes.py
# 在依赖装齐的 .venv-build 里跑真实 import 链（main_window / YOLO / supervision /
# trackers / openvino），用 meta_path 钩子记录每个被真正 import 到的排除项，一趟
# 就能拿到完整名单。**动这个列表之前先跑那个探针。**
# 最近一次实测（ultralytics 8.4.123）：35 项里只有 matplotlib 命中，其余 34 项安全。
#
# 三个已核实的边界：
#   * scipy 绝对不能排 —— supervision/annotators/core.py 顶层就
#     `from scipy.interpolate import splev, splprep`，而 detection_engine.py
#     直接 `import supervision as sv`，排掉即运行时崩。
#   * matplotlib 同样不能排，尽管本程序一张图都不画。`from ultralytics import YOLO`
#     会触发 ultralytics/__init__.py 的 __getattr__ → import ultralytics.models
#     → .fastsam.predict → models/yolo/__init__.py，而后者一次性 import 了
#     classify/depth/detect/obb/pose/segment/semantic/world/yoloe 全部任务包，
#     其中 semantic/train.py:8 顶层写着 `import matplotlib.pyplot as plt`。
#     曾按"只有训练路径才懒加载"把它排掉，结果打包版启动即 ModuleNotFoundError。
#     实测代价只有 +6 MB（_internal/matplotlib 11.6 MB，其中 mpl-data 字体 8.6 MB；
#     纯 py 源码进压缩 PYZ 后 exe 涨 1.6 MB；kiwisolver + contourpy 0.6 MB）。
#     别拿 site-packages 里的 36 MB 估它 —— 那里面大头是 .py 源码和测试，不会原样搬。
#   * polars / pandas 可以排 —— ultralytics 只在 benchmark 与数据集统计里函数内部
#     懒加载它们（源码注释写着 "scope for faster 'import ultralytics'"）。
excludes = [
    # 重型但本程序用不到的三方库
    "polars",
    "pandas",
    "onnx",
    "onnxruntime",
    "nncf",
    # 其它 GUI 工具包
    "tkinter",
    "PyQt5",
    "PyQt6",
    "PySide2",
    # 开发期工具（unittest 不排：torch/numpy 的若干模块会在运行时 import 它）
    "IPython",
    "pytest",
    # 本程序只用 QtCore/QtGui/QtWidgets，Qml/Quick 及各类 addon 一律不要
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuickWidgets",
    "PySide6.QtQuickControls2",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtGraphs",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtDesigner",
    "PySide6.QtUiTools",
    "PySide6.QtHelp",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtLocation",
    "PySide6.QtPositioning",
    # 项目内的遗留独立脚本：自带 main/cv2.imshow，GUI 从不引用它
    "security_monitor",
]


# --- 分析入口 -------------------------------------------------------------------
a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(Path(SPECPATH) / "runtime_hook.py")],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,       # 单目录模式关键：二进制留给 COLLECT，不塞进 exe
    name=BUILD_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                   # 不压缩：避免杀软误报，也避免启动变慢
    console=False,               # PySide6 GUI 程序，不弹黑窗
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory="_internal",
    icon=_optional_resource(
        _ICON_PATH, "程序图标", "exe 将使用 PyInstaller 默认图标。跑 execode/make_icon.py 可生成。"
    ),
    version=_optional_resource(
        _VERSION_PATH, "版本信息", "exe 属性面板里将没有版本号与程序名。"
    ),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=BUILD_NAME,             # dist/<BUILD_NAME>/ 即最终产物（待改名）
)
