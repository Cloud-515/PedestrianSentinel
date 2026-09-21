"""PyInstaller 运行时钩子：在任何三方库加载前设好几个环境变量。

路径解析已经由 app_paths.py 统一负责（源码模式与冻结模式同一套逻辑），
所以这里**不再**做 os.chdir 或猴子补丁 —— 那些是上一版遗留，已删除。

钩子在 bootloader 解包完成之后、用户脚本之前执行，正好是设 env 的时机：
这几个变量都必须在对应的 DLL / 模块首次加载前生效，写在 main.py 里就晚了。
"""
from __future__ import annotations

import os
import sys


def _apply_frozen_environment() -> None:
    if not getattr(sys, "frozen", False):
        return

    # torch 和 openvino 各自带了一份 libiomp5md.dll，同目录共存会触发
    # "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
    # already initialized" 并直接 abort。允许重复加载是官方给的规避方式。
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # ultralytics 首次 import 会写一份 settings.json，默认落在
    # %APPDATA%\Ultralytics。绿色版不该往用户目录里撒文件，改指到自己的数据目录。
    if "YOLO_CONFIG_DIR" not in os.environ:
        try:
            import app_paths

            config_dir = app_paths.data("ultralytics")
            config_dir.mkdir(parents=True, exist_ok=True)
            os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
        except Exception:  # noqa: BLE001 - 拿不到可写目录就退回 ultralytics 的默认位置
            # 钩子在解释器启动最早期执行，这里绝不能抛：抛出去程序连界面都起不来。
            pass

    # matplotlib 是被 ultralytics 硬拖进来的（models/yolo/semantic/train.py 顶层
    # import pyplot），本程序一张图都不画。别让它在我们自己的 PySide6 进程里去挑
    # 交互式后端 —— 默认的后端自动探测会尝试加载 Qt binding，和已就位的 PySide6
    # 抢 QApplication 属于纯粹的风险，没有任何收益。
    os.environ.setdefault("MPLBACKEND", "Agg")

    # MPLCONFIGDIR **不能**在这里设：PyInstaller 自带的 pyi_rth_mplconfig 在本钩子
    # 之后执行，且是无条件赋值（os.environ['MPLCONFIGDIR'] = secure_mkdtemp()），
    # 写在这里必被覆盖。自定义钩子的顺序改不了，所以那件事挪到了 main.py 的
    # _redirect_matplotlib_cache()，在 matplotlib 首次 import 之前生效。


_apply_frozen_environment()
