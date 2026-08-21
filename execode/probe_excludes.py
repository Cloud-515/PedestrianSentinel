"""探针：一次性查出 spec 的 excludes 里哪些其实是运行时真需要的。

背景：打包版启动即 `ModuleNotFoundError: No module named 'matplotlib'`。
一个个试错要重建一次（5 分钟）才知道下一个是谁，太慢。这里换个思路 ——
在**完整依赖都在**的 .venv-build 里跑真实的 import 链，用 meta_path 钩子记录
"谁被 import 了"但**不拦截**，于是一趟就能拿到全部名单。

    E:\\c#code\\PP-Human\\.venv-build\\Scripts\\python.exe execode\\probe_excludes.py

输出分两类：
    命中 = 这些必须从 excludes 里摘掉（或另想办法），否则打包版启动就崩
    未命中 = 排掉是安全的
"""

from __future__ import annotations

import ast
import importlib.abc
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPEC_PATH = Path(__file__).resolve().parent / "PedestrianZoneMonitor.spec"


def _load_excludes() -> list[str]:
    """从 spec 里直接读 excludes，而不是在这儿再抄一份。

    抄一份注定漂移：spec 摘掉 matplotlib 之后，本文件里那份旧清单会把它报成
    「命中，必须摘掉」—— 而它早已不在列表里，纯属误导。spec 不能 import
    （里头用了 PyInstaller 注入的 SPECPATH/Analysis），所以按语法树取那个字面量。
    """
    tree = ast.parse(SPEC_PATH.read_text(encoding="utf-8"), filename=str(SPEC_PATH))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "excludes" for target in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise SystemExit(f"在 {SPEC_PATH.name} 里找不到模块级的 excludes = [...] 赋值")


EXCLUDES = _load_excludes()

# name -> 第一次被 import 时的调用栈（只留项目/三方库那几帧，够定位就行）
hits: dict[str, str] = {}


def _matches(name: str) -> str | None:
    for excluded in EXCLUDES:
        if name == excluded or name.startswith(excluded + "."):
            return excluded
    return None


class _Recorder(importlib.abc.MetaPathFinder):
    """记录后放行 —— 不抛异常，这样一趟能收集到全部命中项。"""

    def find_module(self, fullname, path=None):  # 兼容老接口
        return None

    def find_spec(self, fullname, path=None, target=None):
        excluded = _matches(fullname)
        if excluded is not None and excluded not in hits:
            frames = [
                f"  {Path(frame.filename).name}:{frame.lineno} in {frame.name}"
                for frame in traceback.extract_stack()[:-1]
                if "importlib" not in frame.filename and "probe_excludes" not in frame.filename
            ]
            hits[excluded] = f"由 {fullname} 触发\n" + "\n".join(frames[-4:])
        return None  # 放行，交给后面的 finder


sys.meta_path.insert(0, _Recorder())


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"探针覆盖 {len(EXCLUDES)} 个 excludes 项\n")

    # 走的是打包版真实崩溃点的同一条链：main.py --selftest 会 import main_window。
    steps = [
        ("import main_window", lambda: __import__("main_window")),
        ("from ultralytics import YOLO", lambda: __import__("ultralytics").YOLO),
        ("import supervision", lambda: __import__("supervision")),
        ("import trackers", lambda: __import__("trackers")),
        ("import openvino", lambda: __import__("openvino")),
        ("import inference_profiles", lambda: __import__("inference_profiles")),
        ("import compute_devices", lambda: __import__("compute_devices")),
    ]
    failures = 0
    for label, action in steps:
        try:
            action()
        except Exception as error:  # noqa: BLE001 - 探针要报告任何失败
            print(f"[FAIL] {label}: {error!r}")
            failures += 1
        else:
            print(f"[ok]   {label}")

    print("\n" + "=" * 70)
    print(f"命中（必须保留，共 {len(hits)} 项）")
    print("=" * 70)
    for name in sorted(hits):
        print(f"\n!!! {name}\n{hits[name]}")

    safe = [name for name in EXCLUDES if name not in hits]
    print("\n" + "=" * 70)
    print(f"未命中（排掉是安全的，共 {len(safe)} 项）")
    print("=" * 70)
    print("\n".join(f"  {name}" for name in safe))
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
