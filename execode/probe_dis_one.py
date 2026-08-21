"""判断某个解释器的 dis 是否带 bpo-45757 修复。

对给定文件做「编译 + 递归反汇编」，与 PyInstaller 的做法一致。
路径由命令行给出（用 3.10.0 已知会崩的那几个大模块）。

用法:
    <某个python.exe> execode/probe_dis_one.py <file.py> [<file.py> ...]
"""
from __future__ import annotations

import dis
import inspect
import sys
import traceback
from pathlib import Path


def iterate_instructions(code_object):
    yield from (i for i in dis.get_instructions(code_object) if i.opname != "EXTENDED_ARG")
    for constant in code_object.co_consts:
        if inspect.iscode(constant):
            yield from iterate_instructions(constant)


def main(paths: list[str]) -> int:
    print(f"python: {sys.version.split()[0]}")
    failures = 0
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            print(f"  跳过（不存在）: {path}")
            continue
        try:
            code = compile(path.read_bytes(), str(path), "exec")
        except SyntaxError as error:
            # 语法不兼容说明这个解释器版本读不了该文件，与 dis 无关，单独区分。
            print(f"  SYNTAX  {path.name}: {error}")
            failures += 1
            continue
        try:
            count = sum(1 for _ in iterate_instructions(code))
        except Exception:
            print(f"  FAIL    {path.name}")
            print("    " + traceback.format_exc().strip().splitlines()[-1])
            failures += 1
        else:
            print(f"  ok      {path.name} ({count} 条指令)")
    print("RESULT: " + ("FAIL" if failures else "PASS"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
