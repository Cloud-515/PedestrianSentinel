"""验证 bpo-45757 的一行修复能否让 3.10.0 正确反汇编那几个大模块。

假设：3.10.0 的编译器会产出「EXTENDED_ARG 紧跟无参数操作码（如 NOP）」的序列，
而 dis._unpack_opargs 在 else 分支里没有把 extended_arg 归零，残留值就串到了
下一条指令上，把 LOAD_CONST 的下标撑爆 —— 于是 _get_const_info 抛 IndexError。
上游 3.10.1 的修复正是在 else 分支加一句 extended_arg = 0。

用法:
    E:\\python\\python.exe execode/probe_dis_patch.py <file.py> [...]
"""
from __future__ import annotations

import dis
import inspect
import sys
import traceback
from pathlib import Path


def patched_unpack_opargs(code):
    """dis._unpack_opargs 的 bpo-45757 修复版（3.10.1+ 的行为）。"""
    extended_arg = 0
    for i in range(0, len(code), 2):
        op = code[i]
        if op >= dis.HAVE_ARGUMENT:
            arg = code[i + 1] | extended_arg
            extended_arg = (arg << 8) if op == dis.EXTENDED_ARG else 0
        else:
            arg = None
            extended_arg = 0  # <-- 上游补的就是这一行
        yield (i, op, arg)


def iterate_instructions(code_object):
    yield from (i for i in dis.get_instructions(code_object) if i.opname != "EXTENDED_ARG")
    for constant in code_object.co_consts:
        if inspect.iscode(constant):
            yield from iterate_instructions(constant)


def scan(paths: list[Path]) -> tuple[int, int]:
    ok = bad = 0
    for path in paths:
        try:
            code = compile(path.read_bytes(), str(path), "exec")
            count = sum(1 for _ in iterate_instructions(code))
        except Exception:
            bad += 1
            print(f"    FAIL {path.name}: {traceback.format_exc().strip().splitlines()[-1]}")
        else:
            ok += 1
            print(f"    ok   {path.name} ({count} 条指令)")
    return ok, bad


def count_extended_arg_before_noarg(paths: list[Path]) -> int:
    """直接数一数「EXTENDED_ARG 后面跟无参数操作码」出现了多少次，验证假设本身。"""
    total = 0

    def walk(code_object) -> None:
        nonlocal total
        raw = code_object.co_code
        for i in range(0, len(raw) - 2, 2):
            if raw[i] == dis.EXTENDED_ARG and raw[i + 2] < dis.HAVE_ARGUMENT:
                total += 1
        for constant in code_object.co_consts:
            if inspect.iscode(constant):
                walk(constant)

    for path in paths:
        try:
            walk(compile(path.read_bytes(), str(path), "exec"))
        except Exception:
            pass
    return total


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv if Path(a).is_file()]
    print(f"python: {sys.version.split()[0]}  待检 {len(paths)} 个文件")

    print("\n[1] 原版 dis._unpack_opargs")
    _, bad_before = scan(paths)

    hits = count_extended_arg_before_noarg(paths)
    print(f"\n[2] 「EXTENDED_ARG + 无参操作码」出现次数: {hits}")

    print("\n[3] 打上 bpo-45757 一行修复后")
    dis._unpack_opargs = patched_unpack_opargs
    _, bad_after = scan(paths)

    print(f"\n修复前失败 {bad_before} 个 / 修复后失败 {bad_after} 个")
    verdict = bad_before > 0 and bad_after == 0
    print("RESULT: " + ("假设成立，一行修复有效" if verdict else "假设不成立，需另找原因"))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
