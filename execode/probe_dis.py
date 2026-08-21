"""排查 PyInstaller 扫描字节码时 dis 抛 IndexError 的具体模块。

复现 PyInstaller/lib/modulegraph/util.py:iterate_instructions 的做法：
编译每个 .py，再递归 dis 所有嵌套 code object。

用法:
    python execode/probe_dis.py charset_normalizer requests
    python execode/probe_dis.py --all        # 扫整个 site-packages（慢）
"""
from __future__ import annotations

import dis
import sys
import sysconfig
import traceback
from pathlib import Path


def iterate_instructions(code_object):
    """与 PyInstaller 同款遍历：先本层，再递归进嵌套 code object。"""
    yield from (i for i in dis.get_instructions(code_object) if i.opname != "EXTENDED_ARG")
    for constant in code_object.co_consts:
        if isinstance(constant, type(code_object)):
            yield from iterate_instructions(constant)


def probe(path: Path) -> str | None:
    try:
        source = path.read_bytes()
    except OSError as error:
        return f"读文件失败: {error!r}"
    try:
        code = compile(source, str(path), "exec")
    except SyntaxError as error:
        return f"compile 失败: {error!r}"
    try:
        for _ in iterate_instructions(code):
            pass
    except Exception:
        return "dis 失败:\n" + traceback.format_exc()
    return None


def main(argv: list[str]) -> int:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    print(f"python      : {sys.version}")
    print(f"site-packages: {site_packages}")

    if "--all" in argv:
        targets = sorted(site_packages.rglob("*.py"))
    else:
        names = [a for a in argv if not a.startswith("-")] or ["charset_normalizer"]
        targets = []
        for name in names:
            package = site_packages / name
            if package.is_dir():
                targets.extend(sorted(package.rglob("*.py")))
            elif (site_packages / f"{name}.py").is_file():
                targets.append(site_packages / f"{name}.py")
            else:
                print(f"跳过（找不到）: {name}")

    print(f"待扫描      : {len(targets)} 个文件")
    failures = 0
    for target in targets:
        problem = probe(target)
        if problem is not None:
            failures += 1
            print(f"\n!!! {target.relative_to(site_packages)}\n{problem}")
    print(f"\n扫描完成: {len(targets)} 个文件, {failures} 个失败")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
