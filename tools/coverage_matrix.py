#!/usr/bin/env python3
"""Generate or verify the operator × backend coverage matrix.

覆盖矩阵是 FR-1 的守门人。事实源有两个，都不允许猜：

  * `include/tensorvia/ops.hpp` —— 公开算子名（重载按名字收敛）；
  * 四份后端 ops 头里的 `template <> struct XxxImpl<Device::X>` 全特化 ——
    原生实现（native）。派发点 `execute_on`（src/core/ops.cpp:37）只认全特化，
    偏特化不满足 `has_*_impl` 概念，因此偏特化一律报错而不是算作覆盖。

推不出来的一律进 `tools/coverage_overrides.toml` 人工标注：

  * `[host_side]`   宿主侧 API（不经过后端派发，不进矩阵）；
  * `[core_impl]`   实现在 src/core/ops.cpp、只服务 CPU 的直实现（附 file:line 证据）；
  * `[emulated]`    组合实现：无自己的内核，由其他算子拼成（`parts` 会被校验：
                    所声称可用的每个后端上，部件本身必须受支持）；
  * `[unsupported]` 显式拒绝：设备上调用会抛错（附 file:line 证据）。

一个格子既没有 Impl 特化、又没有标注，`--check` 直接失败并指名算子与后端——
"文档没写、设备上悄悄拒绝"的组合就是这样暴露的（FR-2）。

Usage:
    python3 tools/coverage_matrix.py --write   # regenerate docs/coverage.md
    python3 tools/coverage_matrix.py --check   # fail on drift
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OPS_HPP_REL = "include/tensorvia/ops.hpp"
OVERRIDES_REL = "tools/coverage_overrides.toml"
DOC_REL = "docs/coverage.md"
IMPL_HEADER_REL = {
    "CPU": "include/tensorvia/cpu_ops.hpp",
    "CUDA": "include/tensorvia/backend/cuda/cuda_ops.hpp",
    "SYCL": "include/tensorvia/backend/sycl/sycl_ops.hpp",
    "VULKAN": "include/tensorvia/backend/vulkan/vulkan_ops.hpp",
}
BACKENDS = ("CPU", "CUDA", "SYCL", "VULKAN")
DEVICE_BACKENDS = ("CUDA", "SYCL", "VULKAN")
# overrides 里的伪后端键：`Op.devices` = 三个设备后端。
PSEUDO_BACKENDS = {"devices": DEVICE_BACKENDS}

NATIVE = "native"
EMULATED = "emulated"
UNSUPPORTED = "unsupported"


class CoverageError(Exception):
    """矩阵与源码/标注不一致。消息里必须能指名算子与后端。"""


# ---------------------------------------------------------------- 源码解析

# 语句前缀：这些开头的行不是声明（与 tools/gen_facade.py 同一套思路）。
STMT_PREFIXES = ("return", "if", "for", "while", "switch", "else", "using",
                 "static_assert", "friend", "typename", "template", "case",
                 "default", "do", "catch", "break", "continue", "goto",
                 "public:", "private:", "protected:", "{", "}", "#")
# 函数声明/定义：返回类型 + 名字 + 左圆括号。名字允许落在多行声明的首行。
FUNC_DECL = re.compile(r"(?:template\s*<[^>]*>\s*)?[A-Za-z_][\w:<>,\*&\s]*?\s(\w+)\s*\(")
TYPE_DECL = re.compile(r"(?:class|struct)\s+(\w+)")
NS_OPEN = re.compile(r"namespace\s+ops\s*\{")
# 全特化：`template <> struct XxxImpl<Device::CPU>`（template 与 struct 可跨行）。
FULL_SPEC = re.compile(r"template\s*<>\s*struct\s+(\w+)Impl\s*<\s*Device::(\w+)\s*>")
# 偏特化：template 参数表非空。派发概念不认它，出现即报错。
PARTIAL_SPEC = re.compile(
    r"template\s*<[^<>]*[^\s<>][^<>]*>\s*struct\s+(\w+)Impl\s*<\s*Device::(\w+)\s*>")


def strip_comments(text: str) -> str:
    """去掉 C 注释，保留换行——行号与证据标注对齐。

    假设这些头文件里没有含 `//` 或 `/*` 的字符串字面量（错误消息均为纯文本）。
    块注释内容替换为空格，行注释只截掉行尾，行数因此不变。
    """
    out: list[str] = []
    i, n, in_block = 0, len(text), False
    while i < n:
        if in_block:
            end = text.find("*/", i)
            if end == -1:
                out.append(" " * (n - i))
                break
            chunk = text[i:end]
            out.append("".join("\n" if c == "\n" else " " for c in chunk))
            i = end + 2
            in_block = False
            continue
        line = text.find("//", i)
        block = text.find("/*", i)
        if line == -1 and block == -1:
            out.append(text[i:])
            break
        if block == -1 or (line != -1 and line < block):
            out.append(text[i:line])
            end = text.find("\n", line)
            if end == -1:
                break
            out.append("\n")
            i = end + 1
        else:
            out.append(text[i:block])
            i = block + 2
            in_block = True
    return "".join(out)


def parse_public_ops(path: pathlib.Path) -> list[str]:
    """返回 `namespace ops` 里的顶层公开名字（函数与类型），按名字去重排序。

    类/函数体整体跳过（成员随类发布，不是面）；模板函数（println）放行——
    "template" 虽在 STMT_PREFIXES 里，但模板函数声明必须认。
    """
    names: set[str] = set()
    in_ops = False
    depth = 0  # >0：正在类体或函数体里
    for raw in strip_comments(path.read_text(encoding="utf-8")).splitlines():
        s = raw.strip()
        if not in_ops:
            if NS_OPEN.match(s):
                in_ops = True
            continue
        if depth > 0:
            depth += s.count("{") - s.count("}")
            depth = max(depth, 0)
            continue
        if s.startswith("}"):
            in_ops = False  # namespace ops 收尾（本文件只有一个顶层命名空间）
            continue
        m = TYPE_DECL.match(s)
        if m and (not s.endswith(";") or "{" in s):
            names.add(m.group(1))
            depth = s.count("{") - s.count("}")
            if depth <= 0:
                depth = 1  # 定义体的 { 在后续行
            continue
        m = FUNC_DECL.match(s)
        if m and (s.startswith("template") or not s.startswith(STMT_PREFIXES)):
            names.add(m.group(1))
            depth = max(s.count("{") - s.count("}"), 0)
            continue
        if s.startswith(STMT_PREFIXES):
            continue
    return sorted(names)


def scan_impl_header(path: pathlib.Path) -> tuple[dict[str, set[str]], list[tuple[str, str, int]]]:
    """扫一份后端 ops 头，返回 (全特化: Impl 名 -> 后端集合, 偏特化列表)。"""
    text = strip_comments(path.read_text(encoding="utf-8"))
    full: dict[str, set[str]] = {}
    for m in FULL_SPEC.finditer(text):
        full.setdefault(m.group(1), set()).add(m.group(2))
    partial = [(m.group(1), m.group(2), text.count("\n", 0, m.start()) + 1)
               for m in PARTIAL_SPEC.finditer(text)]
    return full, partial


# ---------------------------------------------------------------- 标注文件

class Entry:
    """overrides 里的一条标注（已展开后端）。"""

    __slots__ = ("op", "backends", "note", "parts", "evidence", "section")

    def __init__(self, op: str, backends: tuple[str, ...], note: str,
                 parts: tuple[str, ...] = (), evidence: str | None = None,
                 section: str = ""):
        self.op = op
        self.backends = backends
        self.note = note
        self.parts = parts
        self.evidence = evidence
        self.section = section


class Overrides:
    def __init__(self) -> None:
        self.host_side: list[str] = []
        self.core_impl: dict[str, Entry] = {}
        self.emulated: list[Entry] = []
        self.unsupported: list[Entry] = []


def _leaf_entries(section: dict, section_name: str) -> list[tuple[str, dict]]:
    """把 `Op = {...}` 与 `Op.BACKEND = {...}` 两种键统一成 (Op.BACKEND, 表)。"""
    leaves: list[tuple[str, dict]] = []
    for key, value in section.items():
        if isinstance(value, dict) and "note" in value:
            leaves.append((key, value))
        elif isinstance(value, dict):
            for backend, sub in value.items():
                if backend not in BACKENDS and backend not in PSEUDO_BACKENDS:
                    raise CoverageError(
                        f"{section_name}: key {key}.{backend}: unknown backend "
                        f"(expected one of {', '.join(BACKENDS + tuple(PSEUDO_BACKENDS))})")
                if not (isinstance(sub, dict) and "note" in sub):
                    raise CoverageError(
                        f"{section_name}: key {key}.{backend}: entry must be a table with a note")
                leaves.append((f"{key}.{backend}", sub))
        else:
            raise CoverageError(
                f"{section_name}: key {key}: entry must be a table with a note")
    return leaves


def _expand_backends(key: str, section_name: str) -> tuple[str, tuple[str, ...]]:
    if "." not in key:
        return key, BACKENDS
    op, _, backend = key.rpartition(".")
    if backend in PSEUDO_BACKENDS:
        return op, PSEUDO_BACKENDS[backend]
    if backend not in BACKENDS:
        raise CoverageError(
            f"{section_name}: key {key}: unknown backend (expected one of "
            f"{', '.join(BACKENDS + tuple(PSEUDO_BACKENDS))})")
    return op, (backend,)


def _entry_from(key: str, table: dict, section: str) -> Entry:
    op, backends = _expand_backends(key, section)
    note = table.get("note")
    if not isinstance(note, str) or not note.strip():
        raise CoverageError(f"{section}: key {key}: note must be a non-empty string")
    parts = tuple(table.get("parts", ()))
    if parts and section != "emulated":
        raise CoverageError(f"{section}: key {key}: parts is only valid for [emulated]")
    for part in parts:
        if not isinstance(part, str) or not part:
            raise CoverageError(f"{section}: key {key}: parts must be op names")
    evidence = table.get("evidence")
    if evidence is not None and not isinstance(evidence, str):
        raise CoverageError(f"{section}: key {key}: evidence must be 'path:line'")
    return Entry(op, backends, note.strip(), parts, evidence, section)


def load_overrides(path: pathlib.Path) -> Overrides:
    if not path.is_file():
        raise CoverageError(f"missing overrides file: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    ov = Overrides()

    host = data.get("host_side", {})
    names = host.get("names", [])
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise CoverageError("[host_side].names must be a list of strings")
    ov.host_side = sorted(names)

    core = data.get("core_impl", {})
    if not isinstance(core, dict):
        raise CoverageError("[core_impl] must be a table")
    for key, table in core.items():
        if "." in key:
            raise CoverageError(
                f"[core_impl]: key {key}: core implementations serve the CPU column only; "
                f"drop the backend suffix")
        if not (isinstance(table, dict) and "note" in table):
            raise CoverageError(f"[core_impl]: key {key}: entry must be a table with a note")
        ov.core_impl[key] = _entry_from(key, table, "core_impl")

    for section, target in (("emulated", ov.emulated), ("unsupported", ov.unsupported)):
        table = data.get(section, {})
        if not isinstance(table, dict):
            raise CoverageError(f"[{section}] must be a table")
        for key, leaf in _leaf_entries(table, section):
            target.append(_entry_from(key, leaf, section))
    return ov


# ---------------------------------------------------------------- 矩阵

class Cell:
    __slots__ = ("value", "note")

    def __init__(self, value: str, note: str = ""):
        self.value = value
        self.note = note


class Matrix:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Cell]] = {}
        self.host_side: list[str] = []

    def stats(self) -> dict[str, dict[str, int]]:
        out = {}
        for backend in BACKENDS:
            counts = {NATIVE: 0, EMULATED: 0, UNSUPPORTED: 0}
            for cells in self.rows.values():
                counts[cells[backend].value] += 1
            counts["supported"] = counts[NATIVE] + counts[EMULATED]
            out[backend] = counts
        return out


def _verify_evidence(root: pathlib.Path, entry: Entry) -> None:
    if not entry.evidence:
        return
    m = re.fullmatch(r"(.+):(\d+)", entry.evidence.strip())
    if not m:
        raise CoverageError(
            f"[{entry.section}]: {entry.op}: evidence must be 'path:line', got {entry.evidence!r}")
    rel, lineno = m.group(1), int(m.group(2))
    path = root / rel
    if not path.is_file():
        raise CoverageError(f"[{entry.section}]: {entry.op}: evidence file not found: {rel}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not 1 <= lineno <= len(lines):
        raise CoverageError(
            f"[{entry.section}]: {entry.op}: evidence line {lineno} out of range in {rel}")
    line = lines[lineno - 1].lower()
    # 接受算子名本身、Impl 名、以及 snake_case 形式（LogSoftmax → log_softmax_composed）。
    tokens = {entry.op.lower(), f"{entry.op.lower()}impl",
              re.sub(r"(?<!^)(?=[A-Z])", "_", entry.op).lower()}
    if not any(t in line for t in tokens):
        raise CoverageError(
            f"[{entry.section}]: {entry.op}: evidence {rel}:{lineno} does not mention "
            f"{entry.op} — stale line number?")


def _match_public(impl_base: str, public: set[str], where: str) -> str:
    """Impl 名 → 公开算子名：忽略大小写与下划线的唯一匹配。

    ArgMaxImpl → Argmax、ExportCsvImpl → export_csv、NonZeroImpl → Nonzero。
    """
    def norm(name: str) -> str:
        return name.lower().replace("_", "")

    hits = [name for name in public if norm(name) == norm(impl_base)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise CoverageError(
            f"{where}: {impl_base}Impl matches several public ops "
            f"({', '.join(sorted(hits))}) — rename one of them")
    raise CoverageError(
        f"{where}: {impl_base}Impl has no matching public op in {OPS_HPP_REL}")


def build_matrix(root: pathlib.Path) -> Matrix:
    root = pathlib.Path(root)
    errors: list[str] = []

    public = parse_public_ops(root / OPS_HPP_REL)
    public_set = set(public)
    ov = load_overrides(root / OVERRIDES_REL)

    # ---- host_side：必须是真实公开名，否则清单过期 ----
    for name in ov.host_side:
        if name not in public_set:
            errors.append(f"[host_side]: {name} is not a public op in {OPS_HPP_REL}")
    host_side = set(ov.host_side)

    # ---- 原生实现：四份后端头的全特化 ----
    native: dict[str, set[str]] = {}   # 公开算子名 -> 后端集合
    core_note: dict[str, str] = {}     # CPU 直实现的备注
    for backend, rel in IMPL_HEADER_REL.items():
        path = root / rel
        if not path.is_file():
            raise CoverageError(f"missing backend header: {rel}")
        full, partial = scan_impl_header(path)
        for impl_base, spec_backend, lineno in partial:
            errors.append(
                f"{rel}:{lineno}: partial specialization {impl_base}Impl<Device::{spec_backend}> "
                f"does not satisfy the execute_on dispatch concept — use a full specialization")
        for impl_base, backends in sorted(full.items()):
            if backends != {backend}:
                errors.append(
                    f"{rel}: {impl_base}Impl specializes {', '.join(sorted(backends))} "
                    f"inside the {backend} header")
                continue
            try:
                op = _match_public(impl_base, public_set, rel)
            except CoverageError as e:
                errors.append(str(e))
                continue
            native.setdefault(op, set()).add(backend)

    # ---- core_impl：src/core/ops.cpp 里的 CPU 直实现 ----
    for op, entry in sorted(ov.core_impl.items()):
        if op not in public_set:
            errors.append(f"[core_impl]: {op} is not a public op in {OPS_HPP_REL}")
            continue
        if "CPU" in native.get(op, set()):
            errors.append(
                f"[core_impl]: {op}: a CPU Impl specialization exists — drop the entry")
            continue
        try:
            _verify_evidence(root, entry)
        except CoverageError as e:
            errors.append(str(e))
            continue
        native.setdefault(op, set()).add("CPU")
        core_note[op] = entry.note

    # ---- emulated / unsupported 标注 ----
    annotated: dict[tuple[str, str], Entry] = {}
    for entry in ov.emulated + ov.unsupported:
        if entry.op not in public_set:
            errors.append(f"[{entry.section}]: {entry.op} is not a public op in {OPS_HPP_REL}")
            continue
        try:
            _verify_evidence(root, entry)
        except CoverageError as e:
            errors.append(str(e))
        for backend in entry.backends:
            if backend in native.get(entry.op, set()):
                errors.append(
                    f"[{entry.section}]: {entry.op}.{backend}: a native implementation "
                    f"exists — remove the annotation")
                continue
            key = (entry.op, backend)
            if key in annotated:
                errors.append(
                    f"[{entry.section}]: {entry.op}.{backend}: overlapping annotations "
                    f"(bare key and per-backend key both cover it)")
                continue
            annotated[key] = entry

    def status(op: str, backend: str) -> str:
        if backend in native.get(op, set()):
            return NATIVE
        cell = annotated.get((op, backend))
        return cell.section if cell is not None else UNSUPPORTED

    # ---- 组合实现的部件校验：声称可用的后端上，部件必须受支持 ----
    for entry in ov.emulated:
        if entry.op not in public_set:
            continue
        for part in entry.parts:
            if part not in public_set:
                errors.append(
                    f"[emulated]: {entry.op}: part {part} is not a public op in {OPS_HPP_REL}")
                continue
            for backend in entry.backends:
                if status(part, backend) == UNSUPPORTED:
                    errors.append(
                        f"[emulated]: {entry.op} is composed of {part}, but {part} is "
                        f"unsupported on {backend} — the composite claim is false")

    if errors:
        raise CoverageError("coverage matrix:\n  " + "\n  ".join(errors))

    # ---- 汇总：每个非宿主侧算子 × 四个后端都必须有取值 ----
    matrix = Matrix()
    matrix.host_side = ov.host_side
    for op in public:
        if op in host_side:
            continue
        cells: dict[str, Cell] = {}
        for backend in BACKENDS:
            if backend in native.get(op, set()):
                cells[backend] = Cell(NATIVE, core_note.get(op, ""))
            else:
                entry = annotated.get((op, backend))
                if entry is None:
                    errors.append(
                        f"{op}.{backend}: neither an Impl specialization nor an override "
                        f"entry — annotate it in {OVERRIDES_REL}")
                    continue
                cells[backend] = Cell(entry.section, entry.note)
        if errors:
            continue
        matrix.rows[op] = cells
    if errors:
        raise CoverageError("coverage matrix:\n  " + "\n  ".join(errors))
    return matrix


# ---------------------------------------------------------------- 渲染

LEGEND = (
    "**native** = 后端自己的实现（`XxxImpl<Device::X>` 全特化，或 `src/core/ops.cpp` "
    "中标注的 CPU 直实现）；**emulated** = 非原生但可用（由其他算子拼成，或在宿主生成后"
    "一次性搬运到设备）；**unsupported** = 显式拒绝（设备上调用抛错）。"
)


def _row_note(cells: dict[str, Cell]) -> str:
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for backend in BACKENDS:
        cell = cells[backend]
        note = cell.note
        if not note:
            continue
        if note not in grouped:
            grouped[note] = []
            order.append(note)
        grouped[note].append(backend)
    if not order:
        return ""
    if len(order) == 1:
        return order[0]
    return "；".join(f"{','.join(grouped[n])}: {n}" for n in order)


def render(matrix: Matrix) -> str:
    lines: list[str] = [
        "# 算子 × 后端覆盖矩阵",
        "",
        f"> 由 `tools/coverage_matrix.py` 生成，请勿手改。漂移守门："
        f"`python3 -I tools/coverage_matrix.py --check`（CI 同款）。",
        f"> 人工标注唯一来源：`tools/coverage_overrides.toml`"
        f"（组合实现与显式拒绝不从代码推断）。",
        "",
        LEGEND,
        "",
        f"矩阵共 {len(matrix.rows)} 个算子。",
        "",
        "| 算子 | CPU | CUDA | SYCL | VULKAN | 备注 |",
        "|------|-----|------|------|--------|------|",
    ]
    for op in sorted(matrix.rows):
        cells = matrix.rows[op]
        values = " | ".join(cells[b].value for b in BACKENDS)
        note = _row_note(cells)
        lines.append(f"| `{op}` | {values} | {note} |")
    lines += [
        "",
        "## 统计",
        "",
        "| 后端 | native | emulated | unsupported | 支持率 |",
        "|------|--------|----------|-------------|--------|",
    ]
    stats = matrix.stats()
    total = len(matrix.rows)
    for backend in BACKENDS:
        c = stats[backend]
        rate = f"{100.0 * c['supported'] / total:.1f}%" if total else "—"
        lines.append(
            f"| {backend} | {c[NATIVE]} | {c[EMULATED]} | {c[UNSUPPORTED]} | {rate} |")
    lines += [
        "",
        f"支持率 = (native + emulated) / 算子总数（{total}）。",
        "",
        "## 宿主侧 API（不经过后端派发，不在矩阵中）",
        "",
        "、".join(f"`{name}`" for name in matrix.host_side) + "。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="regenerate the matrix doc")
    group.add_argument("--check", action="store_true", help="verify it is current")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT,
                        help="repository root (default: this tool's repo)")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="matrix document path (default: <root>/docs/coverage.md)")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    out = args.out or (root / DOC_REL)
    try:
        matrix = build_matrix(root)
    except CoverageError as e:
        print(str(e), file=sys.stderr)
        return 1
    text = render(matrix)

    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
        return 0

    try:
        actual = out.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"{out} is missing; run tools/coverage_matrix.py --write", file=sys.stderr)
        return 1
    if actual != text:
        print(f"{out} is out of date; run tools/coverage_matrix.py --write", file=sys.stderr)
        return 1
    print("coverage matrix is up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
