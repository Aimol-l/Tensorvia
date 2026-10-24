#!/usr/bin/env python3
"""Derive a traditional header-based source tree from Tensorvia's module sources.

The .ixx / .cpp files are the single source of truth: mcpp builds them as C++23
modules.  CMake needs ordinary translation units, so this generator rewrites the
module directives mechanically into includes:

    module;                          -> dropped (GMF includes stay)
    export module tensorvia:<part>;  -> dropped, output tensorvia/<part>.h
    export module tensorvia;         -> dropped, output tensorvia.hpp
    module tensorvia;                -> #include <tensorvia.hpp>
    import :<part>;                  -> #include <tensorvia/<part>.h>
    export import :<part>;           -> #include <tensorvia/<part>.h>
    import tensorvia;                -> #include <tensorvia.hpp>
    export { ... }                   -> wrapper removed
    export <declaration>             -> export removed

Implementation units (src/**), tests, examples and benchmarks are copied with
their imports rewritten.  The output directory is an include root:

    <out>/tensorvia.hpp
    <out>/tensorvia/<part>.h
    <out>/tensorvia/{float16.hpp,bfloat16.hpp,_platform.h,_simd_math.h}
    <out>/src/... <out>/tests/... <out>/examples/... <out>/benchmarks/...

The source tree is never modified.  Outputs are written atomically and only when
their content changed; a .manifest records what this tool produced so outputs of
removed inputs are pruned on the next run.

--print-outputs lists every path the generator would write (absolute, one per
line) without touching the filesystem; CMake uses it to declare its build rule.

Usage:
    python3 tools/gen_header_mode.py --out build/gen/header
    python3 tools/gen_header_mode.py --out build/gen/header --print-outputs
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

TOOL = "gen_header"

INTERFACE_DIR = "src/interface"
# Module units that are lane facades, not partitions to convert.
FACADE_STEMS = ("header_core",)
PLAIN_HEADERS = ("float16.hpp", "bfloat16.hpp", "_platform.h", "_simd_math.h")
SOURCE_DIRS = ("src", "tests", "examples", "benchmarks")
LIBRARY_SOURCE_DIRS = ("src",)

MODULE_DECL_RE = re.compile(
    r"^\s*export\s+module\s+tensorvia(?::(?P<part>[A-Za-z_][A-Za-z0-9_]*))?\s*;\s*(?://.*)?$")
MODULE_IMPL_RE = re.compile(
    r"^\s*module\s+tensorvia(?::(?P<part>[A-Za-z_][A-Za-z0-9_]*))?\s*;\s*(?://.*)?$")
IMPORT_PART_RE = re.compile(
    r"^\s*(?:export\s+)?import\s+:(?P<part>[A-Za-z_][A-Za-z0-9_]*)\s*;\s*(?://.*)?$")
IMPORT_TOP_RE = re.compile(
    r"^\s*(?:export\s+)?import\s+tensorvia\s*;\s*(?://.*)?$")
IMPORT_ANY_RE = re.compile(r"^\s*(?:export\s+)?import\b")
EXPORT_PREFIX_RE = re.compile(r"^(\s*)export\s+")
VERSION_INIT_RE = re.compile(
    r"(inline\s+constexpr\s+std::string_view\s+version\s*=\s*)TENSORVIA_VERSION(\s*;)")


def fail(message: str) -> None:
    sys.stderr.write(f"{TOOL}: {message}\n")
    raise SystemExit(1)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_if_changed(path: Path, text: str) -> None:
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    atomic_write(path, text)


def scan_braces(line: str, depth: int, in_block_comment: bool):
    """Update the brace depth of one line, skipping comments and literals.

    Returns (depth, in_block_comment, close_col|null).  ``close_col`` is the
    index of the '}' that brought a positive depth back to zero.
    """
    close_col = None
    i, n = 0, len(line)
    while i < n:
        if in_block_comment:
            end = line.find("*/", i)
            if end == -1:
                return depth, True, close_col
            i = end + 2
            in_block_comment = False
            continue
        c = line[i]
        if c == "/" and i + 1 < n:
            if line[i + 1] == "/":
                break
            if line[i + 1] == "*":
                in_block_comment = True
                i += 2
                continue
        if c in ('"', "'"):
            quote = c
            i += 1
            while i < n:
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                close_col = i
                # Keep scanning only to validate the rest of the line.
        i += 1
    return depth, in_block_comment, close_col


def rewrite_plain_includes(line: str) -> str:
    for header in PLAIN_HEADERS:
        line = line.replace(f'#include "{header}"', f"#include <tensorvia/{header}>")
    return line


def transform_interface(path: Path, text: str, partitions: set[str],
                        version: str | None) -> tuple[str | None, str]:
    """Return (partition|None, generated header text)."""
    lines = text.splitlines()
    partition: str | None = None
    out = ["#pragma once", ""]
    in_export_block = False
    export_depth = 0
    in_block_comment = False

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()

        if in_export_block:
            export_depth, in_block_comment, close_col = scan_braces(
                line, export_depth, in_block_comment)
            if close_col is None:
                out.append(line)
                continue
            prefix = line[:close_col]
            if prefix.strip():
                out.append(prefix)
            remainder = line[close_col + 1:].strip()
            if remainder and not remainder.startswith("//"):
                fail(f"{path}:{lineno}: unexpected text after the export block: {remainder!r}")
            in_export_block = False
            continue

        if stripped.replace(" ", "") == "export{":
            export_depth, in_block_comment, _ = scan_braces(line, 0, False)
            if export_depth != 1:
                fail(f"{path}:{lineno}: malformed export block")
            in_export_block = True
            continue

        if stripped == "module;":
            continue

        match = MODULE_DECL_RE.match(line)
        if match:
            declared = match.group("part")
            if partition is not None:
                fail(f"{path}:{lineno}: multiple module declarations")
            partition = declared
            continue

        match = IMPORT_PART_RE.match(line)
        if match:
            dep = match.group("part")
            if dep not in partitions:
                fail(f"{path}:{lineno}: unknown partition ':{dep}'")
            out.append(f"#include <tensorvia/{dep}.h>")
            continue

        if IMPORT_TOP_RE.match(line):
            fail(f"{path}:{lineno}: an interface unit cannot import the primary module")

        if IMPORT_ANY_RE.match(line):
            fail(f"{path}:{lineno}: unrecognized import directive: {line.strip()!r}")

        if EXPORT_PREFIX_RE.match(line):
            line = EXPORT_PREFIX_RE.sub(r"\1", line)

        if stripped.startswith("module ") or stripped == "module":
            fail(f"{path}:{lineno}: unrecognized module directive: {line.strip()!r}")

        out.append(rewrite_plain_includes(line))

    if in_export_block:
        fail(f"{path}: unterminated export block")
    if partition is None and path.name != "tensorvia.ixx":
        fail(f"{path}: no 'export module tensorvia...;' declaration")

    result = "\n".join(out) + "\n"
    if partition == "version" and version:
        if not VERSION_INIT_RE.search(result):
            fail(f"{path}: could not find the version initializer to bake")
        result = VERSION_INIT_RE.sub(rf'\1"{version}"\2', result)
    return partition, result


def transform_source(path: Path, text: str, partitions: set[str]) -> str:
    """Rewrite one implementation/consumer translation unit."""
    out: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()

        if stripped == "module;":
            continue

        match = MODULE_IMPL_RE.match(line)
        if match:
            part = match.group("part")
            if part is None:
                out.append("#include <tensorvia.hpp>")
            else:
                if part not in partitions:
                    fail(f"{path}:{lineno}: unknown partition ':{part}'")
                out.append(f"#include <tensorvia/{part}.h>")
            continue

        match = IMPORT_PART_RE.match(line)
        if match:
            part = match.group("part")
            if part not in partitions:
                fail(f"{path}:{lineno}: unknown partition ':{part}'")
            out.append(f"#include <tensorvia/{part}.h>")
            continue

        if IMPORT_TOP_RE.match(line):
            out.append("#include <tensorvia.hpp>")
            continue

        if IMPORT_ANY_RE.match(line):
            fail(f"{path}:{lineno}: unrecognized import directive: {stripped!r}")

        out.append(rewrite_plain_includes(line))

    return "\n".join(out) + "\n"


def discover(root: Path, source_dirs=SOURCE_DIRS):
    interface_root = root / INTERFACE_DIR
    # header_core is the CUDA lane's facade: it declares `export module
    # tensorvia:core` on top of <tensorvia.hpp>, competing with core.ixx, and it
    # is compiled as a module unit rather than converted. It is never part of
    # the generated tree.
    interface_paths = sorted(p for p in interface_root.rglob("*.ixx")
                             if p.stem not in FACADE_STEMS)
    if not interface_paths:
        fail(f"no .ixx interfaces found under {interface_root}")

    parsed: list[tuple[Path, str | None, str]] = []
    partitions: dict[str, Path] = {}
    for path in interface_paths:
        text = path.read_text(encoding="utf-8")
        partition = None
        for lineno, line in enumerate(text.splitlines(), 1):
            match = MODULE_DECL_RE.match(line)
            if match:
                partition = match.group("part")
                break
            if line.strip() == "module;":
                continue
        if partition is None and path.name != "tensorvia.ixx":
            fail(f"{path}: no 'export module tensorvia...;' declaration")
        if partition is not None:
            if partition in partitions:
                fail(f"{path}: duplicate partition ':{partition}' "
                     f"(already defined by {partitions[partition]})")
            partitions[partition] = path
        parsed.append((path, partition, text))

    if not any(part is None for _, part, _ in parsed):
        fail("primary module interface tensorvia.ixx not found")

    sources: list[Path] = []
    for directory in source_dirs:
        base = root / directory
        if base.is_dir():
            sources.extend(sorted(base.rglob("*.cpp")))
    return parsed, set(partitions), sources


def build_outputs(root: Path, out: Path, version: str | None,
                  source_dirs=SOURCE_DIRS, software_float16: bool = False):
    parsed, partitions, sources = discover(root, source_dirs)

    headers: dict[str, str] = {}
    umbrella: str | None = None
    for path, partition, text in parsed:
        part, generated = transform_interface(path, text, partitions, version)
        if part is None:
            umbrella = generated
        else:
            headers[part] = generated

    if umbrella is None:
        fail("primary module interface tensorvia.ixx not found")

    if software_float16:
        # The header-first (CUDA) artifact compiles host TUs as C++23 but nvcc
        # TUs as C++20, where std::float16_t does not exist. Pin the software
        # type in the one header everyone includes so the artifact has a single
        # float16 ABI.
        types_header = headers.get("types")
        if types_header is None:
            fail("no :types partition to pin software float16 in")
        marker = "#pragma once\n"
        if not types_header.startswith(marker):
            fail("generated types header does not start with '#pragma once'")
        headers["types"] = (marker + "\n#define TENSORVIA_FORCE_SOFTWARE_FLOAT16 1\n"
                            + types_header[len(marker):])

    files: dict[Path, str] = {out / "tensorvia.hpp": umbrella}
    for part, text in headers.items():
        files[out / "tensorvia" / f"{part}.h"] = text
    for header in PLAIN_HEADERS:
        source = root / "include" / header
        if not source.is_file():
            fail(f"missing plain header {source}")
        files[out / "tensorvia" / header] = source.read_text(encoding="utf-8")
    for source in sources:
        rel = source.relative_to(root)
        files[out / rel] = transform_source(source, source.read_text(encoding="utf-8"),
                                            partitions)

    return files


def prune(out: Path, produced: set[Path]) -> None:
    manifest = out / ".manifest"
    try:
        previous = {out / line for line in manifest.read_text(encoding="utf-8").splitlines()
                    if line}
    except OSError:
        previous = set()
    for stale in sorted(previous - produced):
        try:
            stale.unlink()
        except OSError:
            continue
    # Drop directories left empty by the pruning, deepest first.
    for directory in sorted((p for p in out.rglob("*") if p.is_dir()),
                            key=lambda p: len(p.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate header-mode sources from modules.")
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parent.parent,
                        help="repository root (default: the parent of tools/)")
    parser.add_argument("--out", type=Path, required=True,
                        help="output include root, e.g. <build>/gen/header")
    parser.add_argument("--project-version", default=None,
                        help="version baked into tensorvia/version.h")
    parser.add_argument("--float16", choices=("auto", "software"), default="auto",
                        help="pin the software float16 type in the generated types header "
                             "(required by the header-first CUDA lane)")
    parser.add_argument("--library-only", action="store_true",
                        help="only scan src/ (skip tests, examples and benchmarks)")
    parser.add_argument("--print-outputs", action="store_true",
                        help="print output paths instead of writing")
    args = parser.parse_args()

    root = args.source_root.resolve()
    out = args.out.resolve()
    files = build_outputs(
        root, out, args.project_version,
        source_dirs=LIBRARY_SOURCE_DIRS if args.library_only else SOURCE_DIRS,
        software_float16=args.float16 == "software")

    if args.print_outputs:
        for path in sorted(files):
            print(path)
        return 0

    for path, text in sorted(files.items()):
        write_if_changed(path, text)
    produced = set(files)
    prune(out, produced)
    write_if_changed(
        out / ".manifest",
        "".join(path.relative_to(out).as_posix() + "\n" for path in sorted(produced)))
    print(f"{TOOL}: generated {len(files)} files in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
