#!/usr/bin/env python3
"""Build and package Tensorvia through the CMake lane.

One entry point covers both halves of the story:

    configure -> build -> (ctest) -> install          # -b / -test
    source tarball -> cmake -> DESTDIR staging -> dist  # -pkg

Usage:
    python3 build.py -b cpu                        # 编译并安装到 build-cpu/install
    python3 build.py -b cpu -test on               # 编译 + ctest + 安装
    python3 build.py -b cuda -pkg deb              # 编译，再打 deb 包
    python3 build.py -b vulkan -test on -pkg arch  # 编译 + 测试 + pacman 包
    python3 build.py -b sycl -pkg all              # 一次出 arch/deb/rpm 三种包

一次只编一个后端（CPU 始终在内）。产物目录：

    build-<backend>/           CMake 构建树
    build-<backend>/install/   cmake --install 的前缀（仅非打包路径）
    build-pkg/                 打包中间物：源码 tarball、PKGBUILD、spec、staging
    dist/                      最终安装包
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
WORK_DIR = ROOT / "build-pkg"
PROJECT = "tensorvia"
VERSION_RE = re.compile(r'project\(\s*tensorvia\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)',
                        re.IGNORECASE | re.DOTALL)
URL = "https://github.com/Aimol-l/Tensorvia"
SUMMARY = "C++23 module tensor library with CPU, Vulkan, CUDA and SYCL backends"
LICENSE = "MIT"

# Package names carry the backend so the variants can be told apart in a repo:
# tensorvia-cpu, tensorvia-vk, tensorvia-cuda, tensorvia-sycl.
BACKEND_SUFFIX = {"cpu": "cpu", "vulkan": "vk", "cuda": "cuda", "sycl": "sycl"}
BACKEND_ALIASES = {"vk": "vulkan"}
SUPPORTED_BACKENDS = ("cpu", "vulkan", "cuda", "sycl")

# Fallback when no NVIDIA GPU can be queried (matches CMakeLists.txt's default).
DEFAULT_CUDA_ARCH = "89"

ARCH_DEPS = {
    "cpu":    (("gcc-libs",), ("cmake", "gcc", "ninja")),
    "vulkan": (("gcc-libs", "vulkan-icd-loader"),
               ("cmake", "gcc", "ninja", "glslang", "vulkan-headers", "python")),
    "cuda":   (("gcc-libs", "cuda"),
               ("cmake", "gcc", "ninja", "cuda")),
    "sycl":   (("gcc-libs", "intel-oneapi-runtime-dpcpp-cpp"),
               ("cmake", "gcc", "ninja", "intel-oneapi-dpcpp-cpp")),
}
DEB_DEPS = {
    "cpu":    ("libc6, libgomp1", "cmake, g++, ninja-build"),
    "vulkan": ("libc6, libgomp1, libvulkan1",
               "cmake, g++, ninja-build, glslang-tools, libvulkan-dev, python3"),
    "cuda":   ("libc6, libgomp1, cuda-cudart",
               "cmake, g++, ninja-build, nvidia-cuda-toolkit, python3"),
    "sycl":   ("libc6, libgomp1, intel-oneapi-runtime-dpcpp-cpp",
               "cmake, g++, ninja-build, python3, intel-oneapi-compiler-dpcpp-cpp"),
}
RPM_DEPS = {
    "cpu":    ("gcc-libs", "cmake, gcc-c++, ninja-build"),
    "vulkan": ("gcc-libs, vulkan-loader",
               "cmake, gcc-c++, ninja-build, glslang, vulkan-loader-devel, python3"),
    "cuda":   ("gcc-libs, cuda-cudart",
               "cmake, gcc-c++, ninja-build, cuda-toolkit, python3"),
    "sycl":   ("gcc-libs, intel-oneapi-runtime-dpcpp-cpp",
               "cmake, gcc-c++, ninja-build, python3, intel-oneapi-compiler-dpcpp-cpp"),
}

# Filled by --dry-run: print each command instead of running it.
DRY_RUN = False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    printable = " ".join(str(c) for c in cmd)
    print("+", printable, flush=True)
    if DRY_RUN:
        return
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def capture(cmd: list[str]) -> str | None:
    """Run a probe; None when the tool is missing or the call fails."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def require(*tools: str) -> None:
    if DRY_RUN:  # nothing is executed, so a missing tool is not a blocker
        return
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise SystemExit(f"❌ missing tool(s): {', '.join(missing)}")


def read_version() -> str:
    m = VERSION_RE.search((ROOT / "CMakeLists.txt").read_text())
    if not m:
        raise SystemExit("❌ project(VERSION x.y.z) not found in CMakeLists.txt")
    return m.group(1)


def normalize_backend(backend: str) -> str:
    backend = BACKEND_ALIASES.get(backend, backend)
    if backend not in SUPPORTED_BACKENDS:
        raise SystemExit(
            f"❌ backend '{backend}' is not implemented in this project "
            f"(supported: {', '.join(SUPPORTED_BACKENDS)})")
    return backend


def package_name(backend: str) -> str:
    return f"{PROJECT}-{BACKEND_SUFFIX[backend]}"


def other_package_names(backend: str) -> list[str]:
    """Sibling variants that install the same files and therefore conflict."""
    return [package_name(b) for b in SUPPORTED_BACKENDS if b != backend]


def host_arch() -> tuple[str, str, str]:
    """Return (tar/arch, deb, rpm) architecture triples for this machine."""
    mapping = {
        "x86_64":  ("x86_64", "amd64", "x86_64"),
        "amd64":   ("x86_64", "amd64", "x86_64"),
        "aarch64": ("aarch64", "arm64", "aarch64"),
        "arm64":   ("aarch64", "arm64", "aarch64"),
    }
    machine = platform.machine().lower()
    if machine not in mapping:
        raise SystemExit(f"❌ unsupported architecture: {machine}")
    return mapping[machine]


def recipe_flags(backend: str, indent: str, extra: tuple[str, ...] = ()) -> str:
    """Render the configure flags for a shell recipe (PKGBUILD / rpm spec).

    A flag whose value contains spaces has to be quoted or the shell splits it
    into two arguments; `-DCMAKE_CXX_FLAGS=-fsycl -fsycl-targets=…` is exactly
    that case for the SYCL lane.
    """
    flags = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=/usr",
        *extra,
        "-DBUILD_TEST=OFF",
        "-DBUILD_BENCH=OFF",
        *backend_cmake_flags(backend),
    ]
    rendered = [f"'{f}'" if " " in f else f for f in flags]
    return indent + (" \\\n" + indent).join(rendered)


# ---------------------------------------------------------------------------
# Toolchain probes
# ---------------------------------------------------------------------------

_cuda_arch: str | None = None


def nvidia_compute_cap() -> str | None:
    """Compute capability as a CMake architecture number ('8.9' -> '89')."""
    out = capture(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if not out:
        return None
    first = out.splitlines()[0].strip()
    return first.replace(".", "") or None


def cuda_arch() -> str:
    global _cuda_arch
    if _cuda_arch is None:
        detected = nvidia_compute_cap()
        _cuda_arch = detected or DEFAULT_CUDA_ARCH
        print(f"ℹ️  CUDA architecture: {_cuda_arch} "
              f"({'nvidia-smi' if detected else 'default'})")
    return _cuda_arch


def sycl_openmp_flags() -> list[str]:
    """Point FindOpenMP at icpx's own runtime.

    icpx normally probes fine, but the probe depends on oneAPI's layout and on
    `libiomp5.so` being discoverable; pinning the three cache entries makes the
    configure deterministic across oneAPI releases.
    """
    if shutil.which("icpx") is None:
        return []
    printed = capture(["icpx", "-print-file-name=libiomp5.so"])
    if not printed:
        return []
    lib = Path(printed)
    if not lib.is_file():
        return []
    return [
        "-DOpenMP_CXX_FLAGS=-fiopenmp",
        "-DOpenMP_CXX_LIB_NAMES=iomp5",
        f"-DOpenMP_iomp5_LIBRARY={lib.resolve()}",
    ]


def sycl_device_flags() -> list[str]:
    """Compile device code for the GPU that is actually present.

    Plain `-fsycl` only emits SPIR-V (spir64), which the CUDA adapter cannot
    run — an NVIDIA machine needs an nvptx64 image built for its own SM. With
    no NVIDIA GPU the spir64 default is what we want, so no flags are added.
    """
    arch = nvidia_compute_cap()
    if arch is None:
        print("ℹ️  SYCL target: spir64 (no NVIDIA GPU detected)")
        return []
    print(f"ℹ️  SYCL target: nvptx64-nvidia-cuda sm_{arch}")
    device = ("-fsycl -fsycl-targets=nvptx64-nvidia-cuda "
              f"-Xsycl-target-backend --cuda-gpu-arch=sm_{arch}")
    # The shared library links the device image, and the test executables link
    # the library, so the flag belongs on every link line as well.
    return [
        f"-DCMAKE_CXX_FLAGS={device}",
        f"-DCMAKE_SHARED_LINKER_FLAGS={device}",
        f"-DCMAKE_EXE_LINKER_FLAGS={device}",
    ]


# ---------------------------------------------------------------------------
# Configure flags — the single source of truth shared by build and packaging
# ---------------------------------------------------------------------------

def backend_cmake_flags(backend: str) -> list[str]:
    flags: list[str] = []
    if backend == "vulkan":
        flags.append("-DBACKEND_VULKAN=ON")
    elif backend == "cuda":
        flags += ["-DBACKEND_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch()}"]
    elif backend == "sycl":
        flags += ["-DBACKEND_SYCL=ON", "-DCMAKE_CXX_COMPILER=icpx"]
        flags += sycl_openmp_flags()
        flags += sycl_device_flags()
    return flags


def configure_args(backend: str, build: Path, prefix: str, tests: bool) -> list[str]:
    return [
        "cmake", "-G", "Ninja", "-S", str(ROOT), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        f"-DBUILD_TEST={'ON' if tests else 'OFF'}",
        "-DBUILD_BENCH=OFF",
        *backend_cmake_flags(backend),
    ]


def runtime_env(backend: str) -> dict:
    """Environment for running the tests / bench of this backend."""
    env = dict(os.environ)
    if backend == "sycl" and nvidia_compute_cap() is not None:
        # Without a filter the default selector may land on the OpenCL CPU
        # device, which is not what this lane is meant to exercise.
        env.setdefault("SYCL_DEVICE_FILTER", "cuda:*")
    return env


# ---------------------------------------------------------------------------
# Shader compilation (SPIR-V registry) — the Vulkan lane's half of the build
#
# Inlined from the former tools/gen_spv.py so the CMake lane has exactly one
# entry point: CMake only CONSUMES <build>/gen/spirv/spv_registry.cpp and
# refuses to configure without it (see CMakeLists.txt), so this program is
# what compiles the shaders. `-b vulkan` runs it before every configure; the
# packaging recipes (PKGBUILD / spec / deb staging) call `-gen-spv` directly,
# because they run cmake without this program's main flow.
# ---------------------------------------------------------------------------

def find_shaders(shader_root: Path) -> list[Path]:
    return sorted(shader_root.glob("*/*.comp"))


def compile_one(shader: Path, spv_dir: Path, glslang: str) -> Path:
    out_path = spv_dir / (shader.stem + ".spv")
    cmd = [
        glslang,
        "-V",
        "--target-env",
        "vulkan1.4",
        str(shader),
        "-o",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(
            f"build.py: failed to compile {shader}\n{proc.stdout}{proc.stderr}"
        )
        raise SystemExit(1)
    if not out_path.is_file() or out_path.stat().st_size == 0:
        sys.stderr.write(f"build.py: {shader} produced no SPIR-V\n")
        raise SystemExit(1)
    return out_path


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic output, so an unchanged shader set yields identical bytes:
    # skipping the write keeps the mtime stable and ninja from recompiling
    # spv_registry.cpp (and relinking) on every `-b vulkan`.
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except (OSError, UnicodeDecodeError):
        pass
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


def emit_registry(spv_files: list[Path], out_dir: Path) -> None:
    """Write the registry translation unit."""
    entries = []
    for spv in sorted(spv_files, key=lambda p: p.stem):
        data = spv.read_bytes()
        if len(data) % 4 != 0:
            raise SystemExit(f"build.py: {spv} size {len(data)} is not a multiple of 4")
        words = [
            int.from_bytes(data[i : i + 4], "little")
            for i in range(0, len(data), 4)
        ]
        entries.append((spv.stem, words))

    cpp = [
        "// Generated by build.py -gen-spv for the CMake lane -- do not edit.",
        "//",
        "// The mcpp lane does not use this file: build.mcpp generates the same",
        "// table from mcpp.rules.spirv's surface, which is what removes both Python",
        "// and a host shader compiler from that build. The interface is shared --",
        "// include/tensorvia/backend/vulkan/spv_registry.hpp -- so a source file",
        "// cannot tell which lane produced the bytes.",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <stdexcept>",
        "#include <string>",
        "#include <string_view>",
        "#include <unordered_map>",
        "",
        '#include "tensorvia/backend/vulkan/spv_registry.hpp"',
        "",
        "namespace vkspv {",
        "namespace {",
        "",
    ]
    for key, words in entries:
        # 12 words per line keeps the generated file readable without making
        # the compiler work through one enormous line.
        cpp.append(f"const std::uint32_t {key}_data[] = {{")
        for i in range(0, len(words), 12):
            chunk = ", ".join(f"0x{w:08x}u" for w in words[i : i + 12])
            cpp.append(f"    {chunk},")
        cpp.append("};")
        cpp.append("")

    cpp.append("struct entry {")
    cpp.append("    const std::uint32_t* code;")
    cpp.append("    std::size_t          words;")
    cpp.append("};")
    cpp.append("")
    cpp.append("const std::unordered_map<std::string_view, entry>& table() {")
    cpp.append("    static const std::unordered_map<std::string_view, entry> t = {")
    for key, words in entries:
        cpp.append(f'        {{"{key}", {{{key}_data, {len(words)}}}}},')
    cpp.append("    };")
    cpp.append("    return t;")
    cpp.append("}")
    cpp.append("")
    cpp.append("}  // namespace")
    cpp.append("")
    cpp.append("spv_view get_spv_code(std::string_view key) {")
    cpp.append("    const auto& t  = table();")
    cpp.append("    const auto  it = t.find(key);")
    cpp.append("    if (it == t.end()) {")
    cpp.append('        throw std::runtime_error("SPIR-V not found: " + std::string(key));')
    cpp.append("    }")
    cpp.append("    return {it->second.code, it->second.words};")
    cpp.append("}")
    cpp.append("")
    cpp.append("}  // namespace vkspv")
    cpp.append("")

    # Only the translation unit: the declarations are hand-written and shared
    # with the mcpp lane, so there is no generated header to keep in step.
    atomic_write(out_dir / "spirv" / "spv_registry.cpp", "\n".join(cpp))


def gen_spv(out: Path, shader_root: Path, glslang: str,
            jobs: int | None = None) -> int:
    shaders = find_shaders(shader_root)
    if not shaders:
        sys.stderr.write(f"build.py: no .comp shaders under {shader_root}\n")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tv-spv-", dir=out) as tmp:
        spv_dir = Path(tmp)
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=jobs or (os.cpu_count() or 1)) as pool:
            futures = [
                pool.submit(compile_one, shader, spv_dir, glslang)
                for shader in shaders
            ]
            try:
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            except FileNotFoundError:
                sys.stderr.write(
                    f"build.py: glslang compiler not found: {glslang}\n"
                    "         install glslang, add it to PATH, or pass "
                    "-glslang <path>\n"
                )
                return 1
        emit_registry(list(spv_dir.glob("*.spv")), out)

    print(f"✅ SPIR-V: {len(shaders)} shaders -> {out / 'spirv'}")
    return 0


def require_glslang() -> str:
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        raise SystemExit(
            "❌ glslangValidator not found on PATH; the Vulkan lane compiles its "
            "shaders with it (install glslang / glslang-tools), or pass "
            "-glslang <path> to -gen-spv")
    return glslang


def ensure_spv(build: Path) -> None:
    """Compile the shader set into <build>/gen before configure.

    CMake consumes the registry and nothing else, so every `-b vulkan`
    regenerates it from the current tree; atomic_write keeps an unchanged
    shader set from triggering a recompile.
    """
    if gen_spv(build / "gen", ROOT / "shader", require_glslang()) != 0:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_dir(backend: str) -> Path:
    return ROOT / f"build-{backend}"


def build(backend: str, tests: bool, install: bool) -> Path:
    build = build_dir(backend)
    prefix = build / "install"
    if backend == "vulkan":
        print(f"\n=== shaders ({backend}) -> {build / 'gen'} ===")
        ensure_spv(build)

    print(f"\n=== configure ({backend}) -> {build} ===")
    run(configure_args(backend, build, str(prefix), tests))

    print(f"\n=== build ({backend}) ===")
    run(["cmake", "--build", str(build), "--parallel"])

    if tests:
        print(f"\n=== ctest ({backend}) ===")
        run(["ctest", "--test-dir", str(build), "--output-on-failure"],
            env=runtime_env(backend))

    if install:
        print(f"\n=== install -> {prefix} ===")
        run(["cmake", "--install", str(build), "--prefix", str(prefix)])

    return build


# ---------------------------------------------------------------------------
# Packaging: source tarball -> cmake -> DESTDIR staging -> dist/
# ---------------------------------------------------------------------------

def make_source_tarball(dest: Path, version: str) -> Path:
    """Archive the working tree as tensorvia-<version>/. Distro builds never
    touch the development tree, and generated artifacts cannot leak in."""
    prefix = f"{PROJECT}-{version}"
    if not DRY_RUN:
        dest.parent.mkdir(parents=True, exist_ok=True)
    excludes = [
        "--exclude=./.git",
        "--exclude=./build",
        "--exclude=./build-*",
        "--exclude=./target",
        "--exclude=./dist",
        "--exclude=./.cache",
        "--exclude=./.vscode",
        "--exclude=__pycache__",
        "--exclude=compile_commands.json",
        "--exclude=*.o",
        "--exclude=*.so",
        "--exclude=*.so.*",
        "--exclude=*.a",
        "--exclude=*.deb",
        "--exclude=*.rpm",
        "--exclude=*.pkg.tar.*",
        "--exclude=*.tar.gz",
    ]
    run(["tar", "-czf", str(dest), *excludes,
         f"--transform=s,^\\./,{prefix}/,",
         "-C", str(ROOT), "."])
    return dest


def package_arch(backend: str) -> None:
    require("makepkg", "fakeroot", "tar", "cmake", "ninja")
    version = read_version()
    work = WORK_DIR / "arch"
    if not DRY_RUN:
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        DIST_DIR.mkdir(exist_ok=True)

    tarball = work / f"{PROJECT}-{version}.tar.gz"
    print(f"📦 source tarball -> {tarball}")
    make_source_tarball(tarball, version)

    depends, makedepends = ARCH_DEPS[backend]
    arch, _, _ = host_arch()
    # The Vulkan lane's shaders are compiled by build.py before cmake runs;
    # CMake itself consumes <build>/gen/spirv/spv_registry.cpp only.
    spv_step = ""
    if backend == "vulkan":
        spv_step = ("    python3 build.py -gen-spv --out build/gen "
                    "--shader-root shader\n")
    pkgbuild = f"""# Generated by build.py -pkg arch. Do not edit by hand.
pkgname={package_name(backend)}
pkgver={version}
pkgrel=1
pkgdesc="{SUMMARY} ({backend} backend)"
arch=('{arch}')
url="{URL}"
license=('{LICENSE}')
provides=('{PROJECT}')
conflicts=({' '.join(f"'{n}'" for n in other_package_names(backend))})
depends=({' '.join(f"'{d}'" for d in depends)})
makedepends=({' '.join(f"'{d}'" for d in makedepends)})
source=("{PROJECT}-$pkgver.tar.gz")
sha256sums=('SKIP')

build() {{
    cd "{PROJECT}-$pkgver"
{spv_step}    cmake -G Ninja -S . -B build \\
{recipe_flags(backend, '        ')}
    cmake --build build --parallel
}}

package() {{
    cd "{PROJECT}-$pkgver"
    DESTDIR="$pkgdir" cmake --install build
}}
"""
    print(f"🔨 makepkg ({backend} backend)")
    if DRY_RUN:
        print(pkgbuild)
        return
    (work / "PKGBUILD").write_text(pkgbuild)
    run(["makepkg", "-f", "--noconfirm"], cwd=work)

    produced = sorted(work.glob("*.pkg.tar.*"))
    if not produced:
        raise SystemExit("❌ makepkg produced no package")
    for pkg in produced:
        target = DIST_DIR / pkg.name
        shutil.move(str(pkg), str(target))
        print(f"✅ {target}")


def build_into_staging(backend: str, version: str, work: Path) -> Path:
    src = work / "src"
    if not DRY_RUN:
        if src.exists():
            shutil.rmtree(src)
        src.mkdir(parents=True)

    tarball = work / f"{PROJECT}-{version}.tar.gz"
    print(f"📦 source tarball -> {tarball}")
    make_source_tarball(tarball, version)
    run(["tar", "-xzf", str(tarball), "-C", str(src)])

    project = src / f"{PROJECT}-{version}"
    build_tree = project / "build"
    if backend == "vulkan":
        # CMake only consumes the registry, so compile the shaders here --
        # exactly what `-b vulkan` does on the development tree.
        print(f"\n=== shaders ({backend}) -> {build_tree / 'gen'} ===")
        run([sys.executable, str(project / "build.py"), "-gen-spv",
             "--out", str(build_tree / "gen"),
             "--shader-root", str(project / "shader")], cwd=project)
    run(configure_args(backend, build_tree, "/usr", tests=False))
    run(["cmake", "--build", str(build_tree), "--parallel"])

    staging = work / "staging"
    if not DRY_RUN and staging.exists():
        shutil.rmtree(staging)
    print(f"📂 staging install -> {staging}")
    run(["cmake", "--install", str(build_tree), "--prefix", "/usr"],
        env=dict(os.environ, DESTDIR=str(staging)))
    return staging


def package_deb(backend: str) -> None:
    require("dpkg-deb", "tar", "cmake", "ninja")
    version = read_version()
    _, deb_arch, _ = host_arch()
    work = WORK_DIR / "deb"
    if not DRY_RUN:
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        DIST_DIR.mkdir(exist_ok=True)

    staging = build_into_staging(backend, version, work)
    depends, _ = DEB_DEPS[backend]
    name = package_name(backend)
    ctrl = staging / "DEBIAN"
    control = (
        f"Package: {name}\n"
        f"Version: {version}-1\n"
        f"Section: libs\n"
        f"Priority: optional\n"
        f"Architecture: {deb_arch}\n"
        f"Depends: {depends}\n"
        f"Provides: {PROJECT}\n"
        f"Conflicts: {', '.join(other_package_names(backend))}\n"
        f"Maintainer: Aimol-l <noreply@example.com>\n"
        f"Description: {SUMMARY}\n"
        f" ({backend} backend)\n")
    if DRY_RUN:
        print(control)
    else:
        ctrl.mkdir(exist_ok=True)
        (ctrl / "control").write_text(control)

    out = DIST_DIR / f"{name}_{version}-1_{deb_arch}.deb"
    run(["dpkg-deb", "--root-owner-group", "--build", str(staging), str(out)])
    print(f"✅ {out}")


def package_rpm(backend: str) -> None:
    require("rpmbuild", "tar", "cmake", "ninja")
    version = read_version()
    work = WORK_DIR / "rpm"
    if not DRY_RUN:
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        DIST_DIR.mkdir(exist_ok=True)

    tarball = work / f"{PROJECT}-{version}.tar.gz"
    print(f"📦 source tarball -> {tarball}")
    make_source_tarball(tarball, version)

    requires, build_requires = RPM_DEPS[backend]
    # %{_lib} is an rpmbuild macro; keep it out of the f-string so the braces
    # are not read as replacement fields.
    libdir_flag = ("-DCMAKE_INSTALL_LIBDIR=%{_lib}",)
    # The Vulkan lane's shaders are compiled by build.py before cmake runs;
    # CMake itself consumes <build>/gen/spirv/spv_registry.cpp only.
    spv_step = ""
    if backend == "vulkan":
        spv_step = ("python3 build.py -gen-spv --out build/gen "
                    "--shader-root shader\n")
    spec = f"""Name:           {package_name(backend)}
Version:        {version}
Release:        1%{{?dist}}
Summary:        {SUMMARY}
License:        {LICENSE}
URL:            {URL}
Source0:        {PROJECT}-%{{version}}.tar.gz
BuildRequires:  {build_requires}
Requires:       {requires}
Provides:       {PROJECT}
Conflicts:      {', '.join(other_package_names(backend))}

%description
{SUMMARY} ({backend} backend).

%prep
# The package name carries the backend, the source archive does not.
%setup -q -n {PROJECT}-%{{version}}

%build
{spv_step}cmake -G Ninja -S . -B build \\
{recipe_flags(backend, '    ', libdir_flag)}
cmake --build build --parallel

%install
DESTDIR=%{{buildroot}} cmake --install build --prefix /usr

%files
%{{_libdir}}/libtensorvia.so*
%{{_libdir}}/cmake/tensorvia
%{{_includedir}}/tensorvia.hpp
%{{_includedir}}/tensorvia
%license LICENSE
"""
    specs = work / "SPECS"
    spec_file = specs / f"{PROJECT}.spec"
    if DRY_RUN:
        print(spec)
        return
    specs.mkdir()
    spec_file.write_text(spec)

    # rpmbuild macro directories must be absolute paths.
    topdir = (work / "rpmbuild").resolve()
    run([
        # --nodeps keeps the build working in non-RPM containers (the Requires
        # metadata is still written into the package). Drop it on Fedora/openSUSE
        # if you want BuildRequires enforced.
        "rpmbuild", "-bb", "--nodeps",
        "--define", f"_topdir {topdir}",
        "--define", f"_sourcedir {work.resolve()}",
        "--define", f"_builddir {topdir}/BUILD",
        str(spec_file.resolve()),
    ])

    rpms = sorted((topdir / "RPMS").rglob("*.rpm"))
    if not rpms:
        raise SystemExit("❌ rpmbuild produced no .rpm")
    for pkg in rpms:
        target = DIST_DIR / pkg.name
        shutil.move(str(pkg), str(target))
        print(f"✅ {target}")


PACKAGERS = {"arch": package_arch, "deb": package_deb, "rpm": package_rpm}


def main() -> int:
    global DRY_RUN
    parser = argparse.ArgumentParser(
        description="Build and package Tensorvia (CMake lane).")
    parser.add_argument("-b", "--backend",
                        choices=["cpu", "vulkan", "vk", "cuda", "sycl"],
                        default="cpu",
                        help="backend compiled into the build (default: cpu)")
    parser.add_argument("-gen-spv", "--gen-spv", action="store_true",
                        dest="gen_spv",
                        help="standalone mode: compile shader/**/*.comp and emit "
                             "<out>/spirv/spv_registry.cpp, the registry CMake "
                             "consumes (the Vulkan lane runs this before every "
                             "configure)")
    parser.add_argument("--out", type=Path,
                        help="output root for -gen-spv (writes <out>/spirv)")
    parser.add_argument("--shader-root", type=Path,
                        help="shader tree for -gen-spv (default: <repo>/shader)")
    parser.add_argument("--glslang",
                        help="glslangValidator path for -gen-spv "
                             "(default: PATH lookup)")
    parser.add_argument("-test", "--test", dest="test", choices=["on", "off"],
                        default="off",
                        help="build and run the tests/ suite (default: off)")
    parser.add_argument("-pkg", "--package", dest="package",
                        choices=["arch", "deb", "rpm", "all"],
                        help="also produce distro packages in dist/")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="print the commands instead of running them")
    args = parser.parse_args()

    DRY_RUN = args.dry_run
    backend = normalize_backend(args.backend)
    tests = args.test == "on"

    if args.gen_spv:
        if args.out is None:
            raise SystemExit("❌ -gen-spv requires --out <dir>")
        if tests or args.package:
            raise SystemExit("❌ -gen-spv is a standalone mode; drop -test/-pkg")
        print(f"🔧 {PROJECT} | -gen-spv | out={args.out}")
        return gen_spv(args.out, args.shader_root or ROOT / "shader",
                       args.glslang or require_glslang())

    print(f"🔧 {PROJECT} {read_version()} | backend={backend} | tests={'on' if tests else 'off'}"
          f" | package={args.package or '-'}")
    build(backend, tests=tests, install=args.package is None)

    if args.package:
        targets = ["arch", "deb", "rpm"] if args.package == "all" else [args.package]
        for target in targets:
            print(f"\n=== {target} ({backend} -> {package_name(backend)}) ===")
            PACKAGERS[target](backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
