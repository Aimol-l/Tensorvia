#!/usr/bin/env python3
"""Build distribution packages for Tensorvia (pacman / deb / rpm).

Every packager follows the same pipeline so the artifacts are reproducible:

    workspace -> source tarball -> cmake -DCMAKE_INSTALL_PREFIX=/usr
              -> cmake --install (DESTDIR staging) -> dist/<package>

The source tarball contains the working tree (committed or not) with build
directories, VCS metadata and previous artifacts removed. ``arch`` lets
makepkg drive the build; ``deb`` and ``rpm`` reuse one shared staging build.

Usage:
    python3 tools/package.py arch --backend vulkan
    python3 tools/package.py deb
    python3 tools/package.py rpm --backend vulkan
    python3 tools/package.py all
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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
# Backends this project can actually build today.
SUPPORTED_BACKENDS = ("cpu", "vulkan", "cuda", "sycl")

# Runtime / build-time dependencies per backend, per distro family.
# Python 3 is a build-time need of the Vulkan backend only: CMake runs
# tools/gen_spv.py to compile the shaders into the SPIR-V registry.
ARCH_DEPS = {
    "cpu":    (("gcc-libs",), ("cmake", "gcc", "ninja")),
    "vulkan": (("gcc-libs", "vulkan-icd-loader"),
               ("cmake", "gcc", "ninja", "glslang", "vulkan-headers", "python")),
    # cuda carries the toolkit (nvcc + cudart) on Arch.
    "cuda":   (("gcc-libs", "cuda"),
               ("cmake", "gcc", "ninja", "cuda")),
    # SYCL needs the DPC++ compiler (icpx) at build time and the SYCL runtime
    # at run time; both live in the oneAPI packages.
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


def run(cmd: list[str], **kwargs) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def require(*tools: str) -> None:
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
    machine = platform.machine().lower()
    mapping = {
        "x86_64":  ("x86_64", "amd64", "x86_64"),
        "amd64":   ("x86_64", "amd64", "x86_64"),
        "aarch64": ("aarch64", "arm64", "aarch64"),
        "arm64":   ("aarch64", "arm64", "aarch64"),
    }
    if machine not in mapping:
        raise SystemExit(f"❌ unsupported architecture: {machine}")
    return mapping[machine]


def make_source_tarball(dest: Path, version: str) -> Path:
    """Archive the working tree as tensorvia-<version>/. Distro builds never
    touch the development tree, and generated artifacts cannot leak in."""
    prefix = f"{PROJECT}-{version}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    excludes = [
        # build trees (CMake, mcpp, packaging) and VCS data
        "--exclude=./.git",
        "--exclude=./build",
        "--exclude=./build-*",
        "--exclude=./target",
        "--exclude=./dist",
        "--exclude=./.cache",
        "--exclude=./.vscode",
        "--exclude=__pycache__",
        "--exclude=compile_commands.json",
        # stray artifacts
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


def cmake_configure_flags(backend: str, build: Path, libdir: str | None = None) -> list[str]:
    flags = [
        "cmake", "-G", "Ninja", "-S", ".", "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=/usr",
    ]
    if backend == "vulkan":
        flags.append("-DBACKEND_VULKAN=ON")
    elif backend == "cuda":
        flags.append("-DBACKEND_CUDA=ON")
        # Packages target the sm_89 baseline; consumers with newer GPUs still
        # run it through PTX/driver compatibility.
        flags.append("-DCMAKE_CUDA_ARCHITECTURES=89")
    elif backend == "sycl":
        flags.append("-DBACKEND_SYCL=ON")
        # CMakeLists.txt needs the compiler before project(); icpx must be on
        # PATH, so this is the same file the build lane would have found.
        flags.append("-DCMAKE_CXX_COMPILER=icpx")
    if libdir:
        flags.append(f"-DCMAKE_INSTALL_LIBDIR={libdir}")
    return flags


# ---------------------------------------------------------------------------
# Arch / pacman (PKGBUILD + makepkg)
# ---------------------------------------------------------------------------

def pkgbuild_text(backend: str, version: str) -> str:
    depends, makedepends = ARCH_DEPS[backend]
    arch, _, _ = host_arch()
    dep_list = " ".join(f"'{d}'" for d in depends)
    make_list = " ".join(f"'{d}'" for d in makedepends)
    conflict_list = " ".join(f"'{n}'" for n in other_package_names(backend))
    backend_line = ""
    if backend == "vulkan":
        backend_line = "        -DBACKEND_VULKAN=ON \\\n"
    elif backend == "cuda":
        backend_line = ("        -DBACKEND_CUDA=ON \\\n"
                        "        -DCMAKE_CUDA_ARCHITECTURES=89 \\\n")
    elif backend == "sycl":
        backend_line = ("        -DBACKEND_SYCL=ON \\\n"
                        "        -DCMAKE_CXX_COMPILER=icpx \\\n")
    return f"""# Generated by tools/package.py --package arch. Do not edit by hand.
pkgname={package_name(backend)}
pkgver={version}
pkgrel=1
pkgdesc="{SUMMARY} ({backend} backend)"
arch=('{arch}')
url="{URL}"
license=('{LICENSE}')
provides=('{PROJECT}')
conflicts=({conflict_list})
depends=({dep_list})
makedepends=({make_list})
source=("{PROJECT}-$pkgver.tar.gz")
sha256sums=('SKIP')

build() {{
    cd "{PROJECT}-$pkgver"
    cmake -G Ninja -S . -B build \\
        -DCMAKE_BUILD_TYPE=Release \\
        -DCMAKE_INSTALL_PREFIX=/usr \\
{backend_line}        -DBUILD_TEST=OFF -DBUILD_BENCH=OFF -DBUILD_EXAMPLES=OFF
    cmake --build build --parallel
}}

package() {{
    cd "{PROJECT}-$pkgver"
    DESTDIR="$pkgdir" cmake --install build
}}
"""


def package_arch(backend: str) -> None:
    require("makepkg", "fakeroot", "tar", "cmake", "ninja")
    version = read_version()
    work = WORK_DIR / "arch"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    DIST_DIR.mkdir(exist_ok=True)

    tarball = work / f"{PROJECT}-{version}.tar.gz"
    print(f"📦 source tarball -> {tarball}")
    make_source_tarball(tarball, version)

    (work / "PKGBUILD").write_text(pkgbuild_text(backend, version))
    print(f"🔨 makepkg ({backend} backend)")
    run(["makepkg", "-f", "--noconfirm"], cwd=work)

    produced = sorted(work.glob("*.pkg.tar.*"))
    if not produced:
        raise SystemExit("❌ makepkg produced no package")
    for pkg in produced:
        target = DIST_DIR / pkg.name
        shutil.move(str(pkg), str(target))
        print(f"✅ {target}")


# ---------------------------------------------------------------------------
# deb / rpm: build once into a DESTDIR staging tree
# ---------------------------------------------------------------------------

def build_into_staging(backend: str, version: str, work: Path) -> Path:
    src = work / "src"
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)

    tarball = work / f"{PROJECT}-{version}.tar.gz"
    print(f"📦 source tarball -> {tarball}")
    make_source_tarball(tarball, version)
    run(["tar", "-xzf", str(tarball), "-C", str(src)])

    project = src / f"{PROJECT}-{version}"
    build = project / "build"
    run(cmake_configure_flags(backend, build))
    run(["cmake", "--build", str(build), "--parallel"])

    staging = work / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    print(f"📂 staging install -> {staging}")
    run(["cmake", "--install", str(build), "--prefix", "/usr"],
        env=dict(os.environ, DESTDIR=str(staging)))
    return staging


def package_deb(backend: str) -> None:
    require("dpkg-deb", "tar", "cmake", "ninja")
    version = read_version()
    _, deb_arch, _ = host_arch()
    work = WORK_DIR / "deb"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    DIST_DIR.mkdir(exist_ok=True)

    staging = build_into_staging(backend, version, work)
    depends, _ = DEB_DEPS[backend]
    name = package_name(backend)
    conflicts = ", ".join(other_package_names(backend))
    ctrl = staging / "DEBIAN"
    ctrl.mkdir(exist_ok=True)
    (ctrl / "control").write_text(
        f"Package: {name}\n"
        f"Version: {version}-1\n"
        f"Section: libs\n"
        f"Priority: optional\n"
        f"Architecture: {deb_arch}\n"
        f"Depends: {depends}\n"
        f"Provides: {PROJECT}\n"
        f"Conflicts: {conflicts}\n"
        f"Maintainer: Aimol-l <noreply@example.com>\n"
        f"Description: {SUMMARY}\n"
        f" ({backend} backend)\n")

    out = DIST_DIR / f"{name}_{version}-1_{deb_arch}.deb"
    run(["dpkg-deb", "--root-owner-group", "--build", str(staging), str(out)])
    print(f"✅ {out}")


def rpm_spec(backend: str, version: str) -> str:
    requires, build_requires = RPM_DEPS[backend]
    conflicts = ", ".join(other_package_names(backend))
    backend_line = ""
    if backend == "vulkan":
        backend_line = "    -DBACKEND_VULKAN=ON \\\n"
    elif backend == "cuda":
        backend_line = ("    -DBACKEND_CUDA=ON \\\n"
                        "    -DCMAKE_CUDA_ARCHITECTURES=89 \\\n")
    return f"""Name:           {package_name(backend)}
Version:        {version}
Release:        1%{{?dist}}
Summary:        {SUMMARY}
License:        {LICENSE}
URL:            {URL}
Source0:        {PROJECT}-%{{version}}.tar.gz
BuildRequires:  {build_requires}
Requires:       {requires}
Provides:       {PROJECT}
Conflicts:      {conflicts}

%description
{SUMMARY} ({backend} backend).

%prep
# The package name carries the backend, the source archive does not.
%setup -q -n {PROJECT}-%{{version}}

%build
cmake -G Ninja -S . -B build \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DCMAKE_INSTALL_PREFIX=/usr \\
    -DCMAKE_INSTALL_LIBDIR=%{{_lib}} \\
{backend_line}    -DBUILD_TEST=OFF -DBUILD_BENCH=OFF -DBUILD_EXAMPLES=OFF
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


def package_rpm(backend: str) -> None:
    require("rpmbuild", "tar", "cmake", "ninja")
    version = read_version()
    work = WORK_DIR / "rpm"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    DIST_DIR.mkdir(exist_ok=True)

    tarball = work / f"{PROJECT}-{version}.tar.gz"
    print(f"📦 source tarball -> {tarball}")
    make_source_tarball(tarball, version)

    specs = work / "SPECS"
    specs.mkdir()
    spec = specs / f"{PROJECT}.spec"
    spec.write_text(rpm_spec(backend, version))

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
        str(spec.resolve()),
    ])

    rpms = sorted((topdir / "RPMS").rglob("*.rpm"))
    if not rpms:
        raise SystemExit("❌ rpmbuild produced no .rpm")
    for pkg in rpms:
        target = DIST_DIR / pkg.name
        shutil.move(str(pkg), str(target))
        print(f"✅ {target}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Tensorvia distribution packages.")
    parser.add_argument("package", nargs="+", choices=["arch", "deb", "rpm", "all"],
                        help="target package format(s)")
    parser.add_argument("-b", "--backend",
                        choices=["cpu", "vulkan", "vk", "cuda", "sycl"],
                        default="cpu",
                        help="backend compiled into the package (default: cpu; "
                             "sycl is a placeholder until implemented)")
    args = parser.parse_args()
    backend = normalize_backend(args.backend)

    targets = ["arch", "deb", "rpm"] if "all" in args.package else args.package
    seen: set[str] = set()
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        print(f"\n=== {target} ({backend} -> {package_name(backend)}) ===")
        {"arch": package_arch,
         "deb": package_deb,
         "rpm": package_rpm}[target](backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
