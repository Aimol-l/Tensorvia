#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from pathlib import Path

# 配置
GLSLANG = "glslangValidator"
SHADER_ROOT = Path("./shader")
SPV_ROOT = Path("./spv")
BUILD_DIR = Path("build")

SUPPORTED_BACKENDS = ["CPU", "VULKAN", "CUDA", "SYCL"]

def compile_spv(op_name: str | None = None):
    print("🔍 Compiling SPIR-V shaders...")
    SPV_ROOT.mkdir(parents=True, exist_ok=True)

    if op_name:
        # 只编译 shader/{op_name}/*.comp
        op_dir = SHADER_ROOT / op_name
        if not op_dir.exists():
            print(f"⚠️  Operator directory not found: {op_dir}")
            return
        shader_files = list(op_dir.glob("*.comp"))
        if not shader_files:
            print(f"⚠️  No .comp files found in {op_dir}")
            return
    else:
        # 编译所有 shader/*/*.comp
        shader_files = list(SHADER_ROOT.glob("*/*.comp"))

    if not shader_files:
        print("⚠️  No .comp shaders found.")
        return

    success_count = 0
    for shader in shader_files:
        out_path = SPV_ROOT / (shader.stem + ".spv")
        cmd = [
            GLSLANG,
            "-V",
            "--target-env", "vulkan1.4",
            str(shader),
            "-o", 
            str(out_path)
        ]
        try:
            subprocess.run(cmd, check=True)
            success_count += 1
            print(f"✅ Compiled {shader}")
        except FileNotFoundError:
            print(f"❌ glslangValidator not found. Please install it or add to PATH.")
            sys.exit(1)
        except subprocess.CalledProcessError:
            print(f"❌ Compilation failed: {shader}")
            continue

    total = len(shader_files)
    print(f"✅ Successfully compiled {success_count}/{total} shaders{' for operator ' + op_name if op_name else ''}.")

def compile_library(backend: str, build_test: bool = True):
    print(f"⚙️  Configuring CMake for backend: {backend}...")
    if backend not in SUPPORTED_BACKENDS:
        print(f"❌ Unsupported backend: {backend}. Supported: {', '.join(SUPPORTED_BACKENDS)}")
        sys.exit(1)

    BUILD_DIR.mkdir(exist_ok=True)

    cmd = [
        "cmake",
        "-B", str(BUILD_DIR),
        "-S", ".",
        f"-DBACKEND_{backend}=ON",
    ]
    if build_test:
        cmd.append("-DBUILD_TEST=ON")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("❌ CMake configure failed.")
        sys.exit(1)

    print("🔨 Building the Tensorvia library...")
    build_cmd = ["cmake", "--build", str(BUILD_DIR), "-j6"]
    try:
        subprocess.run(build_cmd, check=True)
    except subprocess.CalledProcessError:
        print("❌ Build failed.")
        sys.exit(1)

    print("✅ Tensorvia library built successfully.")

def main():
    parser = argparse.ArgumentParser(
        prog="build.py",
        description="Build tensor library and/or compile SPIR-V shaders."
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=SUPPORTED_BACKENDS,
        default="VULKAN",
        help="Target backend (default: VULKAN)"
    )
    parser.add_argument(
        "--lib",
        action="store_true",
        help="Skip building the tensor library"
    )
    parser.add_argument(
        "--spirv",
        action="store_true",
        help="Skip compiling SPIR-V shaders"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Disable BUILD_TEST in CMake"
    )
    parser.add_argument(
        "--op",
        type=str,
        metavar="OPERATOR",
        help="Only compile SPIR-V shaders for a specific operator (e.g. --op div)"
    )


    #=============================================================
    args = parser.parse_args()

    # 1. 编译张量库
    if args.lib:
        compile_library(backend=args.backend, build_test=args.test)
    else:
        print("⏭️  Skipping Tensorvia library build.")

    # 2. 编译 SPIR-V（可选：仅特定算子）
    if args.spirv:
        should_compile_spirv = (
            args.backend == "VULKAN" or
            os.getenv("FORCE_SPIRV", "0") == "1" or
            args.op is not None  # 显式指定 --op 时，即使非 Vulkan 也编译
        )
        if should_compile_spirv:
            compile_spv(op_name=args.op)
        else:
            print("⏭️  Skipping SPIR-V compilation (not needed for non-Vulkan backend).")
    else:
        print("⏭️  Skipping SPIR-V compilation (--no-spirv).")

if __name__ == "__main__":
    main()