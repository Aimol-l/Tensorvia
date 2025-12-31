import os
import sys
import argparse
import subprocess
from pathlib import Path


BUILD_DIR = Path("build")
SPV_DIR = Path("./spv")
SHADER_ROOT = Path("./shader")
OUTPUT_CPP = Path("include/backend/vulkan/spirv/spv_registry.cpp")
OUTPUT_H = Path("include/backend/vulkan/spirv/spv_registry.h")

# 生成 SPIR-V 注册表的函数
def gen_spv_registry():
    spv_files = list(SPV_DIR.glob("*.spv"))
    registry = {}
    for spv in spv_files:
        # 文件名: "relu_float32.spv" → key = "relu_float32"
        key = spv.stem
        # 用 xxd 生成 C 数组
        result = subprocess.run(["xxd", "-i", str(spv)], capture_output=True, text=True)
        array_def = result.stdout.strip()
        registry[key] = array_def

    # 生成 .h
    with open(OUTPUT_H, "w") as f:
        f.write("#pragma once\n")
        f.write("#include <vector>\n")
        f.write("#include <string>\n")
        f.write("#include <cstdint>\n")
        f.write("#include <unordered_map>\n")
        f.write("namespace vkspv {\n")
        f.write("    std::vector<uint32_t> get_spv_code(const std::string& key);\n")
        f.write("}\n")

    # 生成 .cpp
    with open(OUTPUT_CPP, "w") as f:
        f.write("#include \"spv_registry.h\"\n")
        f.write("#include <stdexcept>\n")
        f.write("namespace vkspv {\n")
        
        # 声明所有数组
        for key in registry:
            f.write(f"static unsigned char {key}_data[] = {{\n")
            # xxd 输出已经是 {0x03, 0x02, ...};
            lines = registry[key].splitlines()
            for line in lines:
                if line.strip().startswith("unsigned char"):
                    continue
                f.write(f"{line}\n")
            # f.write("};\n\n")

        # 生成 map
        f.write("std::vector<uint32_t> get_spv_code(const std::string& key) {\n")
        f.write("    static const std::unordered_map<std::string, std::pair<unsigned char*, size_t>> registry = {\n")
        for key in registry:
            # 从 xxd 提取长度：查找 "xxx_len = N;"
            len_var = f"{key}_len"
            f.write(f"        {{\"{key}\", {{{key}_data, sizeof({key}_data)}}}},\n")
        f.write("    };\n")
        f.write("    auto it = registry.find(key);\n")
        f.write("    if (it == registry.end()) {\n")
        f.write("        throw std::runtime_error(\"SPIR-V not found: \" + key);\n")
        f.write("    }\n")
        f.write("    auto* data = it->second.first;\n")
        f.write("    auto size = it->second.second;\n")
        f.write("    return std::vector<uint32_t>(\n")
        f.write("        reinterpret_cast<uint32_t*>(data),\n")
        f.write("        reinterpret_cast<uint32_t*>(data + size)\n")
        f.write("    );\n")
        f.write("}\n")
        f.write("} // namespace vkspv\n")


# 编译 SPIR-V 着色器的函数
def compile_spv(op_name: str | None = None):
    print("🔍 Compiling SPIR-V shaders...")
    SPV_DIR.mkdir(parents=True, exist_ok=True)

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
        out_path = SPV_DIR / (shader.stem + ".spv")
        cmd = [
            "glslangValidator",
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
    BUILD_DIR.mkdir(exist_ok=True)

    cmd = [
        "cmake","-B", str(BUILD_DIR),"-S", ".",f"-DBACKEND_{backend}=ON","-DCMAKE_INSTALL_PREFIX=./build/install",f"-DBUILD_TEST={'ON' if build_test else 'OFF'}"
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("❌ CMake configuration failed.")
        sys.exit(1) 

    print("🔨 Building the Tensorvia library...")
    build_cmd = ["cmake", "--build", str(BUILD_DIR), "--parallel"]
    install_cmd = ["cmake" ,"--install","build"]
    try:
        subprocess.run(build_cmd, check=True)
        subprocess.run(install_cmd, check=True)
    except subprocess.CalledProcessError:
        print("❌ Build failed.")
        sys.exit(1)
    print("✅ Tensorvia library built successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-spv', '--spirv', default='all',required=False)
    parser.add_argument('-b', '--backend', choices=['cpu', 'cuda', 'sycl', 'vulkan'], required=True)
    parser.add_argument('-test', '--test', choices=['on', 'off'], default='off',required=True)
    args = parser.parse_args()


    # 生成 SPIR-V 注册表
    if args.backend == 'vulkan':
        # 先编译 SPIR-V 文件
        if args.spirv != 'none':
            compile_spv(None if args.spirv == 'all' else args.spirv)
        # 然后生成注册表
        gen_spv_registry()
    
    # 编译库
    compile_library(args.backend.upper(), build_test=(args.test == 'on'))
    