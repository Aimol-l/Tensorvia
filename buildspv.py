#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# glslangValidator path（如果不在 PATH 中，可写绝对路径）
GLSLANG = "glslangValidator"

SHADER_ROOT = Path("./shader")
SPV_ROOT = Path("./spv")

def compile_all():
    # 创建输出目录
    SPV_ROOT.mkdir(parents=True, exist_ok=True)

    # 递归查找 shader/*/*.comp
    shader_files = list(SHADER_ROOT.glob("*/*.comp"))

    if not shader_files:
        print("No .comp shader found in ./shader/*/")
        return

    for shader in shader_files:
        # 输出文件名保持名称，但扩展名改为 .spv
        out_name = shader.stem + ".spv"
        out_path = SPV_ROOT / out_name

        cmd = [
            GLSLANG,
            "-V",  # 编译为 SPIR-V
            "--target-env", "vulkan1.4",
            str(shader),
            "-o", str(out_path)
        ]

        # print(f"Compiling {shader} -> {out_path}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Compile failed: {shader}")
            print(e)
            continue

    print("✨ All shaders compiled.")

if __name__ == "__main__":
    compile_all()
