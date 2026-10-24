# Tensorvia: 跨平台多后端张量加速库

[![CMake](https://img.shields.io/badge/CMake-3.28+-brightgreen)](https://cmake.org/)
[![C++23](https://img.shields.io/badge/C++-23-blue)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey)](https://github.com/Aimol-l/Tensorvia)
[![Backends](https://img.shields.io/badge/Backends-CPU%20%7C%20Vulkan%20%7C%20CUDA%20%7C%20SYCL-brightgreen)](https://github.com/Aimol-l/Tensorvia)

**Tensorvia** 是一个高性能的 C++23 张量计算库，通过统一 API 使用 CPU、Vulkan、CUDA 或 SYCL 后端。库主体是
头文件源码（`include/tensorvia/`，单一事实源），同一条消费路径两种形态：

- **mcpp**：`src/interface/tensorvia.ixx` 薄包装成模块，`import tensorvia;`
- **CMake ≥ 3.28 / 安装包**：直接 `#include <tensorvia.hpp>`，不要求编译器支持模块

## 🌟 特性

- **多后端**：CPU、Vulkan、CUDA、SYCL 四种后端共用同一套 API；**一次构建选一个设备后端**
  （CPU 始终在内），运行时通过 `Device` 在已编入的后端间选择
- **统一 API**：`Tensor` 外观类 + `ops::` 命名空间 + 运算符重载，切换后端无需改代码
- **内存管理**：引用语义；`t()` / `slice()` / `permute()` / `view()` 零拷贝，深拷贝用 `clone()`；
  `to_device()` / `to_host()` 负责跨设备往返
- **类型安全**：支持 `INT8` / `INT16` / `INT32` / `INT64` / `FLOAT16` / `FLOAT32` / `FLOAT64` / `BFLOAT16`
- **头文件源码 + mcpp 薄模块包装**：库主体是头文件（`src/interface/*.hpp`，单一事实源）；
  mcpp 路径由 `tensorvia.ixx` 薄包装成模块（`import tensorvia;`），CMake 与安装包直接
  `#include <tensorvia.hpp>`。两条路径对使用方是同一套 API
- **不做静默回退**：设备后端未实现的算子在设备张量上直接抛异常，不会偷偷改用 CPU

## 🛠 支持的后端

| 后端 | 编译器 | 标准 | 加速 | 状态 |
| --- | --- | --- | --- | --- |
| **CPU** | Clang / GCC | C++23 | OpenMP、AVX2/FMA/F16C | ✅ 默认启用 |
| **Vulkan** | Clang / GCC | C++23 | Vulkan Compute（SPIR-V） | ✅ 可选启用 |
| **CUDA** | Clang 22（mcpp）/ NVCC（CMake） | C++20 设备单元 | CUDA Runtime（cuRAND） | ✅ 可选启用 |
| **SYCL** | Clang（host）+ DPC++（设备） | C++23 host + SYCL 设备单元 | SYCL（USM，自动选择设备） | ✅ 可选启用 |

设备枚举为 `Device::CPU`、`Device::VULKAN`、`Device::CUDA` 和 `Device::SYCL`。CPU 后端实现完整的
`ops` API；Vulkan、CUDA 与 SYCL 覆盖同一组算子：

- 创建：`Zeros`、`Ones`、`Fill`、`Random`
- 算术：`Add`、`Sub`、`Mul`（逐元素）、`Div`、`MatMul`（矩阵乘）
- 一元：`Abs`、`Sin`、`Cos`、`Tan`、`Exp`、`Sqrt`、`Log`、`Pow`、`Clamp`
- 激活：`Relu`、`Silu`、`Tanh`、`Sigmoid`、`Softmax`
- 归约：`Sum`、`Min`、`Max`、`Mean`、`Argmax`、`Argmin`
- 比较/逻辑：`Equal`、`NotEqual`、`Greater`、`Less`、`GreaterEqual`、`LessEqual`、`All`、`Any`、`Nonzero`
- 形状/类型：`Concat`、`Transpose`、`Slice`、`Typecast`

## 📦 依赖

| 构建方式 | 依赖 |
| --- | --- |
| mcpp | 仅 [mcpp](https://github.com/mcpp-community/mcpp)（自带隔离工具链） |
| CMake | CMake ≥ 3.28、支持 C++23 的 GCC/Clang、OpenMP |
| 测试 | GoogleTest（`BUILD_TEST=ON` 或 `mcpp test`） |
| Vulkan | Vulkan 1.4 头文件/loader、`glslangValidator`、Python 3（构建期编译 SPIR-V） |
| CUDA | CUDA Toolkit（mcpp 按 `--accel` 声明的线自动安装 payload；CMake 需自行提供 `nvcc`） |
| SYCL | DPC++（mcpp 自动安装 `dpcpp` payload；CMake 需 `icpx` 在 `PATH` 上）与 SYCL 运行时 |

## 一、当使用MCPP构建时

### 1.1 直接使用远端的包，避免打包的麻烦

#### 
```sh
# bash —— 使用远端包：后端在 [dependencies] 里用 features 选（见下方 mcpp.toml）
mcpp new hello && cd hello
mcpp build
mcpp run  # 运行main.cpp
```

```toml
# mcpp.toml
[package]
name        = "hello"
version     = "0.1.0"
description = "tensorvia package use project"
license     = "Apache-2.0"

[dependencies.aimol]
tensorvia = { version = "0.3.0"}  # 默认就是使用cpu后端
#tensorvia = { version = "0.3.0", features = ["cuda"] }
#tensorvia = { version = "0.3.0", features = ["sycl"] }
#tensorvia = { version = "0.3.0", features = ["vulkan"] }
# 消费者通过 features 选产物形态（不开的后端连编译都不参与）；
# CUDA/SYCL 的设备编译开关在 --accel（根级配置，只有使用方的构建命令能给）——
# 选了 cuda/sycl 就照下方 1.3 的表格配对 --accel，否则构建被 build.mcpp 拒绝。
```
```cpp
// main.cpp
import std;
import tensorvia;

int main() {
    auto a = Tensor::Ones({128, 256}, DataType::FLOAT32);
    auto b = Tensor::Fill({128, 256}, 2.0f, DataType::FLOAT32);

    auto c = a + b; 

    auto d = ops::MatMul(a, b.t());          // 矩阵乘

    ops::println(d);

    auto v = Tensor::Random({1024}, 1.0f, 10.0f, DataType::FLOAT32);

    std::println("{}", ops::Sum(ops::Softmax(v, 0)));

    return 0;
}
```

### 1.2 使用本地包

```sh
git clone --recursive https://github.com/Aimol-l/Tensorvia.git

cd Tensorvia

# 构建本地二进制分发包，mcpp暂不支持CUDA和SYCL后端的打包
mcpp pack tensorvia                              # CPU 后端
# mcpp pack tensorvia --features vulkan          # CPU + Vulkan 后端

# 打包到: ./target/dist/tensorvia-0.x.x-xxxx.tar.gz
```

> **二进制包的已知限制**：包内 `libtensorvia.so` 静态自含 libc++（mcpp 的
> portable 默认）。若使用方的程序 `catch` 库内抛出的 C++ 异常，两侧的
> libc++ 副本类型不一致时会 `bad_cast` 逃逸——二进制包建议仅用于不穿越
> `.so` 边界捕获异常的场景；异常级 API 的消费请用源码渠道（registry 源码
> 包 / git / path 依赖，使用方与库同工具链编译，无此问题）。

```sh
# bash
mcpp new hello && cd hello
mcpp build
mcpp run  # 运行main.cpp
```

```toml
# mcpp.toml
[package]
name        = "hello"
version     = "0.1.0"
description = "tensorvia package use project"
license     = "Apache-2.0"

[dependencies]
tensorvia = {path = "./3rd/tensorvia-0.x.x-xxxx"} # tar.gz解压的地址
```
```cpp
// main.cpp
import std;
import tensorvia;

int main() {
    auto a = Tensor::Ones({128, 256}, DataType::FLOAT32);
    auto b = Tensor::Fill({128, 256}, 2.0f, DataType::FLOAT32);

    auto c = a + b; 

    auto d = ops::MatMul(a, b.t());          // 矩阵乘

    ops::println(d);

    auto v = Tensor::Random({1024}, 1.0f, 10.0f, DataType::FLOAT32);

    std::println("{}", ops::Sum(ops::Softmax(v, 0)));

    return 0;
}
```

### 1.3 从源码构建库本身

```sh
cd Tensorvia

mcpp build --dev                                                  # CPU 后端（默认）
mcpp build --dev --features cuda   --accel "cuda13.3+{sm_89}"     # CPU + CUDA
mcpp build --dev --features sycl   --accel "sycl"                 # CPU + SYCL
mcpp build --dev --features vulkan                                 # CPU + Vulkan（无需 --accel）
```

**`--features` 和 `--accel` 是两个轴，CUDA/SYCL 必须配对给**（Vulkan 无 `--accel`
轴：它的 shader 由 `build.mcpp` 调 glslang 编译后嵌入）：

| 轴 | 控制什么 |
|---|---|
| `--features cuda` | 宿主代码里编不编 CUDA 后端（定义 `BACKEND_CUDA` 宏 → 注册与分发） |
| `--accel cuda...` | 编不编设备源码（`.cu`）、按哪个架构编、rules-cuda 走哪条兼容路径 |

只给 `--accel` 不给 `--features` 会被 `build.mcpp` 直接拒绝
（`--accel names cuda but the 'cuda' feature is off`）——设备代码编出来却没有宿主
注册，是个空壳。`--accel` 里的数字是 **toolkit 线声明**（`cuda13.3` 让 rules-cuda
按 13.3 线的兼容路径编译；写 12.9 或不写版本会撞当前 glibc 的 C23 `noexcept`
冲突），架构（`sm_XX`）按自己的 GPU 选；驱动版本由规则包探测比对
（`cuda.driver`），与 toolkit 线不是一回事。

## 二、当使用cmake构建时

### 2.1 从源码编译

| CMake选项 | 描述 | 默认值 |
|--------------|-------------|---------|
| `BACKEND_CPU` | 启用带OpenMP的CPU后端 | ON |
| `BACKEND_CUDA` | 启用CUDA后端 | OFF |
| `BACKEND_SYCL` | 启用SYCL后端 | OFF |
| `BACKEND_VULKAN` | 启用Vulkan后端 | OFF |
| `BUILD_TEST` | 构建测试可执行文件（`tests/`，facade 与 bench 除外） | OFF |
| `BUILD_BENCH` | 构建 `tests/bench.cpp` | OFF |
| `CMAKE_BUILD_TYPE` | 构建类型 (Release/Debug) | Release |

Vulkan 后端需要 `glslangValidator` 与 Vulkan 头文件（同 mcpp 路径）；SYCL 需要
`icpx` 可被 `find_program` 找到（或在 `PATH` 上）。`-march=x86-64-v3` 随库的
PUBLIC 编译选项传给使用方。

```bash
git clone --recursive https://github.com/Aimol-l/Tensorvia.git

cd Tensorvia

# 使用CPU后端构建
cmake -B build -DBACKEND_CPU=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel

# 使用CUDA后端构建
# cmake -B build -DBACKEND_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel

# 使用Vulkan后端构建（SPIR-V 由构建系统自动编译，见下）
# cmake -B build -DBACKEND_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel

# 使用SYCL后端构建（icpx 需在 PATH 上，或显式 -DTVV_SYCL_COMPILER=…）
# cmake -B build -DBACKEND_SYCL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
```

### 2.2 使用构建脚本

| 参数 | 作用 |
|------|------|
| `-b, --backend <cpu\|cuda\|sycl\|vulkan>` | 配置、编译并安装该后端到 `build-<backend>/`。 |
| `-test on\|off` | 是否同时构建测试（默认 `off`）。 |
| `-o, --out-dir DIR` | 生成产物的根目录（默认 `./build`，**不写源码树**）。 |
| `--glslang PATH` | `glslangValidator` 路径（默认从 `PATH` 查找）。 |
| `-pkg, --package arch\|deb\|rpm` | 生成不同发行版的系统包|
| `-c, --clean [build\|pkg\|dist\|all]` | 清理产物：`build`（默认，各 `build-*/` 与 `build/`）、`pkg`（`build-pkg/`）、`dist`（`dist/`）、`all`。配 `-b` 只清该后端。 |
```python
python3 build.py -pkg arch -b cpu -test on      # 编译生成CPU后端的张量库，同时自动打arch包
# python3 build.py -pkg arch -b sycl -test on   # 编译生成SYCL后端的张量库，同时自动打arch包
# python3 build.py -pkg deb -b cuda -test on    # 编译生成CUDA后端的张量库，同时自动打deb包
# python3 build.py -pkg rpm -b vulkan -test on  # 编译生成Vulkan后端的张量库，同时自动打rpm包
```

### 2.3 安装张量包

```bash
# 采用cmake的install
cmake --install build --prefix /your/prefix

# sudo pacman -U tensor-cpu-xxxxx
# sudo dpkg -i tensor-cpu-xxxx.deb
# sudo dnf - tensor-cpu-xxxx.rpm
```
### 2.4 使用

```cmake
# CMakelists.txt

find_package(Tensorvia REQUIRED)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE Tensorvia::tensorvia)
```

```cpp
// main.cpp
#include <tensorvia.hpp>

int main() {
    auto a = Tensor::Ones({128, 256}, DataType::FLOAT32);
    auto b = Tensor::Fill({128, 256}, 2.0f, DataType::FLOAT32);

    auto c = a + b; 

    auto d = ops::MatMul(a, b.t());          // 矩阵乘

    ops::println(d);

    auto v = Tensor::Random({1024}, 1.0f, 10.0f, DataType::FLOAT32);

    std::println("{}", ops::Sum(ops::Softmax(v, 0)));

    return 0;
}
```

## 🧪 测试

```sh
# mcpp：编译并运行 tests/**/*.cpp（dev-dependencies 里的 gtest 只在此时进图）
mcpp test

# CMake
cmake -B build -DBUILD_TEST=ON && cmake --build build && ctest --test-dir build
```

测试通过 `[dev-dependencies]` 拿 GoogleTest——它只进测试二进制，不进 `libtensorvia.so`。

## 📄 许可证

MIT，见 [LICENSE](LICENSE)。
