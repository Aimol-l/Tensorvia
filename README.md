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
`ops` API；Vulkan、CUDA 与 SYCL 覆盖下面这组算子（少数算子仍为 CPU-only，设备上调用会显式报错，
完整覆盖矩阵见 [docs/coverage.md](docs/coverage.md)）：

- 创建：`Zeros`、`Ones`、`Fill`、`Random`；生成器 `Arange`、`Linspace`、`Eye`、`Randn` 的
  `TensorOptions` 重载在宿主生成后一次性搬到目标设备（同一 seed 各后端取值相同）
- 算术：`Add`、`Sub`、`Mul`、`Div`、`MatMul`（N-D batched + batch 广播）、`Bmm`、`Addmm`、`Linear`、
  `Conv2d`（NCHW 的 2-D 卷积，stride/padding + 可选 bias，接受 `[N,C,H,W]` 与 `[C,H,W]`）
- 一元：`Abs`、`Sin`、`Cos`、`Tan`、`Exp`、`Sqrt`、`Log`、`Pow`、`Clamp`
- 激活：`Relu`、`Silu`、`Tanh`、`Sigmoid`、`Softmax`、`LogSoftmax`
- 归一化/统计：`LayerNorm`、`RmsNorm`、`Var`、`Std`、`Norm`（设备端由上面的逐元素与归约算子组合实现，
  全程留在设备上，无隐式回拷）
- 归约：`Sum`、`Min`、`Max`、`Mean`、`Argmax`、`Argmin`（`Var/Std/Norm` 见上）
- 比较/逻辑：`Equal`、`NotEqual`、`Greater`、`Less`、`GreaterEqual`、`LessEqual`、`All`、`Any`、`Nonzero`、`CountNonzero`
- 条件/索引/扫描：`Where`（三路广播）、`IndexSelect`、`Gather`（embedding 取行与逐元素取址）、`Cumsum`
- 形状/搬移/类型：`Concat`、`Transpose`、`Slice`、`Split`、`Chunk`、`Stack`、`Pad`、`Repeat`、`Flip`、`Roll`、`Typecast`

以上分组是便于阅读的摘要。权威的覆盖矩阵在 [docs/coverage.md](docs/coverage.md)：算子 × 后端逐格标注
native / emulated（组合实现）/ unsupported（显式拒绝），由 `tools/coverage_matrix.py` 从源码与
`tools/coverage_overrides.toml` 生成，CI 守漂移——那份矩阵取代了这里的手写清单。

设备端未覆盖的算子（在设备张量上调用会带算子名抛错，不会静默回退 CPU）不在这里手写清单：
以 [docs/coverage.md](docs/coverage.md) 的矩阵为准，其中 `unsupported` 的每一格都注明拒绝位置
（`require_cpu` 或 `execute_on` 的“后端未实现”路径）。算子提交的同步语义（哪些调用会让宿主
等待设备）见 [docs/async-semantics.md](docs/async-semantics.md)。

`examples/inference.cpp` 按真实推理引擎的分段跑一次前向：text → tokens → embedding →
prefill（整段 prompt 一次算完，K/V 写进 cache）→ decode（一次一个 token，复用 cache）→ logits。
模型是随机权重的两层 decoder-only Transformer：词表与位置查表走 `IndexSelect`，多头注意力的
因果掩码走 `Where`。`inference [cpu|cuda|vulkan|sycl] [prompt]` 指定后端与提示词。

`examples/cnn_inference.cpp` 跑一个两层卷积网络：conv → relu → maxpool → conv → relu → maxpool
→ 全连接 → softmax。卷积直接调 `Conv2d`，权重随机；程序会在宿主上按卷积的定义重算几个输出位置
与算子结果比对。`cnn_inference [cpu|cuda|vulkan|sycl]` 指定后端。

## 📦 依赖

| 构建方式 | 依赖 |
| --- | --- |
| mcpp | 仅 [mcpp](https://github.com/mcpp-community/mcpp)（自带隔离工具链） |
| CMake | CMake ≥ 3.28、支持 C++23 的 GCC/Clang、OpenMP |
| 测试 | GoogleTest（`BUILD_TEST=ON` 或 `mcpp test`） |
| Vulkan | Vulkan 1.4 头文件/loader；shader 编译器由 mcpp 按需安装（`xim:shaderc`，即 `glslc`）。CMake 构建需自行提供 Python 3 + 支持 `GL_EXT_bfloat16` 的 `glslangValidator`（≥ 16.5） |
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

    std::println("{}", ops::Sum(ops::Softmax(v, 0)).item<float>());  // 归约结果取回标量：item<T>()

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

    std::println("{}", ops::Sum(ops::Softmax(v, 0)).item<float>());  // 归约结果取回标量：item<T>()

    return 0;
}
```

### 1.3 从源码构建库本身

```sh
cd Tensorvia

mcpp build --dev                                                  # CPU 后端（默认）
mcpp build --dev --features cuda   --accel "cuda13.3+{sm_89}"     # CPU + CUDA
mcpp build --dev --features sycl   --accel "sycl"                 # CPU + SYCL
mcpp build --dev --features vulkan --accel vulkan1.4               # CPU + Vulkan
```

**`--features` 和 `--accel` 是两个轴，三个设备后端都必须配对给**。Vulkan 的
shader 由 `mcpp.rules.spirv` 编译（规则自带 `xim:shaderc`/`glslc`，构建时不需要
Python，也不需要宿主机上装 shader 编译器）；规则从 `[build] sources` 里
`accel = "vulkan1.4"` 的 glob 取 shader 集合，所以这个轴现在也是 Vulkan 必需的：

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

Vulkan 后端需要 `glslangValidator`（≥ 16.5，`GL_EXT_bfloat16`）与 Vulkan 头文件；
这条路径用 `tools/gen_spv.py` 生成 SPIR-V 登记表（CMake 没有规则引擎），mcpp
那条路径则由 `mcpp.rules.spirv` 编译 shader，两者共用同一个运行时接口；SYCL 需要
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

    std::println("{}", ops::Sum(ops::Softmax(v, 0)).item<float>());  // 归约结果取回标量：item<T>()

    return 0;
}
```

## 🎛 多设备（多卡）

`Device` 只选后端种类；同一个后端下的第几张卡由运行期序号决定，两者合成
`DeviceId { Device kind; int index; }`。CPU 恒为 `{CPU, 0}`。

```cpp
// 每张卡一个实例：设备数、当前设备、切换（线程私有，torch 风格）
const int n = ops::device_count(Device::CUDA);
ops::set_device(Device::CUDA, 1);                 // 越界/负数直接抛
{
    ops::DeviceGuard guard(Device::CUDA, 0);      // 作用域结束恢复上一个
    Tensor a = Tensor::Ones({4, 4}, DataType::FLOAT32, Device::CUDA); // 落在 0 号卡
}                                                  // 回到 1 号卡

Tensor t(values, {3});                             // 先按 {CPU, 0} 构造
t.to_device(DeviceId{Device::CUDA, 1});            // 跨设备拷贝经 host 中转
std::println("{}", device_id_to_string(t.device_id()));   // CUDA:1
```

几条约定：

- 算子的输出跟随输入所在实例；同一后端的不同序号也算跨设备，作为操作数混用会被拒绝，
  不会拿 A 卡的 context 去读 B 卡的指针。
- `to_device(Device)` 的目标序号是当前线程的当前设备；`to_device(DeviceId)` 精确指定。
- `ops::synchronize()` / `ops::empty_cache()` 覆盖所有已经起起来的实例。
- 序号空间由各后端自己定义：CUDA 按 `cudaGetDeviceCount` 顺序（默认是显存最大的那张），
  Vulkan 按"过滤 + 类型优先序"后的列表，SYCL 按 `sycl::device::get_devices()` 的位置，
  默认值都是该后端原本会选中的设备——不调用 `set_device` 的程序行为不变。

完整示例见 `examples/multi_device.cpp`：在只有一张卡的机器上它照常跑完，跨卡那一段
打印一条提示后跳过。

> **迁移提示**：旧的 `to_device(uint32_t id = 0)` 已删除。它把传进来的参数整个忽略，只按
> 编译期开了哪些后端挑目标（同时编入多个后端时静默选 Vulkan），在多设备下没法表达
> "第几张卡"。改用 `to_device(Device)`（目标后端当前线程的当前卡）或
> `to_device(DeviceId{kind, index})` / `to_device(kind, index)`（精确指定）。

## 🧪 测试

```sh
# mcpp：编译并运行 tests/**/*.cpp（dev-dependencies 里的 gtest 只在此时进图）
mcpp test                                          # CPU（默认）
#mcpp test --features vulkan                        # + Vulkan lane 的完整构建
#mcpp test --features sycl --accel sycl             # + SYCL lane 的完整构建
#mcpp test --features cuda --accel "cuda13.3+{sm_89}"  # + CUDA lane 的完整构建

# CMake
cmake -B build -DBACKEND_CPU=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON && cmake --build build --parallel
# cmake -B build -DBACKEND_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON && cmake --build build --parallel
# cmake -B build -DBACKEND_SYCL=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON && cmake --build build --parallel
```

`tests/` 是头文件范式（`#include "tensorvia.hpp"`），mcpp 与 CMake 两条路径编译同一批文件。每个 `tests/*.cpp` 各自是一个独立的测试二进制：`device`（后端生命周期、多设备寻址、CUDA context/stream）、`view`（视图/offset/引用语义）、`ops`（算子与数值覆盖）、`contract`（FR-1/FR-2/FR-3 契约）、`p0`（P0 语义契约 FR-3..FR-7）、`p1`（FR-9..FR-12：N-D matmul、推理关键算子的跨后端一致性、搬运算子、不支持组合的显式报错）、`dim`；共享夹具与助手在 `tests/test_support.hpp`。另有 `benchmark/bench.cpp`（基准，自带 main；`mcpp.toml` 的 `[test] discover` 与 `-DBUILD_BENCH=ON` 都会构建它）。上面的 lane 命令验证的是**该 lane 完整构建 + 公共测试通过**；各设备后端编入后的算子行为由消费方冒烟覆盖。测试通过 `[dev-dependencies]` 拿 GoogleTest——它只进测试二进制，不进 `libtensorvia.so`。模块导出面（`import tensorvia;` 的符号清单）由 `tools/gen_facade.py --check` 在每次 `mcpp build` 时守卫。

> **SYCL lane 的 llvm 构建需要 mcpp ≥ 2026.10.26.1**：该版本起 mcpp 把
> libsycl 带来的 libstdc++ 识别为 C++ ABI island 并做链接后验证，`.so` 内
> 异常类型恢复匹配（此前 llvm + SYCL lane 有 1 个 `View.
> InPlaceRejectsNonContiguous` 异常类型失败，需 `--toolchain gcc@16.1.0`
> 绕过；现已实测 23/23 全过）。

## 📄 许可证

MIT，见 [LICENSE](LICENSE)。
