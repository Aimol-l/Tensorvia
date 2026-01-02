# Tensorvia: 跨平台多后端张量加速库

[![CMake](https://img.shields.io/badge/CMake-3.25+-brightgreen)](https://cmake.org/)
[![C++23](https://img.shields.io/badge/C++-23-blue)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)](https://github.com/Aimol-l/Tensorvia)
[![Backends](https://img.shields.io/badge/Backends-CPU%20%7C%20CUDA%20%7C%20SYCL%20%7C%20Vulkan-brightgreen)](https://github.com/Aimol-l/Tensorvia)

**Tensorvia** 是一个高性能的张量计算库，支持使用统一API接口的多种硬件后端。专为需要跨平台加速的科学计算、计算机视觉和深度学习应用而设计。

## 🌟 特性

- **多后端支持**: 在CPU、CUDA、SYCL和Vulkan后端之间无缝切换
- **现代C++23**: 利用最新的C++标准实现最佳性能和安全性
- **硬件加速**: 利用OpenMP、Intel oneAPI、CUDA和Vulkan实现最大性能
- **跨平台**: 兼容Linux和Windows系统
- **统一API**: 所有支持的后端具有一致的接口
- **内存管理**: 跨不同设备的自动内存处理
- **类型安全**: 支持多种数据类型 (INT8,INT16,INT32,INT64, FLOAT16,FLOAT32,FLOAT64, BFLOAT16)

## 🛠 支持的后端

| 后端 | 编译器 | C++标准 | 加速 | 目标设备 | 状态 |
|---------|----------|--------------|--------------|----------------|---------|
| **CPU** | GCC/Clang | C++23 | OpenMP/SIMD | 多核CPU | ✅ 正常 |
| **CUDA** | NVCC | C++23 | CUDA Toolkit | NVIDIA GPU | ✅ 正常 |
| **SYCL** | ICPX | C++23 | DPC++ | Intel/NVIDIA GPU | ⚠️ 实验性 |
| **Vulkan** | GCC/Clang | C++23 | Vulkan API | GPU | ⚠️ 实验性 |

## 📦 安装

### Arch Linux

- CPU后端: `pacman -S gcc cmake openmp`
- CUDA后端: `pacman -S nvidia opencl-nvidia cuda cmake openmp`
- SYCL后端: `pacman -S intel-oneapi-basekit cmake`
- Vulkan后端: `pacman -S vulkan-tools cmake openmp clang`

### Ubuntu/Debian

- CPU后端: `apt install build-essential cmake libtbb-dev`
- CUDA后端: `apt install nvidia-cuda-toolkit cmake`
- SYCL后端: `apt install intel-oneapi-basekit cmake`
- Vulkan后端: `apt install libvulkan-dev vulkan-tools cmake`

### 从源码构建

```bash
git clone --recursive https://github.com/Aimol-l/Tensorvia.git
cd Tensorvia

# 使用CPU后端构建
cmake -B build -DBACKEND_CPU=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel

# 使用CUDA后端构建
cmake -B build -DBACKEND_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel

# 使用Vulkan后端构建
cmake -B build -DBACKEND_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel

# 使用SYCL后端构建
cmake -B build -DBACKEND_SYCL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
```

### 构建选项

| CMake选项 | 描述 | 默认值 |
|--------------|-------------|---------|
| `BACKEND_CPU` | 启用带OpenMP的CPU后端 | OFF |
| `BACKEND_CUDA` | 启用CUDA后端 | OFF |
| `BACKEND_SYCL` | 启用SYCL后端 | OFF |
| `BACKEND_VULKAN` | 启用Vulkan后端 | OFF |
| `BUILD_TEST` | 构建测试可执行文件 | OFF |
| `CMAKE_BUILD_TYPE` | 构建类型 (Release/Debug) | Release |

## 🚀 快速开始

### CMAKE
```sh
find_package(Tensorvia REQUIRED)

add_executable(main main.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME} Tensorvia::tensorvia)
```

### 基本张量操作

```cpp
#include <tensorvia/tensor.h>
#include <tensorvia/ops.h>

int main() {
    // 创建不同数据类型的张量
    Tensor a = Tensor::Random({5, 5}, -10, 10, DataType::INT8);
    Tensor b = Tensor::Random({5, 5}, -10, 10, DataType::INT16);
    Tensor c = Tensor::Random({5, 5}, -10, 10, DataType::INT32);
    Tensor d = Tensor::Random({5, 5}, -10, 10, DataType::INT64);
    Tensor e = Tensor::Random({5, 5}, -10, 10, DataType::FLOAT16);
    Tensor f = Tensor::Random({5, 5}, -10, 10, DataType::BFLOAT16);
    Tensor g = Tensor::Random({5, 5}, -10, 10, DataType::FLOAT32);
    Tensor h = Tensor::Random({5, 5}, -10, 10, DataType::FLOAT64);

    // 执行基本操作
    ops::println(a + b);
    ops::println(c - d);
    ops::println(e * f);
    ops::println(g / h);
    
    // 矩阵乘法
    Tensor mat_a = Tensor::Random({100, 100}, -1.0, 1.0, DataType::FLOAT32);
    Tensor mat_b = Tensor::Random({100, 100}, -1.0, 1.0, DataType::FLOAT32);
    Tensor result = ops::matmul(mat_a, mat_b);
    
    return 0;
}
```

### 高级操作

```cpp
#include <tensorvia/tensor.h>
#include <tensorvia/ops.h>

int main() {
    // 创建张量
    Tensor tensor = Tensor::Random({10, 20, 30}, -5.0, 5.0, DataType::FLOAT32);
    
    // 张量操作
    Tensor reshaped = tensor.view({20, 15, 20});  // 重塑
    Tensor transposed = ops::transpose(tensor, {2, 0, 1});  // 转置
    Tensor sliced = tensor.slice({0, 5, 10}, {5, 15, 20});  // 切片
    
    // 归约操作
    Tensor sum = ops::sum(tensor, {0});  // 沿轴0求和
    Tensor max_vals = ops::max(tensor, {1});  // 沿轴1求最大值
    Tensor argmax = ops::argmax(tensor, {2});  // 沿轴2求最大值索引
    
    // 激活函数
    Tensor relu_result = ops::relu(tensor);
    Tensor sigmoid_result = ops::sigmoid(tensor);
    Tensor tanh_result = ops::tanh(tensor);
    
    return 0;
}
```

## 📊 性能基准测试

### 矩阵乘法 (2592x2048 @ 2048x4096, fp32)

| 后端 | 平均时间 | 加速比 |
|---------|-----------|---------|
| CPU (OpenMP) | 716 ms | 1x |
| CUDA | 19 ms | 37.6x |
| SYCL | 20 ms | 35.8x |
| VULKAN | 27 ms | 26.5x |

> 注意: Vulkan后端性能需要优化，目前处于实验阶段。

## 🧪 测试

运行测试套件以验证构建:

```bash
# 构建测试
cmake -B build -DBACKEND_CPU=ON -DBUILD_TEST=ON && cmake --build build

# 或者
python build.py -b cpu -test on

# 运行所有测试
cd build && make test

# 运行特定测试
./tests/activate_test
./tests/math_test
./tests/reduce_test
```

## 📚 文档

以下部分提供Tensorvia不同方面的详细文档:

- [API参考](docs/api.md) - 详细的API文档
- [后端指南](docs/backends.md) - 如何使用不同的后端
- [性能提示](docs/performance.md) - 优化策略
- [从源码构建](docs/building.md) - 详细的构建说明
- [示例](examples/) - 完整的示例项目
- [贡献](docs/contributing.md) - 如何为项目做贡献

## 🐛 故障排除

### 常见问题

1. **找不到CUDA后端**
   - 确保已安装CUDA工具包且`nvcc`在PATH中
   - 检查CUDA驱动程序是否与工具包版本兼容

2. **Vulkan验证错误**
   - 安装Vulkan SDK和验证层
   - 设置`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`进行调试

3. **内存问题**
   - 确保GPU有足够内存用于大张量
   - 使用`nvidia-smi`监控CUDA内存使用情况

4. **链接错误**
   - 验证所有必需库都已安装
   - 检查CMake是否能找到所需的依赖项

## 🤝 贡献

我们欢迎社区的贡献！以下是您如何帮助的方法:

1. **错误报告**: 提交带有重现步骤的详细错误报告
2. **功能请求**: 建议新功能或改进
3. **代码贡献**: 为修复或功能提交Pull Request
4. **文档**: 改进文档和示例
5. **测试**: 添加测试用例并验证不同的后端

请参阅我们的[贡献指南](docs/contributing.md)获取详细说明。

## 📄 许可证

本项目根据MIT许可证授权 - 详情请见[LICENSE](LICENSE)文件。

## 🏆 致谢

- Intel提供oneAPI和SYCL支持
- NVIDIA提供CUDA生态系统
- Khronos Group提供Vulkan API
- C++社区提供标准和库
- 所有使这成为可能的开源项目

## 📞 支持

- 💬 [GitHub讨论](https://github.com/Aimol-l/Tensorvia/discussions): 一般问题和社区支持
- 🐛 [问题](https://github.com/Aimol-l/Tensorvia/issues): 错误报告和功能请求
- 📧 联系: 直接询问，请联系维护者

----
### <center> Tensorvia - 架起硬件与性能之间的桥梁 🚀 </center>