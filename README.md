# Tensorvia: Cross-Platform Multi-Backend Tensor Acceleration Library

[![CMake](https://img.shields.io/badge/CMake-3.25+-brightgreen)](https://cmake.org/)
[![C++20](https://img.shields.io/badge/C++-20/23-blue)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)](https://github.com/your-username/Tensorvia)
[![Backends](https://img.shields.io/badge/Backends-CPU%20%7C%20CUDA%20%7C%20SYCL%20%7C%20Vulkan-brightgreen)](https://github.com/your-username/Tensorvia)

**Tensorvia** is a high-performance tensor computation library supporting multiple hardware backends with unified API interfaces. Designed for scientific computing, computer vision, and deep learning applications requiring cross-platform acceleration.

## 🌟 Features

- **Multi-Backend Support**: Seamlessly switch between CPU, CUDA, SYCL, and Vulkan backends
- **Modern C++20**: Leverages latest C++ standards for optimal performance and safety
- **Hardware Acceleration**: Utilizes OpenMP, Intel oneAPI, CUDA, and Vulkan for maximum performance
- **Cross-Platform**: Compatible with Linux and Windows systems
- **Unified API**: Consistent interface across all supported backends
- **Memory Management**: Automatic memory handling across different devices

## 🛠 Supported Backends

| Backend | Compiler |C++ Standard| Acceleration | Target Devices | Status |
|---------|----------|--|-------------|----------------|---------|
| **CPU** | GCC15 |C++20/23 |OpenMP | Multi-core CPUs | 🔧 Doing |
| **CUDA** | NVCC | C++20|CUDA Toolkit | NVIDIA GPUs | 🔧 Doing |
| **SYCL** | ICPX2025 |C++23| DPC++ | Intel/AMD/NVIDIA GPUs | 🔧 Doing |
| **Vulkan** | Clang++20 |C++23 |Vulkan API | AMD/Intel/NVIDIA GPUs | ❌ Todo |

## 📦 Installation

### Arch Linux

- CPU Backend:  `pacman -S onetbb gcc cmake openmp`
- CUDA Backend: `pacman -S nvidia opencl-nvidia cuda cmake openmp nccl`
- SYCL Backend: `pacman -S intel-oneapi-basekit cmake`
- Vulkan Backend: `pacman -S vulkan-tools cmake openmp clang`

### Build from Source

```bash
git clone --recursive https://github.com/Aimol-l/Tensorvia.git

cd Tensorvia

cmake -B build -DBACKEND_CPU=ON -DBUILD_TEST=ON && cmake --build build --parallel $(nproc) # cpu
cmake -B build -DBACKEND_CUDA=ON -DBUILD_TEST=ON && cmake --build build --parallel $(nproc) # cuda
cmake -B build -DBACKEND_SYCL=ON -DBUILD_TEST=ON && cmake --build build --parallel $(nproc) # sycl
cmake -B build -DBACKEND_VULKAN=ON -DBUILD_TEST=ON && cmake --build build --parallel $(nproc) # vulkan
```

### Build Options

| CMake Option | Description | Default |
|--------------|-------------|---------|
| `BACKEND_CPU` | Enable CPU backend with OpenMP | OFF |
| `BACKEND_CUDA` | Enable CUDA backend | OFF |
| `BACKEND_SYCL` | Enable SYCL backend | OFF |
| `BACKEND_VULKAN` | Enable Vulkan backend | OFF |
| `BUILD_TEST` | Build test executables | OFF |
| `CMAKE_BUILD_TYPE` | Build type (Release/Debug) | Release |

## 🚀 Quick Start

### Basic Tensor Operations

```c++
#include "ops.h"
int main() {
    Tensor a = Tensor::Random({5,5},-10,10,DataType::INT8);
    Tensor b = Tensor::Random({5,5},-10,10,DataType::INT16);
    Tensor c = Tensor::Random({5,5},-10,10,DataType::INT32);
    Tensor d = Tensor::Random({5,5},-10,10,DataType::INT64);
    Tensor e = Tensor::Random({5,5},-10,10,DataType::FLOAT16);
    Tensor f = Tensor::Random({5,5},-10,10,DataType::BFLOAT16);
    Tensor g = Tensor::Random({5,5},-10,10,DataType::FLOAT32);
    Tensor h = Tensor::Random({5,5},-10,10,DataType::FLOAT64);

    ops::println(a + b);
    ops::println(c - d);
    ops::println(e * f);
    ops::println(g / h);
    return 0;
}
```

## 📊 Performance Benchmarks

### Matrix Multiplication (10x1024x1024 @ 10x1024x1024,fp32)

| Backend | Time (ms) | Speedup |
|---------|-----------|---------|
| CPU | 370 ms | 1x |
| CUDA| 8 ms | 46x |
| SYCL | 10 ms | 37x |
| VULKAN | 9999 ms | ?x |

### todo...

## 📚 Documentation

+ [Todo](https://markdown.com.cn)
+ [Todo](https://markdown.com.cn)
+ [Todo](https://markdown.com.cn)

## 🐛 Troubleshooting

### Todo

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.


## 🏆 Acknowledgments

+ Intel for oneAPI and SYCL support
+ NVIDIA for CUDA ecosystem
+ Khronos Group for Vulkan API
+ TBB for thread building blocks


## 📞 Support
 + 💬 Discussions: GitHub Discussions

----
### <center> Tensorvia - Bridging the gap between hardware and performance 🚀