# Tensorvia: Cross-Platform Multi-Backend Tensor Acceleration Library

[![CMake](https://img.shields.io/badge/CMake-3.25+-brightgreen)](https://cmake.org/)
[![C++23](https://img.shields.io/badge/C++-23-blue)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)](https://github.com/Aimol-l/Tensorvia)
[![Backends](https://img.shields.io/badge/Backends-CPU%20%7C%20CUDA%20%7C%20SYCL%20%7C%20Vulkan-brightgreen)](https://github.com/aimol-l/Tensorvia)

**Tensorvia** is a high-performance tensor computation library supporting multiple hardware backends with unified API interfaces. Designed for scientific computing, computer vision, and deep learning applications requiring cross-platform acceleration.

## 🌟 Features

- **Multi-Backend Support**: Seamlessly switch between CPU, CUDA, SYCL, and Vulkan backends
- **Modern C++23**: Leverages latest C++ standards for optimal performance and safety
- **Hardware Acceleration**: Utilizes OpenMP, Intel oneAPI, CUDA, and Vulkan for maximum performance
- **Cross-Platform**: Compatible with Linux and Windows systems
- **Unified API**: Consistent interface across all supported backends
- **Memory Management**: Automatic memory handling across different devices
- **Type Safety**: Support for multiple data types (INT8,INT16,INT32,INT64, FLOAT16,FLOAT32,FLOAT64, BFLOAT16)

## 🛠 Supported Backends

| Backend | Compiler | C++ Standard | Acceleration | Target Devices | Status |
|---------|----------|--------------|--------------|----------------|---------|
| **CPU** | GCC/Clang | C++23 | OpenMP/SIMD | Multi-core CPUs | ✅ Working |
| **CUDA** | NVCC | C++23 | CUDA Toolkit | NVIDIA GPUs | ✅ Working |
| **SYCL** | ICPX | C++23 | DPC++ | Intel/NVIDIA GPUs | ⚠️ Experimental |
| **Vulkan** | GCC/Clang | C++23 | Vulkan API | GPUs | ⚠️ Experimental |

## 📦 Installation

### Arch Linux

- CPU Backend: `pacman -S gcc cmake openmp`
- CUDA Backend: `pacman -S nvidia opencl-nvidia cuda cmake openmp`
- SYCL Backend: `pacman -S intel-oneapi-basekit cmake`
- Vulkan Backend: `pacman -S vulkan-tools cmake openmp clang`

### Ubuntu/Debian

- CPU Backend: `apt install build-essential cmake libtbb-dev`
- CUDA Backend: `apt install nvidia-cuda-toolkit cmake`
- SYCL Backend: `apt install intel-oneapi-basekit cmake`
- Vulkan Backend: `apt install libvulkan-dev vulkan-tools cmake`

### Build from Source

```bash
git clone --recursive https://github.com/Aimol-l/Tensorvia.git
cd Tensorvia

# Build with CPU backend
cmake -B build -DBACKEND_CPU=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel

# Build with CUDA backend
cmake -B build -DBACKEND_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel

# Build with Vulkan backends
cmake -B build -DBACKEND_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel

# Build with SYCL backends
cmake -B build -DBACKEND_SYCL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
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

### CMAKE
```sh
find_package(Tensorvia REQUIRED)

add_executable(main main.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME} Tensorvia::tensorvia)
```

### Basic Tensor Operations



```cpp
#include <tensorvia/tensor.h>
#include <tensorvia/ops.h>

int main() {
    // Create tensors with different data types
    Tensor a = Tensor::Random({5, 5}, -10, 10, DataType::INT8);
    Tensor b = Tensor::Random({5, 5}, -10, 10, DataType::INT16);
    Tensor c = Tensor::Random({5, 5}, -10, 10, DataType::INT32);
    Tensor d = Tensor::Random({5, 5}, -10, 10, DataType::INT64);
    Tensor e = Tensor::Random({5, 5}, -10, 10, DataType::FLOAT16);
    Tensor f = Tensor::Random({5, 5}, -10, 10, DataType::BFLOAT16);
    Tensor g = Tensor::Random({5, 5}, -10, 10, DataType::FLOAT32);
    Tensor h = Tensor::Random({5, 5}, -10, 10, DataType::FLOAT64);

    // Perform basic operations
    ops::println(a + b);
    ops::println(c - d);
    ops::println(e * f);
    ops::println(g / h);
    
    // Matrix multiplication
    Tensor mat_a = Tensor::Random({100, 100}, -1.0, 1.0, DataType::FLOAT32);
    Tensor mat_b = Tensor::Random({100, 100}, -1.0, 1.0, DataType::FLOAT32);
    Tensor result = ops::matmul(mat_a, mat_b);
    
    return 0;
}
```




### Advanced Operations

```cpp
#include <tensorvia/tensor.h>
#include <tensorvia/ops.h>

int main() {
    // Create tensors
    Tensor tensor = Tensor::Random({10, 20, 30}, -5.0, 5.0, DataType::FLOAT32);
    
    // Tensor operations
    Tensor reshaped = tensor.view({20, 15, 20});  // Reshape
    Tensor transposed = ops::transpose(tensor, {2, 0, 1});  // Transpose
    Tensor sliced = tensor.slice({0, 5, 10}, {5, 15, 20});  // Slice
    
    // Reduction operations
    Tensor sum = ops::sum(tensor, {0});  // Sum along axis 0
    Tensor max_vals = ops::max(tensor, {1});  // Max along axis 1
    Tensor argmax = ops::argmax(tensor, {2});  // Argmax along axis 2
    
    // Activation functions
    Tensor relu_result = ops::relu(tensor);
    Tensor sigmoid_result = ops::sigmoid(tensor);
    Tensor tanh_result = ops::tanh(tensor);
    
    return 0;
}
```

## 📊 Performance Benchmarks

### Matrix Multiplication (2592x2048 @ 2048x4096, fp32)

| Backend | Avg Time | Speedup |
|---------|-----------|---------|
| CPU (OpenMP) | 716 ms | 1x |
| CUDA | 19 ms | 37.6x |
| SYCL | 20 ms | 35.8x |
| VULKAN | 27 ms | 26.5x |

> Note: Vulkan backend performance needs optimization and is currently experimental.

## 🧪 Testing

Run the test suite to verify your build:

```bash
# Build tests
cmake -B build -DBACKEND_CPU=ON -DBUILD_TEST=ON && cmake --build build

# or 
python build.py -b cpu -test on

# Run all tests
cd build && make test

# Run specific test
./tests/activate_test
./tests/math_test
./tests/reduce_test
```

## 📚 Documentation

The following sections provide comprehensive documentation for different aspects of Tensorvia:

- [API Reference](docs/api.md) - Detailed API documentation
- [Backend Guide](docs/backends.md) - How to use different backends
- [Performance Tips](docs/performance.md) - Optimization strategies
- [Building from Source](docs/building.md) - Detailed build instructions
- [Examples](examples/) - Complete example projects
- [Contributing](docs/contributing.md) - How to contribute to the project

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Backend Not Found**
   - Ensure CUDA toolkit is installed and `nvcc` is in your PATH
   - Check that the CUDA driver is compatible with your toolkit version

2. **Vulkan Validation Errors**
   - Install Vulkan SDK and validation layers
   - Set `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` for debugging

3. **Memory Issues**
   - Ensure sufficient GPU memory for large tensors
   - Monitor memory usage with `nvidia-smi` for CUDA

4. **Linking Errors**
   - Verify all required libraries are installed
   - Check that CMake can find the required dependencies


## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **Bug Reports**: File detailed bug reports with reproduction steps
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit pull requests for fixes or features
4. **Documentation**: Improve documentation and examples
5. **Testing**: Add test cases and verify different backends

See our [Contributing Guide](docs/contributing.md) for detailed instructions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- Intel for oneAPI and SYCL support
- NVIDIA for CUDA ecosystem
- Khronos Group for Vulkan API
- The C++ community for standards and libraries
- All open-source projects that make this possible

## 📞 Support

- 💬 [GitHub Discussions](https://github.com/Aimol-l/Tensorvia/discussions): General questions and community support
- 🐛 [Issues](https://github.com/Aimol-l/Tensorvia/issues): Bug reports and feature requests
- 📧 Contact: For direct inquiries, reach out to the maintainers

----
### <center> Tensorvia - Bridging the gap between hardware and performance 🚀 </center>