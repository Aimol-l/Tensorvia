#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <print>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "tensorvia.hpp"

namespace {

constexpr float kEps = 1e-5f;

constexpr std::array<Device, 4> kAllDevices{Device::CPU, Device::CUDA,
                                            Device::SYCL, Device::VULKAN};

// 本次编译启用的后端就是被测后端（Vulkan > CUDA > SYCL > CPU，与各 lane 的
// 构建命令一致）。设备相关的用例全部跑在它上面。
constexpr Device kTestDevice =
#if defined(BACKEND_VULKAN)
    Device::VULKAN;
#elif defined(BACKEND_CUDA)
    Device::CUDA;
#elif defined(BACKEND_SYCL)
    Device::SYCL;
#else
    Device::CPU;
#endif

// 重建多个视图测试共用的 [[1,2,3],[4,5,6],[7,8,9]] 夹具。
Tensor matrix_3x3() {
    static std::vector<float> values{1, 2, 3, 4, 5, 6, 7, 8, 9};
    return Tensor(values, {3, 3});
}

bool backend_available(Device device) {
    try {
        auto probe = Tensor::Zeros({1}, DataType::FLOAT32, device);
        return probe.numel() == 1;
    } catch (...) {
        return false;
    }
}

// 把张量搬回 host，按连续内存取出元素。非连续视图（切片/转置）先物化：
// data() 只保证从视图首元素起线性可读，跨步视图线性读会读到视图之外。
template <typename T>
std::vector<T> host_values(Tensor tensor) {
    tensor.to_host();
    tensor = tensor.contiguous();
    const T* begin = static_cast<const T*>(tensor.data());
    return {begin, begin + tensor.numel()};
}

// 拷贝一份放到目标设备；入参按值传递，调用方的 host 张量不受影响。
Tensor moved_to(Tensor tensor, Device device) {
    if (device != Device::CPU) tensor.to_device(device);
    return tensor;
}

template <typename T>
void expect_close(const std::vector<T>& got, const std::vector<T>& want,
                  std::string_view what) {
    ASSERT_EQ(got.size(), want.size()) << what;
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], kEps) << what << " at index " << i;
    }
}

// 设备用例的夹具：跑在本次编译启用的后端上。CPU 参考值由同一份代码在 host
// 算一遍再逐值对比；后端不可用时整组 SKIP。device.cpp 与 semantics.cpp 共用。
class DeviceOps : public ::testing::Test {
protected:
    void SetUp() override {
        if (!backend_available(kTestDevice)) {
            GTEST_SKIP() << device_to_string(kTestDevice) << " is unavailable in this build";
        }
    }
    static constexpr Device dev = kTestDevice;
};

}  // namespace
