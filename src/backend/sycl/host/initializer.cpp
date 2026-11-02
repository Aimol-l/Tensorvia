// Device::SYCL 的填充 / 随机初始化。
//
// Random 在设备单元中使用基于计数器的哈希（以元素下标作键的 splitmix64），因此
// 无需分配或同步任何生成器状态。种子始终来自宿主。
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

#include "sycl_island.h"

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"
#include "tensorvia/backend/sycl/sycl_ops.hpp"
#include "tensorvia/backend/sycl/sycl_context.hpp"
#include "tensorvia/backend/sycl/sycl_tensor.hpp"
#include "tensorvia/backend/sycl/sycl_ufunc.hpp"

namespace ops {

namespace {

std::shared_ptr<SYCLContext> context_of(const Tensor& t, const char* what) {
    const auto impl = std::dynamic_pointer_cast<SYCLTensor>(t.get_impl());
    if (!impl) throw std::runtime_error(std::string("sycl ") + what + ": expected a SYCLTensor");
    auto ctx = std::dynamic_pointer_cast<SYCLContext>(impl->context());
    if (!ctx) throw std::runtime_error(std::string("sycl ") + what + ": missing SYCLContext");
    return ctx;
}

void fill_tensor(const Tensor& t, double value, const char* what) {
    auto ctx = context_of(t, what);
    ctx->activate();
    // const_cast：填充是唯一的宿主侧写入；这里用 Tensor::data()（感知 offset）
    // 而不是 TensorImpl::data()。
    sycl_ufunc::check(::sycl_c_launch_fill(
                          sycl_ufunc::handle_of(*ctx),
                          const_cast<void*>(t.data()),
                          static_cast<int64_t>(t.numel()),
                          static_cast<int>(t.dtype()), value),
                      what);
}

uint64_t make_seed() {
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
}

}  // namespace

Tensor ZerosImpl<Device::SYCL>::execute(const std::vector<int64_t>& shape, DataType dtype) {
    Tensor temp(shape, dtype, Device::SYCL);
    fill_tensor(temp, 0.0, "Zeros");
    return temp;
}

Tensor OnesImpl<Device::SYCL>::execute(const std::vector<int64_t>& shape, DataType dtype) {
    Tensor temp(shape, dtype, Device::SYCL);
    fill_tensor(temp, 1.0, "Ones");
    return temp;
}

Tensor FillImpl<Device::SYCL>::execute(const std::vector<int64_t>& shape, DataType dtype,
                                       float value) {
    Tensor temp(shape, dtype, Device::SYCL);
    fill_tensor(temp, static_cast<double>(value), "Fill");
    return temp;
}

void FillImpl<Device::SYCL>::execute(Tensor& a, float value) {
    fill_tensor(a, static_cast<double>(value), "Fill");
}

Tensor RandomImpl<Device::SYCL>::execute(const std::vector<int64_t>& shape, DataType dtype,
                                         float min, float max) {
    Tensor temp(shape, dtype, Device::SYCL);
    auto ctx = context_of(temp, "Random");
    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_random(
                          sycl_ufunc::handle_of(*ctx), temp.data(),
                          static_cast<int64_t>(temp.numel()),
                          static_cast<int>(dtype), min, max, make_seed()),
                      "Random");
    return temp;
}

}  // namespace ops
