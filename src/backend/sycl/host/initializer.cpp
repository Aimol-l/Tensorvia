// Fill / random initialization for Device::SYCL.
//
// Random uses a counter-based hash in the device unit (splitmix64 keyed by the
// element index), so no generator state has to be allocated or synchronized.
// The seed always comes from the host.
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
#include "tensorvia/cpu_ops.hpp"
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
    // const_cast: filling is the one host-side write; Tensor::data() is the
    // offset-aware pointer, unlike TensorImpl::data().
    sycl_ufunc::check(::sycl_c_launch_fill(
                          sycl_ufunc::handle_of(*ctx),
                          const_cast<void*>(t.data()),
                          static_cast<int64_t>(t.numel()),
                          static_cast<int>(t.dtype()), value),
                      what);
    ctx->wait();
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
    ctx->wait();
    return temp;
}

}  // namespace ops
