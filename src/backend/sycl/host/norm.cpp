#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

using ops::sycl_ufunc::handle_of;

// 权重/偏置与输入同设备是内核的前提：接缝只传裸指针，跨设备传进来不会有任何
// 检查，读到的就是另一块设备的内存。
void require_same_context(const Tensor& t, const Tensor& like, const char* what) {
    const auto impl = std::dynamic_pointer_cast<SYCLTensor>(t.get_impl());
    const auto ref  = std::dynamic_pointer_cast<SYCLTensor>(like.get_impl());
    if (!impl || !ref)
        throw std::runtime_error(std::string("sycl ") + what +
                                 ": operand does not live on the SYCL device");
    if (impl->context() != ref->context())
        throw std::runtime_error(std::string("sycl ") + what +
                                 ": operands live on different devices");
}

// LayerNorm 与 RmsNorm 只差一个仿射偏置和均值项；其余路径共用。
Tensor norm_impl(const Tensor& a, const Tensor* weight, const Tensor* bias, int axis,
                 float eps, bool rms, const char* what) {
    const int dim = static_cast<int>(a.shape().size());
    if (axis < 0) axis += dim;
    if (axis < 0 || axis >= dim)
        throw std::runtime_error(std::string("sycl ") + what + ": axis out of range");

    int64_t outer = 1;
    for (int i = 0; i < axis; ++i) outer *= a.shape(i);
    int64_t n = 1;
    for (int i = axis; i < dim; ++i) n *= a.shape(i);

    const DataType out_type =
        is_integral_dtype(a.dtype()) ? DataType::FLOAT32 : a.dtype();

    // 内核按 [outer, n] 的连续行主序解释输入；跨步视图先物化。
    const Tensor src = a.is_contiguous() ? a : a.clone();
    const Tensor w = weight->is_contiguous() ? *weight : weight->clone();
    const bool bias_owned = bias != nullptr && !bias->is_contiguous();
    const Tensor b_storage = bias_owned ? bias->clone() : Tensor();
    const Tensor* b = bias_owned ? &b_storage : bias;

    require_same_context(w, src, what);
    if (b != nullptr) require_same_context(*b, src, what);
    const auto a_shape = a.shape();
    Tensor out(std::vector<int64_t>(a_shape.begin(), a_shape.end()), out_type, Device::SYCL);

    auto ctx = sycl_ufunc::context_of(src, what);
    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_norm(handle_of(*ctx), src.data(), w.data(),
                                           b != nullptr ? b->data() : nullptr, out.data(),
                                           static_cast<int>(src.dtype()),
                                           static_cast<int>(out_type), outer, n, eps,
                                           rms ? 1 : 0),
                      what);
    return out;
}

}  // namespace

Tensor LayerNormImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& weight,
                                            const Tensor& bias, int axis, float eps) {
    return norm_impl(a, &weight, &bias, axis, eps, false, "LayerNorm");
}

Tensor RmsNormImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& weight,
                                          int axis, float eps) {
    return norm_impl(a, &weight, nullptr, axis, eps, true, "RmsNorm");
}

}  // namespace ops
