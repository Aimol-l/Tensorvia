// General N-D permutation into a contiguous output, for Device::SYCL. A 2D
// transpose is the {1,0} case; the same kernel covers batches and higher ranks.
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

// Permute `a` (contiguous, possibly with a view offset) into `out`, which must
// already have the permuted shape and a contiguous buffer.
void launch_permute(const Tensor& a, Tensor& out, const std::vector<int64_t>& axes,
                    const char* what) {
    const int ndim = static_cast<int>(a.shape().size());
    if (ndim <= 0 || ndim > tvv::sycl::kMaxDims)
        throw std::runtime_error(std::string("sycl ") + what + ": unsupported rank");
    if (!a.is_contiguous())
        throw std::runtime_error(std::string("sycl ") + what +
                                 ": input must be contiguous (permute() is a view)");
    if (static_cast<int>(axes.size()) != ndim)
        throw std::runtime_error(std::string("sycl ") + what + ": axes size mismatch");

    auto ctx = sycl_ufunc::context_of(out, what);

    tvv::sycl::Transpose p{};
    p.numel = static_cast<int64_t>(a.numel());
    p.ndim = ndim;
    for (int d = 0; d < ndim; ++d) {
        p.in_shape[d] = a.shape(d);
        p.in_strides[d] = a.strides(d);
        p.axes[d] = axes[d];
    }
    // Output is freshly allocated (or reshaped and contiguous): row-major.
    for (int d = ndim - 1; d >= 0; --d) {
        p.out_strides[d] = (d == ndim - 1) ? 1 : p.out_strides[d + 1] * out.shape(d + 1);
    }

    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_transpose(
                          sycl_ufunc::handle_of(*ctx), a.data(), out.data(),
                          static_cast<int>(a.dtype()), &p),
                      what);
    ctx->wait();
}

std::vector<int64_t> permuted_shape(const Tensor& a,
                                    std::initializer_list<int64_t> axes) {
    std::vector<int64_t> shape;
    shape.reserve(axes.size());
    for (int64_t ax : axes) shape.push_back(a.shape(ax));
    return shape;
}

}  // namespace

void TransposeImpl<Device::SYCL>::execute(Tensor& a) {
    // The public API restricts this form to 2D contiguous tensors.
    Tensor out({a.shape(1), a.shape(0)}, a.dtype(), Device::SYCL);
    launch_permute(a, out, {1, 0}, "Transpose");
    a = std::move(out);
}

Tensor TransposeImpl<Device::SYCL>::execute(const Tensor& a,
                                            std::initializer_list<int64_t> axes) {
    Tensor out(permuted_shape(a, axes), a.dtype(), Device::SYCL);
    launch_permute(a, out, std::vector<int64_t>(axes), "Transpose");
    return out;
}

void TransposeImpl<Device::SYCL>::execute(const Tensor& a, Tensor& dst,
                                          std::initializer_list<int64_t> axes) {
    std::vector<int64_t> shape = permuted_shape(a, axes);
    dst.reshape(shape);
    launch_permute(a, dst, std::vector<int64_t>(axes), "Transpose");
}

}  // namespace ops
