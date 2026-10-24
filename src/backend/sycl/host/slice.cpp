// Start/stop slice of a strided input into a contiguous output, for
// Device::SYCL. Every dimension carries a start offset (zero for dimensions
// that are not sliced).
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
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

Tensor SliceImpl<Device::SYCL>::execute(
    const Tensor& t, const std::vector<std::pair<int64_t, int64_t>>& ranges) {
    const int ndim = static_cast<int>(t.shape().size());
    if (ndim > tvv::sycl::kMaxDims) throw std::runtime_error("sycl Slice: too many dimensions");
    if (static_cast<int>(ranges.size()) != ndim)
        throw std::runtime_error("Slice ranges size must be equal to tensor shape size");

    std::vector<int64_t> out_shape(static_cast<size_t>(ndim));
    for (int i = 0; i < ndim; ++i) out_shape[i] = ranges[i].second - ranges[i].first;

    Tensor out(out_shape, t.dtype(), Device::SYCL);
    auto ctx = sycl_ufunc::context_of(out, "Slice");

    tvv::sycl::Slice p{};
    p.numel = static_cast<int64_t>(out.numel());
    // t.data() already points at the view origin; the kernel's offset stays 0.
    p.offset = 0;
    p.ndim = ndim;
    for (int d = 0; d < ndim; ++d) {
        p.in_shape[d] = t.shape(d);
        p.out_shape[d] = out_shape[d];
        p.in_strides[d] = t.strides(d);
        p.starts[d] = ranges[d].first;
    }

    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_slice(
                          sycl_ufunc::handle_of(*ctx), t.data(), out.data(),
                          static_cast<int>(t.dtype()), &p),
                      "Slice launch");
    ctx->wait();
    return out;
}

}  // namespace ops
