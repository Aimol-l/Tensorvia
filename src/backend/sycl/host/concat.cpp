// Concatenation along one axis for Device::SYCL.
//
// The host entry point launches one kernel per input instead of staging a
// device-side pointer array, so no managed memory or temporary allocation is
// involved.
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "sycl_island.h"

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/ops.hpp"
#include "interface/cpu_ops.hpp"
#include "interface/backend/sycl/sycl_ops.hpp"
#include "interface/backend/sycl/sycl_context.hpp"
#include "interface/backend/sycl/sycl_tensor.hpp"
#include "interface/backend/sycl/sycl_ufunc.hpp"

namespace ops {

Tensor ConcatImpl<Device::SYCL>::execute(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) throw std::runtime_error("Concat: empty inputs");
    if (tensors.size() > static_cast<size_t>(tvv::sycl::kMaxDims))
        throw std::runtime_error("Concat: too many inputs");
    const int ndim = static_cast<int>(tensors[0].shape().size());
    if (dim < 0 || dim >= ndim) throw std::runtime_error("Concat: invalid axis");
    if (ndim > tvv::sycl::kMaxDims) throw std::runtime_error("Concat: too many dimensions");
    const DataType dtype = tensors[0].dtype();

    std::vector<int64_t> out_shape(tensors[0].shape().begin(), tensors[0].shape().end());
    out_shape[dim] = 0;
    for (const auto& t : tensors) out_shape[dim] += t.shape(dim);

    Tensor out(out_shape, dtype, Device::SYCL);
    auto ctx = sycl_ufunc::context_of(out, "Concat");

    const size_t n = tensors.size();
    std::vector<const void*> inputs(n);
    std::vector<int64_t> offsets(n);
    std::vector<int64_t> in_shapes(n * static_cast<size_t>(ndim));
    std::vector<int64_t> in_strides(n * static_cast<size_t>(ndim));
    int64_t prefix = 0;
    for (size_t i = 0; i < n; ++i) {
        // tensor.data() includes the view offset, so a contiguous slice of a
        // larger buffer concatenates its own elements.
        inputs[i] = tensors[i].data();
        offsets[i] = prefix;
        prefix += tensors[i].shape(dim);
        for (int d = 0; d < ndim; ++d) {
            in_shapes[i * ndim + d] = tensors[i].shape(d);
            in_strides[i * ndim + d] = tensors[i].strides(d);
        }
    }

    std::vector<int64_t> out_strides(out.strides().begin(), out.strides().end());
    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_concat(
                          sycl_ufunc::handle_of(*ctx), out.data(), inputs.data(),
                          offsets.data(), static_cast<int32_t>(n),
                          static_cast<int>(dtype), out_strides.data(),
                          in_shapes.data(), in_strides.data(), ndim, dim),
                      "Concat launch");
    ctx->wait();
    return out;
}

}  // namespace ops
