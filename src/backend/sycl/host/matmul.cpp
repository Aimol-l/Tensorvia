// Tiled matrix multiplication for Device::SYCL.
//
// A is viewed as [batch, rows, k] and B as [batch, k, cols]; a 2D call is the
// batch=1 case. The host promotes both operands and the destination to a
// common dtype and materializes contiguous buffers, so one scalar type per
// launch is enough here.
#include <cstdint>
#include <memory>
#include <stdexcept>
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

// [rows,k] @ [k,cols] -> [rows,cols]
// [batch,rows,k] @ [batch,k,cols] -> [batch,rows,cols]
Tensor MatMulImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b) {
    const bool batched = a.shape().size() == 3;
    const int64_t batch  = batched ? a.shape(0) : 1;
    const int64_t rows   = batched ? a.shape(1) : a.shape(0);
    const int64_t common = batched ? a.shape(2) : a.shape(1);
    const int64_t cols   = batched ? b.shape(2) : b.shape(1);

    std::vector<int64_t> newshape =
        batched ? std::vector<int64_t>{batch, rows, cols}
                : std::vector<int64_t>{rows, cols};

    // Half-precision operands are widened to FLOAT32 first, exactly like the
    // CPU backend: the 16-bit formats are storage types, not accumulator types.
    Tensor a_storage, b_storage;
    const Tensor* a_p = &a;
    const Tensor* b_p = &b;
    if (a.dtype() == DataType::FLOAT16 || a.dtype() == DataType::BFLOAT16) {
        a_storage = ops::Typecast(a, DataType::FLOAT32);
        a_p = &a_storage;
    }
    if (b.dtype() == DataType::FLOAT16 || b.dtype() == DataType::BFLOAT16) {
        b_storage = ops::Typecast(b, DataType::FLOAT32);
        b_p = &b_storage;
    }

    const DataType res_type = compute_type(a_p->dtype(), b_p->dtype());
    Tensor dst(newshape, res_type, Device::SYCL);
    // The kernel reads contiguous row-major operands; dtype promotion goes
    // through Typecast (no-op when the dtype already matches).
    const Tensor A = a_p->dtype() == res_type ? a_p->contiguous() : ops::Typecast(*a_p, res_type);
    const Tensor B = b_p->dtype() == res_type ? b_p->contiguous() : ops::Typecast(*b_p, res_type);

    auto ctx = sycl_ufunc::context_of(A, "MatMul");
    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_matmul(
                          sycl_ufunc::handle_of(*ctx), A.data(), B.data(), dst.data(),
                          static_cast<int>(res_type), batch, rows, cols, common),
                      "MatMul launch");
    ctx->wait();
    return dst;
}

}  // namespace ops
