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

std::shared_ptr<SYCLContext> sort_context_of(const Tensor& t, const char* what) {
    const auto impl = std::dynamic_pointer_cast<SYCLTensor>(t.get_impl());
    if (!impl)
        throw std::runtime_error(std::string("sycl ") + what +
                                 ": operand does not live on the SYCL device");
    auto ctx = std::dynamic_pointer_cast<SYCLContext>(impl->context());
    if (!ctx) throw std::runtime_error(std::string("sycl ") + what + ": missing SYCLContext");
    return ctx;
}

Tensor sort_materialize(const Tensor& a) {
    return a.is_contiguous() ? a : a.clone();
}

struct SortGeometry {
    tvv::sycl::SortAxis params{};
    std::vector<int64_t> shape;  // 输出形状（dim 维已换成 keep）
};

SortGeometry sort_geometry(const Tensor& in, int dim, int64_t keep, bool descending,
                           const char* what) {
    const int rank = static_cast<int>(in.shape().size());
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank)
        throw std::runtime_error(std::string("sycl ") + what + ": axis out of range");

    SortGeometry g;
    g.params.outer = 1;
    g.params.n     = in.shape(static_cast<size_t>(dim));
    g.params.inner = 1;
    for (int i = 0; i < dim; ++i) g.params.outer *= in.shape(static_cast<size_t>(i));
    for (size_t i = static_cast<size_t>(dim) + 1; i < in.shape().size(); ++i)
        g.params.inner *= in.shape(i);
    if (keep < 0) keep = g.params.n;  // 全排序
    if (keep > g.params.n)
        throw std::runtime_error(std::format(
            "topk: k ({}) must be between 1 and the axis length ({})", keep, g.params.n));
    g.params.keep       = keep;
    g.params.descending = descending ? 1 : 0;
    if (g.params.n > 1024)
        throw std::runtime_error(std::format(
            "sycl {}: axis length {} exceeds the device sort limit (1024)",
            what, g.params.n));

    int64_t padded = 1;
    while (padded < g.params.n) padded <<= 1;
    g.params.padded = padded;

    g.shape = std::vector<int64_t>(in.shape().begin(), in.shape().end());
    g.shape[static_cast<size_t>(dim)] = keep;
    return g;
}

// sort 与 topk 共用的驱动：排好每列后取前 keep 个。keep < 0 表示全排序。
SortResult run_sort(const Tensor& input, int dim, int64_t keep, bool descending,
                    const char* what) {
    const Tensor src = sort_materialize(input);
    auto ctx = sort_context_of(src, what);
    const SortGeometry g = sort_geometry(src, dim, keep, descending, what);

    Tensor values(g.shape, src.dtype(), Device::SYCL);
    Tensor indices(g.shape, DataType::INT64, Device::SYCL);
    // 空几何（空轴、keep 为 0、没有列）没有可提交的内核，输出就是空张量。
    if (g.params.n <= 0 || g.params.keep <= 0 || g.params.outer * g.params.inner <= 0)
        return SortResult{std::move(values), std::move(indices)};

    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_sort(sycl_ufunc::handle_of(*ctx), src.data(),
                                           values.data(), indices.data(),
                                           static_cast<int>(src.dtype()), &g.params),
                      what);
    return SortResult{std::move(values), std::move(indices)};
}

}  // namespace

SortResult SortImpl<Device::SYCL>::execute(const Tensor& input, int dim, bool descending) {
    return run_sort(input, dim, -1, descending, "Sort");
}

TopkResult TopkImpl<Device::SYCL>::execute(const Tensor& input, int64_t k, int dim,
                                           bool largest) {
    // topk 就是"排好序后取前 k 个"：largest 对应降序。k 的合法性（>=1 且 <= 轴长）
    // 由核心层先把守，这里直接排。
    SortResult sorted = run_sort(input, dim, k, largest, "Topk");
    return TopkResult{std::move(sorted.values), std::move(sorted.indices)};
}

}  // namespace ops
