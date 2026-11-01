// sort / topk ——Device::VULKAN 宿主驱动。
//
// 与 CUDA/SYCL 后端同一套契约：排序键是 (值, 原始下标) 对（天然稳定，与 CPU 的
// std::stable_sort 语义逐位一致），按轴长分派两条着色器：
//   n <= 32    sort_small：一个 invocation 一列，私有内存插入排序；
//   n <= 1024  sort_block：一个工作组一列，共享内存 bitonic（shared 1024 对，
//              double 下 16KB，正是 Vulkan 保证的计算共享内存下限）；
// 轴长超过 1024 显式抛错。topk 就是"排好序后取前 k 个"（largest -> 降序），
// 与 sort 共用同一条路径，keep 与 descending 承载区别。
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/cpu_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"

namespace ops {

namespace {

inline void check_sort_vk(const std::shared_ptr<TensorImpl>& impl, const char* what) {
    if (!impl) throw std::runtime_error(std::string("vulkan ") + what + ": expected a VKTensor");
}

struct SortGeometry {
    SortParams params{};
    std::vector<int64_t> shape;  // 输出形状（dim 维已换成 keep）
    bool block_path = false;
};

SortGeometry sort_geometry(const Tensor& in, int dim, int64_t keep, bool descending,
                           const char* what) {
    const int rank = static_cast<int>(in.shape().size());
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank)
        throw std::runtime_error(std::string("vulkan ") + what + ": axis out of range");

    SortGeometry g;
    g.params.outer = 1;
    g.params.n     = static_cast<int32_t>(in.shape(static_cast<size_t>(dim)));
    g.params.inner = 1;
    for (int i = 0; i < dim; ++i)
        g.params.outer *= static_cast<int32_t>(in.shape(static_cast<size_t>(i)));
    for (size_t i = static_cast<size_t>(dim) + 1; i < in.shape().size(); ++i)
        g.params.inner *= static_cast<int32_t>(in.shape(i));
    if (keep < 0) keep = g.params.n;  // 全排序
    if (keep > g.params.n)
        throw std::runtime_error(std::format(
            "topk: k ({}) must be between 0 and the axis length ({})", keep, g.params.n));
    g.params.keep       = static_cast<int32_t>(keep);
    g.params.descending = descending ? 1 : 0;
    if (g.params.n > 1024)
        throw std::runtime_error(std::format(
            "vulkan {}: axis length {} exceeds the device sort limit (1024)",
            what, g.params.n));

    int32_t padded = 1;
    while (padded < g.params.n) padded <<= 1;
    g.params.padded = padded;
    g.block_path   = g.params.n > 32;

    g.shape = std::vector<int64_t>(in.shape().begin(), in.shape().end());
    g.shape[static_cast<size_t>(dim)] = keep;
    return g;
}

// sort 与 topk 共用的驱动。keep < 0 表示全排序。
SortResult run_sort(const Tensor& input, int dim, int64_t keep, bool descending,
                    const char* what) {
    // 内核按扁平行主序寻址，非连续视图先物化；传下去的是 Tensor::data()
    // （已含视图 offset）。
    const Tensor src = input.is_contiguous() ? input : input.clone();
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(src.get_impl());
    check_sort_vk(src_impl, what);
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error(std::string("vulkan ") + what +
                                            ": missing VulkanContext");
    vulkan_detail::require_int32_indexable(src.numel(), what);

    const SortGeometry g = sort_geometry(src, dim, keep, descending, what);
    Tensor values(g.shape, src.dtype(), Device::VULKAN);
    Tensor indices(g.shape, DataType::INT64, Device::VULKAN);
    auto vals_impl = std::dynamic_pointer_cast<VKTensor>(values.get_impl());
    auto idx_impl  = std::dynamic_pointer_cast<VKTensor>(indices.get_impl());
    check_sort_vk(vals_impl, what);
    check_sort_vk(idx_impl, what);

    // 空几何（空轴、keep 为 0、没有列）没有可派发的网格，输出就是空张量。
    if (g.params.n <= 0 || g.params.keep <= 0 || g.params.outer * g.params.inner <= 0)
        return SortResult{std::move(values), std::move(indices)};

    const int64_t columns = static_cast<int64_t>(g.params.outer) * g.params.inner;
    const uint32_t groups =
        g.block_path ? static_cast<uint32_t>(columns)
                     : static_cast<uint32_t>((columns + 255) / 256);
    ctx_impl->submitCompute(g.block_path ? OpType::SortBlock : OpType::SortSmall,
                            src.dtype(),
                            {src_impl->buffer(), vals_impl->buffer(), idx_impl->buffer()},
                            groups, 1, 1, &g.params, sizeof(SortParams));
    return SortResult{std::move(values), std::move(indices)};
}

}  // namespace

SortResult SortImpl<Device::VULKAN>::execute(const Tensor& input, int dim, bool descending) {
    return run_sort(input, dim, -1, descending, "Sort");
}

TopkResult TopkImpl<Device::VULKAN>::execute(const Tensor& input, int64_t k, int dim,
                                             bool largest) {
    // topk 就是"排好序后取前 k 个"：largest 对应降序。k 的合法性（>=1 且 <= 轴长）
    // 由核心层先把守，这里直接排。
    SortResult sorted = run_sort(input, dim, k, largest, "Topk");
    return TopkResult{std::move(sorted.values), std::move(sorted.indices)};
}

}  // namespace ops
