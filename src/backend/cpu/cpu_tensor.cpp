#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <memory_resource>
#include <new>
#include <vector>

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/cpu_tensor.hpp"
#include "interface/cpu_context.hpp"

namespace {

// 64-byte alignment: covers AVX-512 (64B) and AVX2 (32B) for aligned loads.
constexpr size_t kCpuBufferAlignment = 64;

// Per-thread pool: each OpenMP thread gets its own unsynchronized pool, so
// there's zero mutex contention on the hot path. Correctness:
//  - Each thread allocates/frees into its own pool → no race.
//  - If a tensor is allocated in thread A and freed in thread B, B's pool
//    absorbs it (PMR pool treats any same-size block as interchangeable).
//    That can grow B's high-water mark slightly in producer/consumer pipelines,
//    but typical OpenMP parallel-for (each thread does balanced alloc+free
//    inside its own iterations) hits the fast path 100%.
std::pmr::unsynchronized_pool_resource& thread_pool() {
    thread_local std::pmr::unsynchronized_pool_resource pool;
    return pool;
}

void* cpu_pool_allocate(size_t bytes) {
    return thread_pool().allocate(bytes, kCpuBufferAlignment);
}

void cpu_pool_deallocate(void* p, size_t bytes) noexcept {
    if (p == nullptr || bytes == 0) return;
    thread_pool().deallocate(p, bytes, kCpuBufferAlignment);
}

// clone_as_contiguous 的转置型路径：dst 行主序无连续尾段（inner==1），
// 但 src 存在 stride==1 的维度 —— 沿该维连续读、按行主序跨步写，
// 保证读写至少一侧访问连续（2D 转置 / permute 的典型形态）。
// 返回 false 表示不存在可用的 unit-stride 维度，调用方回退 odometer 路径。
bool gather_along_unit_stride(const Metadata& meta, const char* src_base,
                              char* dst, size_t dtype_size) {
    const size_t ndim = meta.shape.size();
    const size_t total = size_t(meta.numel);

    // 选 stride==1 且最长的维度作为连续读方向
    size_t cont_dim = SIZE_MAX;
    int64_t cont_len = 0;
    for (size_t d = 0; d < ndim; ++d) {
        if (meta.strides[d] == 1 && meta.shape[d] > cont_len) {
            cont_dim = d;
            cont_len = meta.shape[d];
        }
    }
    if (cont_dim == SIZE_MAX || cont_len <= 1) return false;

    // 行主序 dst 步长
    std::vector<int64_t> dst_strides(ndim);
    int64_t acc = 1;
    for (size_t d = ndim; d-- > 0;) {
        dst_strides[d] = acc;
        acc *= meta.shape[d];
    }

    // 其余维度做 odometer，同时维护 src / dst 两个段起点偏移
    std::vector<size_t> dims;
    dims.reserve(ndim - 1);
    for (size_t d = 0; d < ndim; ++d)
        if (d != cont_dim) dims.push_back(d);
    std::vector<int64_t> coord(dims.size(), 0);
    size_t src_off = 0, dst_off = 0;
    const size_t runs = total / size_t(cont_len);
    const size_t dst_gap_bytes = size_t(dst_strides[cont_dim]) * dtype_size;

    auto advance = [&]() {
        for (size_t k = dims.size(); k-- > 0;) {
            const size_t d = dims[k];
            ++coord[k];
            src_off += size_t(meta.strides[d]);
            dst_off += size_t(dst_strides[d]);
            if (coord[k] < meta.shape[d]) break;
            coord[k] = 0;
            src_off -= size_t(meta.strides[d]) * size_t(meta.shape[d]);
            dst_off -= size_t(dst_strides[d]) * size_t(meta.shape[d]);
        }
    };
    // 按固定宽度拷贝：运行时 size 的 memcpy 无法内联成单条 move，
    // 这里对 1/2/4/8 字节 dtype 生成直接赋值循环。
    auto run_fixed = [&]<typename U>() {
        const size_t gap = dst_gap_bytes / sizeof(U);
        for (size_t r = 0; r < runs; ++r) {
            auto* t = reinterpret_cast<U*>(dst + dst_off * dtype_size);
            const auto* s = reinterpret_cast<const U*>(src_base + src_off * dtype_size);
            for (int64_t i = 0; i < cont_len; ++i) t[i * gap] = s[i];
            advance();
        }
    };
    switch (dtype_size) {
        case 1: run_fixed.template operator()<std::uint8_t>();  break;
        case 2: run_fixed.template operator()<std::uint16_t>(); break;
        case 4: run_fixed.template operator()<std::uint32_t>(); break;
        case 8: run_fixed.template operator()<std::uint64_t>(); break;
        default: {
            for (size_t r = 0; r < runs; ++r) {
                const char* s = src_base + src_off * dtype_size;
                char* t = dst + dst_off * dtype_size;
                for (int64_t i = 0; i < cont_len; ++i)
                    std::memcpy(t + i * dst_gap_bytes, s + i * dtype_size, dtype_size);
                advance();
            }
        }
    }
    return true;
}

}  // namespace

void cpu_tensor_detail::PoolDeleter::operator()(void* p) const noexcept {
    cpu_pool_deallocate(p, bytes);
}

size_t CPUTensor::numel() const { return m_numel; }

std::shared_ptr<ContextImpl> CPUTensor::context() const {
    // CPUContext 是进程级单例（CPURegistrar 持有），返回带空 deleter 的 aliasing shared_ptr
    // 避免每次构造/析构 CPUTensor 时一对原子引用计数操作。
    return std::shared_ptr<ContextImpl>(static_cast<ContextImpl*>(m_ctx_raw), [](auto*){});
}

void CPUTensor::init(void* ptr, size_t numel, DataType dtype) {
    m_dtype = dtype;
    m_numel = numel;
    size_t total_bytes = m_numel * calc_dtype_size(m_dtype);
    void* raw_ptr = cpu_pool_allocate(total_bytes);
    if (!raw_ptr) throw std::bad_alloc();
    if (ptr != nullptr) std::memcpy(raw_ptr, ptr, total_bytes);
    // Reassign with the byte-count carried in the deleter so the pool knows
    // which size-class freelist to return the block to.
    m_data = std::unique_ptr<void, cpu_tensor_detail::PoolDeleter>(
        raw_ptr, cpu_tensor_detail::PoolDeleter{total_bytes});
}

CPUTensor::CPUTensor(std::shared_ptr<CPUContext> ctx, size_t numel, DataType dtype)
    : m_ctx_raw(ctx.get()) {
    this->init(nullptr, numel, dtype);
}

CPUTensor::CPUTensor(std::shared_ptr<CPUContext> ctx, void* ptr, size_t numel, DataType dtype)
    : m_ctx_raw(ctx.get()) {
    this->init(ptr, numel, dtype);
}

std::unique_ptr<TensorImpl> CPUTensor::clone() const {
    // 单例的 CPUContext 通过 aliasing shared_ptr 再传一次（不增加额外开销路径）
    auto ctx_sp = std::shared_ptr<CPUContext>(this->m_ctx_raw, [](auto*){});
    auto cloned = std::make_unique<CPUTensor>(std::move(ctx_sp), this->m_numel, this->m_dtype);
    size_t bytes = this->m_numel * calc_dtype_size(this->m_dtype);
    std::memcpy(cloned->m_data.get(), this->m_data.get(), bytes);
    return cloned;
}

std::unique_ptr<TensorImpl> CPUTensor::clone_as_contiguous(const Metadata& meta) const {
    auto ctx_sp = std::shared_ptr<CPUContext>(this->m_ctx_raw, [](auto*){});
    auto cloned = std::make_unique<CPUTensor>(std::move(ctx_sp), meta.numel, this->m_dtype);

    size_t dtype_size = calc_dtype_size(this->m_dtype);
    size_t total_elements = meta.numel;
    if (total_elements == 0) return cloned;
    const char* src_base = static_cast<const char*>(this->m_data.get())
                         + meta.offset * dtype_size;
    char* dst = static_cast<char*>(cloned->m_data.get());

    // 1) 找最长连续尾部段：strides 匹配行主序的尾部维度上 src 与 dst
    //    一样连续，可以整段 memcpy（全连续时退化为单次大块拷贝）。
    const size_t ndim = meta.shape.size();
    size_t inner = 1;
    size_t seg_dims = 0;
    for (size_t d = ndim; d-- > 0;) {
        const int64_t expected = (d == ndim - 1)
                               ? 1
                               : meta.strides[d + 1] * meta.shape[d + 1];
        if (meta.shape[d] > 1 && meta.strides[d] == expected) {
            inner *= size_t(meta.shape[d]);
            ++seg_dims;
        } else {
            break;
        }
    }

    // 2) 无连续尾段但存在 unit-stride 维（2D 转置 / permute 典型形态）：
    //    沿该维连续读、跨步写，读写至少一侧连续。
    if (inner == 1 && gather_along_unit_stride(meta, src_base, dst, dtype_size)) {
        return cloned;
    }

    // 3) 段外维度用 odometer（坐标进位加法）维护 src 偏移，
    //    替代旧实现里每个元素一整条 div/mod 重解码；均摊 O(1)/段。
    const size_t outer_count = total_elements / inner;
    const size_t odom_dims = ndim - seg_dims;
    std::vector<int64_t> coord(odom_dims, 0);
    std::vector<int64_t> rad(meta.shape.begin(), meta.shape.begin() + odom_dims);
    std::vector<int64_t> str(meta.strides.begin(), meta.strides.begin() + odom_dims);
    size_t src_off = 0;  // 当前段起点的元素偏移
    const size_t seg_bytes = inner * dtype_size;
    for (size_t s = 0; s < outer_count; ++s) {
        std::memcpy(dst + s * seg_bytes, src_base + src_off * dtype_size, seg_bytes);
        for (size_t d = odom_dims; d-- > 0;) {
            ++coord[d];
            src_off += size_t(str[d]);
            if (coord[d] < rad[d]) break;
            coord[d] = 0;
            src_off -= size_t(str[d]) * size_t(rad[d]);
        }
    }
    return cloned;
}

void CPUTensor::copy_to(std::shared_ptr<TensorImpl> dst) const {
    if (!dst->data()) throw std::runtime_error("Destination pointer is null");
    if (!m_data)       throw std::runtime_error("Source data is null");
    size_t bytes = m_numel * calc_dtype_size(m_dtype);
    std::memcpy(dst->data(), m_data.get(), bytes);
}
