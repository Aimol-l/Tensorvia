// CUDA 运行时：上下文/stream 生命周期、内存分配、宿主传输以及 CUDATensor 的
// buffer 管理。
//
// 过去是一个纯 C island 加两个模块侧宿主驱动。header-first 产物把它们全部作为一个
// 翻译单元编译，因此上下文和张量可以直接使用驱动 API 并在出错时抛异常。
#include "common.cuh"
#include "allocator.cuh"
#include "cuda_launch.hpp"

#include "_sync_counters.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <format>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "tensorvia/context.hpp"
#include "tensorvia/backend/cuda/cuda_context.hpp"
#include "tensorvia/backend/cuda/cuda_tensor.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/types.hpp"

// ---------------------------------------------------------------------------
// 上下文
// ---------------------------------------------------------------------------

namespace {

// 上下文 id 永不复用，因此已销毁上下文残留的线程私有条目绝不会被误当成恰好落在
// 同一地址上的新上下文的流。
std::atomic<uint64_t> g_next_context_id{1};

// 每条（线程, 上下文）一条流。线程私有：查表无需加锁，也没有线程能看到别的线程的
// 流；新创建的流同时登记到 context 的注册表，供 synchronize_all 与析构使用。
thread_local std::unordered_map<uint64_t, cudaStream_t> tls_stream;

}  // namespace

struct CUDAContext::Impl {
    explicit Impl(int dev) : device(dev), allocator(dev) {}

    int                        device = 0;
    uint64_t                   id = 0;
    std::vector<cudaStream_t>  streams;       // 所有线程已创建的流
    mutable std::mutex         streams_mtx;
    tvv::cuda::CachingAllocator allocator;
};

CUDAContext::CUDAContext() : CUDAContext(preferred_device_index()) {}

int CUDAContext::device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count;
}

int CUDAContext::preferred_device_index() {
    const int count = device_count();
    if (count <= 0) return -1;
    int    chosen    = 0;
    size_t best_memory = 0;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
        if (static_cast<size_t>(prop.totalGlobalMem) > best_memory) {
            best_memory = static_cast<size_t>(prop.totalGlobalMem);
            chosen      = i;
        }
    }
    return chosen;
}

CUDAContext::CUDAContext(int device_index) {
    int count = 0;
    tvv::cuda::check_cuda(cudaGetDeviceCount(&count), "CUDAContext: cudaGetDeviceCount");
    if (count <= 0) throw std::runtime_error("No CUDA devices found");

    int chosen = device_index;
    if (chosen < 0) {
        chosen = preferred_device_index();
        if (chosen < 0) throw std::runtime_error("No CUDA devices found");
    } else if (chosen >= count) {
        throw std::runtime_error(std::format(
            "CUDAContext: device index {} out of range ({} device(s) visible)", chosen, count));
    }

    tvv::cuda::check_cuda(cudaSetDevice(chosen), "CUDAContext: cudaSetDevice");

    auto* impl = new (std::nothrow) Impl(chosen);
    if (impl == nullptr) throw std::bad_alloc();
    impl->id = g_next_context_id.fetch_add(1, std::memory_order_relaxed);

    m_impl = impl;
}

CUDAContext::~CUDAContext() {
    if (m_impl == nullptr) return;
    // 析构函数不抛异常；清理失败在这里也无从处理。
    (void)cudaSetDevice(m_impl->device);
    // 缓存的块先于使用它们的流归还驱动，因此分配器里没有东西会比流活得更久。
    std::vector<cudaStream_t> streams = std::move(m_impl->streams);
    delete m_impl;
    for (const cudaStream_t stream : streams) {
        if (stream != nullptr) (void)cudaStreamDestroy(stream);
    }
}

void* CUDAContext::ctx_raw_ptr() { return stream(); }

void* CUDAContext::stream() const { return current_stream().handle; }

int CUDAContext::device_index() const {
    return m_impl == nullptr ? -1 : m_impl->device;
}

void CUDAContext::activate() const {
    tvv::cuda::check_cuda(cudaSetDevice(m_impl->device), "cudaSetDevice");
}

CUDAContext::CUDAStream CUDAContext::current_stream() const {
    if (m_impl == nullptr) return {};
    const auto it = tls_stream.find(m_impl->id);
    if (it != tls_stream.end()) return {static_cast<void*>(it->second), m_impl->device};

    // 首次使用时创建。context 自己的设备在这里确定，与本线程的 current device 无关。
    if (const cudaError_t err = cudaSetDevice(m_impl->device); err != cudaSuccess)
        tvv::cuda::check_cuda(err, "CUDAContext::current_stream: cudaSetDevice");

    cudaStream_t created = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_impl->streams_mtx);
        tvv::cuda::check_cuda(cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking),
                              "CUDAContext: cudaStreamCreateWithFlags");
        m_impl->streams.push_back(created);
    }
    tls_stream[m_impl->id] = created;
    return {static_cast<void*>(created), m_impl->device};
}

void CUDAContext::synchronize_current_stream() const {
    if (m_impl == nullptr) return;
    tvv::cuda::check_cuda(
        cudaStreamSynchronize(static_cast<cudaStream_t>(current_stream().handle)),
        "cudaStreamSynchronize");
    tensorvia::sync::note_wait(tensorvia::sync::Wait::Synchronize);
}

void CUDAContext::synchronize_all() const {
    if (m_impl == nullptr) return;
    (void)cudaSetDevice(m_impl->device);

    std::vector<cudaStream_t> streams;
    {
        std::lock_guard<std::mutex> lock(m_impl->streams_mtx);
        streams = m_impl->streams;
    }
    for (const cudaStream_t stream : streams) {
        if (stream == nullptr) continue;
        tvv::cuda::check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        tensorvia::sync::note_wait(tensorvia::sync::Wait::Synchronize);
    }
}

void* CUDAContext::allocate(size_t bytes, CUDAStream stream) {
    if (m_impl == nullptr) throw std::runtime_error("CUDAContext::allocate: null context");
    activate();
    // cudaMalloc(0) 是错误；零字节张量仍需一个有效句柄，好让析构和拷贝路径保持
    // 统一。分配器会向上取整，所以一字节就足以拿到一个可用块。
    return m_impl->allocator.allocate(bytes == 0 ? 1 : bytes,
                                      static_cast<cudaStream_t>(stream.handle));
}

void CUDAContext::deallocate(void* ptr, CUDAStream stream) noexcept {
    if (m_impl == nullptr || ptr == nullptr) return;
    // device_index() 是 noexcept，与 activate() 不同。
    (void)cudaSetDevice(m_impl->device);
    m_impl->allocator.deallocate(ptr, static_cast<cudaStream_t>(stream.handle));
}

void CUDAContext::empty_cache() noexcept {
    if (m_impl == nullptr) return;
    (void)cudaSetDevice(m_impl->device);
    m_impl->allocator.empty_cache();
}

// ---------------------------------------------------------------------------
// 设备 buffer
// ---------------------------------------------------------------------------

namespace {

// 公开契约是“返回时拷贝已完成”，因此调用方可以立刻释放自己的暂存 buffer。
cudaError_t copy_bytes(cudaStream_t stream, void* dst, const void* src, size_t bytes,
                       cudaMemcpyKind kind) {
    if (bytes == 0) return cudaSuccess;
    if (dst == nullptr || src == nullptr) return cudaErrorInvalidValue;
    const cudaError_t err = cudaMemcpyAsync(dst, src, bytes, kind, stream);
    if (err != cudaSuccess) return err;
    const cudaError_t waited = cudaStreamSynchronize(stream);
    // 每次“返回时拷贝已完成”的契约都是一次宿主等待：FR-4.4 的可观测计数。
    tensorvia::sync::note_wait(tensorvia::sync::Wait::Copy);
    return waited;
}

cudaStream_t stream_of(const std::shared_ptr<CUDAContext>& ctx) {
    return static_cast<cudaStream_t>(ctx->stream());
}

}  // namespace

CUDATensor::CUDATensor(size_t numel, DataType dtype, std::shared_ptr<CUDAContext> context) {
    init(nullptr, numel, dtype, std::move(context));
}

CUDATensor::CUDATensor(void* ptr, size_t numel, DataType dtype,
                       std::shared_ptr<CUDAContext> context) {
    init(ptr, numel, dtype, std::move(context));
}

void CUDATensor::init(void* ptr, size_t numel, DataType dtype,
                      std::shared_ptr<CUDAContext> context) {
    m_numel = numel;
    m_dtype = dtype;
    m_context = std::move(context);
    if (!m_context) throw std::runtime_error("CUDATensor: null CUDAContext");

    m_context->activate();
    // 分配器处理零字节情况并保留块以便复用；本类只关心指针是否有效。分配时所在的
    // stream 会被记住：块的释放事件就记录在那条 stream 上。
    const size_t bytes = numel * calc_dtype_size(dtype);
    m_stream = m_context->current_stream();
    m_data   = m_context->allocate(bytes, m_stream);

    if (ptr != nullptr) {
        const cudaError_t err = copy_bytes(stream_of(m_context), m_data, ptr, bytes,
                                           cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            // 构造函数抛出时，对象的析构函数不会运行。
            m_context->deallocate(m_data, m_stream);
            m_data = nullptr;
            tvv::cuda::check_cuda(err, "CUDATensor::init: host-to-device copy");
        }
    }
}

CUDATensor::~CUDATensor() {
    if (m_data != nullptr) {
        // 析构函数不抛异常；释放失败在这里也无从处理。
        m_context->deallocate(m_data, m_stream);
    }
}

std::shared_ptr<ContextImpl> CUDATensor::context() const {
    return m_context;
}

void CUDATensor::copy_to(std::shared_ptr<TensorImpl> dst) const {
    auto cuda_dst = std::dynamic_pointer_cast<CUDATensor>(dst);
    if (!cuda_dst) throw std::invalid_argument("copy_to: destination is not a CUDATensor");
    if (m_dtype != cuda_dst->m_dtype) {
        throw std::invalid_argument("Tensor dtype mismatch in copy_to!");
    }
    if (m_context != cuda_dst->m_context) {
        throw std::invalid_argument("Cannot copy between different CUDA contexts!");
    }

    m_context->activate();
    // m_numel 是共享基础 buffer 的元素数；dst 可能是更小的视图。按源长度拷贝会写
    // 越 dst 的分配区。
    const size_t count = std::min(m_numel, cuda_dst->m_numel);
    tvv::cuda::check_cuda(copy_bytes(stream_of(m_context), cuda_dst->m_data, m_data,
                          count * calc_dtype_size(m_dtype), cudaMemcpyDeviceToDevice),
               "CUDATensor::copy_to");
}

std::unique_ptr<TensorImpl> CUDATensor::clone() const {
    auto cloned = std::make_unique<CUDATensor>(m_numel, m_dtype, m_context);
    m_context->activate();
    tvv::cuda::check_cuda(copy_bytes(stream_of(m_context), cloned->m_data, m_data,
                          m_numel * calc_dtype_size(m_dtype), cudaMemcpyDeviceToDevice),
               "CUDATensor::clone");
    return cloned;
}

std::unique_ptr<TensorImpl> CUDATensor::clone_as_contiguous(const Metadata& meta) const {
    auto cloned = std::make_unique<CUDATensor>(meta.numel, m_dtype, m_context);
    m_context->activate();
    tvv::cuda::check_cuda(tvv::cuda::launch_repack(stream_of(m_context), m_data, cloned->m_data, meta),
               "repack");
    return cloned;
}

void CUDATensor::copy_to_host(void* dst, DataType dtype) const {
    if (dst == nullptr) throw std::invalid_argument("CUDATensor::copy_to_host: null destination");
    m_context->activate();
    tvv::cuda::check_cuda(copy_bytes(stream_of(m_context), dst, m_data, m_numel * calc_dtype_size(dtype),
                          cudaMemcpyDeviceToHost),
               "CUDATensor::copy_to_host");
}

void CUDATensor::copy_from_host(const void* src, DataType dtype) {
    if (src == nullptr) throw std::invalid_argument("CUDATensor::copy_from_host: null source");
    m_context->activate();
    tvv::cuda::check_cuda(copy_bytes(stream_of(m_context), m_data, src, m_numel * calc_dtype_size(dtype),
                          cudaMemcpyHostToDevice),
               "CUDATensor::copy_from_host");
}
