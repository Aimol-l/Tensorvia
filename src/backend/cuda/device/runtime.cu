// CUDA runtime: context/stream lifetime, allocation, host transfers and the
// CUDATensor buffer management.
//
// Formerly a pure-C island plus two module-side host drivers. The header-first
// artifact compiles all of it as one translation unit, so the context and the
// tensor can use the driver API directly and throw on error.
#include "common.cuh"
#include "allocator.cuh"
#include "cuda_launch.hpp"

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
// Context
// ---------------------------------------------------------------------------

namespace {

// Streams the context hands out on demand. torch's default is 32 per device.
constexpr size_t kStreamPoolSize = 32;

// Contexts never reuse an id, so a thread-local entry left behind by a
// destroyed context can never be read as the current stream of a new one that
// happens to land on the same address.
std::atomic<uint64_t> g_next_context_id{1};

// The current stream is per (thread, context). Thread-local: no lock, and no
// thread can observe another thread's choice.
thread_local std::unordered_map<uint64_t, cudaStream_t> tls_current_stream;

}  // namespace

struct CUDAContext::Impl {
    explicit Impl(int dev) : device(dev), allocator(dev) {}

    int                        device = 0;
    uint64_t                   id = 0;
    cudaStream_t               default_stream = nullptr;
    std::vector<cudaStream_t>  pool;          // created on first pool request
    size_t                     next_pool = 0;
    mutable std::mutex         pool_mtx;      // pool is shared across threads
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

    const cudaError_t err = cudaStreamCreateWithFlags(&impl->default_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        delete impl;
        tvv::cuda::check_cuda(err, "CUDAContext: cudaStreamCreateWithFlags");
    }
    m_impl = impl;
}

CUDAContext::~CUDAContext() {
    if (m_impl == nullptr) return;
    // Destructors do not throw; a failed teardown is not actionable here.
    (void)cudaSetDevice(m_impl->device);
    // The cached blocks go back to the driver before the streams they were
    // recorded on, so no event outlives its stream.
    const cudaStream_t              default_stream = m_impl->default_stream;
    std::vector<cudaStream_t>       pool           = std::move(m_impl->pool);
    delete m_impl;
    for (const cudaStream_t stream : pool) {
        if (stream != nullptr) (void)cudaStreamDestroy(stream);
    }
    if (default_stream != nullptr) (void)cudaStreamDestroy(default_stream);
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
    const auto it = tls_current_stream.find(m_impl->id);
    const cudaStream_t handle =
        it == tls_current_stream.end() ? m_impl->default_stream : it->second;
    return {static_cast<void*>(handle), m_impl->device};
}

CUDAContext::CUDAStream CUDAContext::default_stream() const {
    if (m_impl == nullptr) return {};
    return {static_cast<void*>(m_impl->default_stream), m_impl->device};
}

CUDAContext::CUDAStream CUDAContext::stream_from_pool() {
    if (m_impl == nullptr) throw std::runtime_error("CUDAContext::stream_from_pool: null context");
    // A pool stream is created on the context's device, whatever this thread
    // had current.
    if (const cudaError_t err = cudaSetDevice(m_impl->device); err != cudaSuccess)
        tvv::cuda::check_cuda(err, "CUDAContext::stream_from_pool: cudaSetDevice");

    std::lock_guard<std::mutex> lock(m_impl->pool_mtx);
    if (m_impl->pool.empty()) {
        m_impl->pool.reserve(kStreamPoolSize);
        for (size_t i = 0; i < kStreamPoolSize; ++i) {
            cudaStream_t created = nullptr;
            tvv::cuda::check_cuda(cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking),
                                  "CUDAContext: cudaStreamCreateWithFlags (pool)");
            m_impl->pool.push_back(created);
        }
    }
    const size_t slot = m_impl->next_pool++ % m_impl->pool.size();
    return {static_cast<void*>(m_impl->pool[slot]), m_impl->device};
}

void CUDAContext::set_current_stream(CUDAStream stream) const {
    if (m_impl == nullptr) return;
    tls_current_stream[m_impl->id] = static_cast<cudaStream_t>(stream.handle);
}

CUDAContext::StreamGuard::StreamGuard(const CUDAContext& context, CUDAStream stream)
    : m_context(&context), m_previous(context.current_stream()) {
    context.set_current_stream(stream);
}

CUDAContext::StreamGuard::~StreamGuard() {
    if (m_context != nullptr) m_context->set_current_stream(m_previous);
}

void CUDAContext::synchronize_current_stream() const {
    if (m_impl == nullptr) return;
    tvv::cuda::check_cuda(
        cudaStreamSynchronize(static_cast<cudaStream_t>(current_stream().handle)),
        "cudaStreamSynchronize");
}

void CUDAContext::wait() const { synchronize_current_stream(); }

void CUDAContext::synchronize_all() const {
    if (m_impl == nullptr) return;
    (void)cudaSetDevice(m_impl->device);

    std::vector<cudaStream_t> streams;
    {
        std::lock_guard<std::mutex> lock(m_impl->pool_mtx);
        streams = m_impl->pool;
    }
    streams.push_back(m_impl->default_stream);
    for (const cudaStream_t stream : streams) {
        if (stream == nullptr) continue;
        tvv::cuda::check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
}

void* CUDAContext::allocate(size_t bytes, CUDAStream stream) {
    if (m_impl == nullptr) throw std::runtime_error("CUDAContext::allocate: null context");
    activate();
    // cudaMalloc(0) is an error; a zero-byte tensor still needs a valid handle
    // so the destructor and copy paths stay uniform. The allocator rounds up,
    // so one byte is enough to get a usable block.
    return m_impl->allocator.allocate(bytes == 0 ? 1 : bytes,
                                      static_cast<cudaStream_t>(stream.handle));
}

void CUDAContext::deallocate(void* ptr, CUDAStream stream) noexcept {
    if (m_impl == nullptr || ptr == nullptr) return;
    // device_index() is noexcept, unlike activate().
    (void)cudaSetDevice(m_impl->device);
    m_impl->allocator.deallocate(ptr, static_cast<cudaStream_t>(stream.handle));
}

void CUDAContext::empty_cache() noexcept {
    if (m_impl == nullptr) return;
    (void)cudaSetDevice(m_impl->device);
    m_impl->allocator.empty_cache();
}

// ---------------------------------------------------------------------------
// Device buffers
// ---------------------------------------------------------------------------

namespace {

// The public contract is "copies are complete on return", so a caller can free
// its staging buffer immediately.
cudaError_t copy_bytes(cudaStream_t stream, void* dst, const void* src, size_t bytes,
                       cudaMemcpyKind kind) {
    if (bytes == 0) return cudaSuccess;
    if (dst == nullptr || src == nullptr) return cudaErrorInvalidValue;
    const cudaError_t err = cudaMemcpyAsync(dst, src, bytes, kind, stream);
    if (err != cudaSuccess) return err;
    return cudaStreamSynchronize(stream);
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
    // The allocator handles the zero-byte case and keeps the block for reuse;
    // only pointer validity is this class's concern. The allocation stream is
    // remembered: it is where the block's release gets recorded.
    const size_t bytes = numel * calc_dtype_size(dtype);
    m_stream = m_context->current_stream();
    m_data   = m_context->allocate(bytes, m_stream);

    if (ptr != nullptr) {
        const cudaError_t err = copy_bytes(stream_of(m_context), m_data, ptr, bytes,
                                           cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            // The object's destructor does not run when a constructor throws.
            m_context->deallocate(m_data, m_stream);
            m_data = nullptr;
            tvv::cuda::check_cuda(err, "CUDATensor::init: host-to-device copy");
        }
    }
}

CUDATensor::~CUDATensor() {
    if (m_data != nullptr) {
        // Destructors do not throw; a failed release is not actionable here.
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
    // m_numel is the shared base buffer's element count; dst may be a smaller
    // view. Copying the source length would write past dst's allocation.
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
    m_context->wait();
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
