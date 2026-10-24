// CUDA runtime: context/stream lifetime, allocation, host transfers and the
// CUDATensor buffer management.
//
// Formerly a pure-C island plus two module-side host drivers. The header-first
// artifact compiles all of it as one translation unit, so the context and the
// tensor can use the driver API directly and throw on error.
#include "common.cuh"
#include "cuda_launch.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>

#include "tensorvia/context.hpp"
#include "tensorvia/backend/cuda/cuda_context.hpp"
#include "tensorvia/backend/cuda/cuda_tensor.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/types.hpp"

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct CUDAContext::Impl {
    int          device = 0;
    cudaStream_t stream = nullptr;
};

CUDAContext::CUDAContext() : CUDAContext(-1) {}

CUDAContext::CUDAContext(int device_index) {
    int count = 0;
    tvv::cuda::check_cuda(cudaGetDeviceCount(&count), "CUDAContext: cudaGetDeviceCount");
    if (count <= 0) throw std::runtime_error("No CUDA devices found");

    int chosen = device_index;
    if (chosen < 0) {
        size_t best_memory = 0;
        chosen = 0;
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
            if (static_cast<size_t>(prop.totalGlobalMem) > best_memory) {
                best_memory = static_cast<size_t>(prop.totalGlobalMem);
                chosen = i;
            }
        }
    } else if (chosen >= count) {
        throw std::runtime_error("CUDAContext: device index out of range");
    }

    tvv::cuda::check_cuda(cudaSetDevice(chosen), "CUDAContext: cudaSetDevice");

    auto* impl = new (std::nothrow) Impl;
    if (impl == nullptr) throw std::bad_alloc();
    impl->device = chosen;

    const cudaError_t err = cudaStreamCreateWithFlags(&impl->stream, cudaStreamNonBlocking);
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
    if (m_impl->stream != nullptr) (void)cudaStreamDestroy(m_impl->stream);
    delete m_impl;
}

void* CUDAContext::ctx_raw_ptr() { return stream(); }

void* CUDAContext::stream() const {
    return m_impl == nullptr ? nullptr : static_cast<void*>(m_impl->stream);
}

int CUDAContext::device_id() const {
    return m_impl == nullptr ? -1 : m_impl->device;
}

void CUDAContext::activate() const {
    tvv::cuda::check_cuda(cudaSetDevice(m_impl->device), "cudaSetDevice");
}

void CUDAContext::wait() const {
    tvv::cuda::check_cuda(cudaStreamSynchronize(m_impl->stream), "cudaStreamSynchronize");
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
    // cudaMalloc(0) is an error; a zero-byte tensor still needs a valid handle
    // so the destructor and copy paths stay uniform.
    const size_t bytes = numel * calc_dtype_size(dtype);
    tvv::cuda::check_cuda(cudaMalloc(&m_data, bytes == 0 ? 1 : bytes), "CUDATensor::init: cudaMalloc");

    if (ptr != nullptr) {
        const cudaError_t err = copy_bytes(stream_of(m_context), m_data, ptr, bytes,
                                           cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            // The object's destructor does not run when a constructor throws.
            (void)cudaFree(m_data);
            m_data = nullptr;
            tvv::cuda::check_cuda(err, "CUDATensor::init: host-to-device copy");
        }
    }
}

CUDATensor::~CUDATensor() {
    if (m_data != nullptr) {
        // Destructors do not throw; a failed release is not actionable here.
        // device_id() is noexcept, unlike CUDAContext::activate().
        (void)cudaSetDevice(m_context->device_id());
        (void)cudaFree(m_data);
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
