// SYCL runtime: queue lifetime, USM allocation, host transfers and the
// SYCLTensor buffer management.
//
// The queue itself lives in the device unit (runtime.sycl); this file only
// forwards to the seam and translates the status codes into exceptions.
#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

#include "sycl_island.h"

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/context.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/backend/sycl/sycl_context.hpp"
#include "tensorvia/backend/sycl/sycl_tensor.hpp"

namespace {

void check(int status, const char* what) {
    if (status == tvv::sycl::kSyclOk) return;
    throw std::runtime_error(std::string("SYCL error in ") + what);
}

// The public contract is "copies are complete on return", so a caller can free
// its staging buffer immediately.
int copy_bytes(void* ctx, void* dst, const void* src, size_t bytes, int kind) {
    if (bytes == 0) return tvv::sycl::kSyclOk;
    if (dst == nullptr || src == nullptr) return tvv::sycl::kSyclInvalid;
    return ::sycl_c_memcpy(ctx, dst, src, bytes, kind);
}

}  // namespace

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct SYCLContext::Impl {
    void* handle = nullptr;
};

SYCLContext::SYCLContext() {
    void* handle = ::sycl_c_context_create();
    if (handle == nullptr) throw std::runtime_error("no SYCL device available");

    auto* impl = new (std::nothrow) Impl;
    if (impl == nullptr) {
        ::sycl_c_context_destroy(handle);
        throw std::bad_alloc();
    }
    impl->handle = handle;
    m_impl = impl;
}

SYCLContext::~SYCLContext() {
    if (m_impl == nullptr) return;
    ::sycl_c_context_destroy(m_impl->handle);
    delete m_impl;
}

void* SYCLContext::ctx_raw_ptr() { return handle(); }

void* SYCLContext::handle() const {
    return m_impl == nullptr ? nullptr : m_impl->handle;
}

void SYCLContext::wait() const {
    check(::sycl_c_wait(handle()), "SYCLContext::wait");
}

std::string SYCLContext::device_name() const {
    char buffer[256] = {};
    const int n = ::sycl_c_device_name(handle(), buffer, sizeof(buffer));
    if (n <= 0) return {};
    return std::string(buffer, static_cast<size_t>(n));
}

// ---------------------------------------------------------------------------
// Device buffers
// ---------------------------------------------------------------------------

SYCLTensor::SYCLTensor(size_t numel, DataType dtype, std::shared_ptr<SYCLContext> context) {
    init(nullptr, numel, dtype, std::move(context));
}

SYCLTensor::SYCLTensor(void* ptr, size_t numel, DataType dtype,
                       std::shared_ptr<SYCLContext> context) {
    init(ptr, numel, dtype, std::move(context));
}

void SYCLTensor::init(void* ptr, size_t numel, DataType dtype,
                      std::shared_ptr<SYCLContext> context) {
    m_numel = numel;
    m_dtype = dtype;
    m_context = std::move(context);
    if (!m_context) throw std::runtime_error("SYCLTensor: null SYCLContext");

    const size_t bytes = numel * calc_dtype_size(dtype);
    // A zero-byte tensor still needs a valid handle so the destructor and copy
    // paths stay uniform (the seam maps 0 bytes to one byte).
    m_data = ::sycl_c_malloc(m_context->handle(), bytes);
    if (m_data == nullptr) throw std::runtime_error("SYCLTensor: device allocation failed");

    if (ptr != nullptr) {
        const int status = copy_bytes(m_context->handle(), m_data, ptr, bytes,
                                      tvv::sycl::kHostToDevice);
        if (status != tvv::sycl::kSyclOk) {
            // The object's destructor does not run when a constructor throws.
            ::sycl_c_free(m_context->handle(), m_data);
            m_data = nullptr;
            check(status, "SYCLTensor::init: host-to-device copy");
        }
    }
}

SYCLTensor::~SYCLTensor() {
    if (m_data != nullptr) {
        // Destructors do not throw; a failed release is not actionable here.
        ::sycl_c_free(m_context->handle(), m_data);
        m_data = nullptr;
    }
}

std::shared_ptr<ContextImpl> SYCLTensor::context() const {
    return m_context;
}

void SYCLTensor::copy_to(std::shared_ptr<TensorImpl> dst) const {
    auto sycl_dst = std::dynamic_pointer_cast<SYCLTensor>(dst);
    if (!sycl_dst) throw std::invalid_argument("copy_to: destination is not a SYCLTensor");
    if (m_dtype != sycl_dst->m_dtype) {
        throw std::invalid_argument("Tensor dtype mismatch in copy_to!");
    }
    if (m_context != sycl_dst->m_context) {
        throw std::invalid_argument("Cannot copy between different SYCL contexts!");
    }

    // m_numel is the shared base buffer's element count; dst may be a smaller
    // view. Copying the source length would write past dst's allocation.
    const size_t count = std::min(m_numel, sycl_dst->m_numel);
    check(copy_bytes(m_context->handle(), sycl_dst->m_data, m_data,
                     count * calc_dtype_size(m_dtype), tvv::sycl::kDeviceToDevice),
          "SYCLTensor::copy_to");
}

std::unique_ptr<TensorImpl> SYCLTensor::clone() const {
    auto cloned = std::make_unique<SYCLTensor>(m_numel, m_dtype, m_context);
    check(copy_bytes(m_context->handle(), cloned->m_data, m_data,
                     m_numel * calc_dtype_size(m_dtype), tvv::sycl::kDeviceToDevice),
          "SYCLTensor::clone");
    return cloned;
}

std::unique_ptr<TensorImpl> SYCLTensor::clone_as_contiguous(const Metadata& meta) const {
    auto cloned = std::make_unique<SYCLTensor>(meta.numel, m_dtype, m_context);
    check(::sycl_c_launch_repack(
              m_context->handle(), m_data, cloned->m_data,
              static_cast<int64_t>(meta.numel), meta.offset,
              static_cast<int32_t>(meta.shape.size()),
              static_cast<int32_t>(calc_dtype_size(meta.dtype)),
              meta.shape.data(), meta.strides.data()),
          "repack");
    m_context->wait();
    return cloned;
}

void SYCLTensor::copy_to_host(void* dst, DataType dtype) const {
    if (dst == nullptr) throw std::invalid_argument("SYCLTensor::copy_to_host: null destination");
    check(copy_bytes(m_context->handle(), dst, m_data, m_numel * calc_dtype_size(dtype),
                     tvv::sycl::kDeviceToHost),
          "SYCLTensor::copy_to_host");
}

void SYCLTensor::copy_from_host(const void* src, DataType dtype) {
    if (src == nullptr) throw std::invalid_argument("SYCLTensor::copy_from_host: null source");
    check(copy_bytes(m_context->handle(), m_data, src, m_numel * calc_dtype_size(dtype),
                     tvv::sycl::kHostToDevice),
          "SYCLTensor::copy_from_host");
}
