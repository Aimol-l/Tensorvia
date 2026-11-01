// SYCL 运行时：队列生命周期、USM 分配、宿主传输以及 SYCLTensor 的 buffer 管理。
//
// 队列本身位于设备单元（runtime.sycl）；本文件只是转发到接缝并把状态码翻译成
// 异常。
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

// 公开契约是“返回时拷贝已完成”，因此调用方可以立刻释放自己的暂存 buffer。
int copy_bytes(void* ctx, void* dst, const void* src, size_t bytes, int kind) {
    if (bytes == 0) return tvv::sycl::kSyclOk;
    if (dst == nullptr || src == nullptr) return tvv::sycl::kSyclInvalid;
    return ::sycl_c_memcpy(ctx, dst, src, bytes, kind);
}

}  // namespace

// ---------------------------------------------------------------------------
// 上下文
// ---------------------------------------------------------------------------

struct SYCLContext::Impl {
    void* handle = nullptr;
};

SYCLContext::SYCLContext() : SYCLContext(-1) {}

SYCLContext::SYCLContext(int device_index) {
    void* handle = ::sycl_c_context_create(device_index);
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
// 设备 buffer
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
    // 先取本线程的队列再分配：buffer 归属这条队列，释放时要交还给它。零字节
    // 张量也需要有效句柄，好让析构和拷贝路径保持统一（接缝把 0 字节映射为 1 字节）。
    m_queue = ::sycl_c_current_queue(m_context->handle());
    if (m_queue == nullptr) throw std::runtime_error("SYCLTensor: cannot create a SYCL queue");

    m_data = ::sycl_c_malloc(m_context->handle(), m_queue, bytes);
    if (m_data == nullptr) throw std::runtime_error("SYCLTensor: device allocation failed");

    if (ptr != nullptr) {
        const int status = copy_bytes(m_context->handle(), m_data, ptr, bytes,
                                      tvv::sycl::kHostToDevice);
        if (status != tvv::sycl::kSyclOk) {
            // 构造函数抛出时，对象的析构函数不会运行。
            ::sycl_c_free(m_context->handle(), m_queue, m_data);
            m_data = nullptr;
            check(status, "SYCLTensor::init: host-to-device copy");
        }
    }
}

SYCLTensor::~SYCLTensor() {
    if (m_data != nullptr) {
        // 析构函数不抛异常；释放失败在这里也无从处理。交还给分配它的队列，排在
        // 引用该 buffer 的既有工作之后。
        ::sycl_c_free(m_context->handle(), m_queue, m_data);
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

    // m_numel 是共享基础 buffer 的元素数；dst 可能是更小的视图。按源长度拷贝会
    // 写越 dst 的分配区。
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
