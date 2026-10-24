#pragma once

#include <cstddef>
#include <memory>
#include <string>

// SYCL-free interface: the queue and the USM allocator live in the opaque
// handle created by src/backend/sycl/device/runtime.sycl. The generated header
// of this partition is included by both the module-lane host drivers and the
// generated header-mode copies, so it must stay clear of <sycl/sycl.hpp>.
#include "interface/types.hpp"
#include "interface/context.hpp"

class SYCLContext : public ContextImpl {
private:
    struct Impl;
    Impl* m_impl = nullptr;

public:
    // Selects a device with sycl::default_selector_v (GPU preferred, OpenCL CPU
    // as the fallback). Throws when no device is usable.
    SYCLContext();
    ~SYCLContext() override;

    SYCLContext(const SYCLContext&) = delete;
    SYCLContext& operator=(const SYCLContext&) = delete;
    SYCLContext(SYCLContext&&) = delete;
    SYCLContext& operator=(SYCLContext&&) = delete;

    // Opaque queue handle; the pimpl owns the queue. This is the `ctx` argument
    // of every sycl_c_* seam function.
    void* ctx_raw_ptr() override;
    void* handle() const;

    // The queue owns its device, so there is no per-thread current device to
    // switch to. Kept for symmetry with the other backends' drivers.
    void activate() const {}

    // The host drivers are synchronous: every launch is followed by wait(), so
    // callers never observe a half-computed tensor.
    void wait() const;

    // Diagnostic name of the selected device, e.g. "Intel(R) Core(TM) ...".
    std::string device_name() const;
};
