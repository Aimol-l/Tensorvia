#pragma once

#include <cstddef>
#include <memory>

// CUDA-free interface: the device index, stream and driver calls live in the
// pimpl defined by src/backend/cuda/device/runtime.cu. The generated header of
// this partition is included by the CUDA translation units, so keeping it clear
// of <cuda_runtime.h> means the same declaration is also compilable by a plain
// host compiler (and by CMake, where only the header form is built).
#include "interface/types.hpp"
#include "interface/context.hpp"

class CUDAContext : public ContextImpl {
private:
    struct Impl;
    Impl* m_impl = nullptr;

public:
    // Picks the device with the largest global memory.
    CUDAContext();
    explicit CUDAContext(int device_index);
    ~CUDAContext() override;

    CUDAContext(const CUDAContext&) = delete;
    CUDAContext& operator=(const CUDAContext&) = delete;
    CUDAContext(CUDAContext&&) = delete;
    CUDAContext& operator=(CUDAContext&&) = delete;

    // Opaque stream handle; the pimpl owns the stream.
    void* ctx_raw_ptr() override;
    // Same handle, for launches. Callers in .cu files cast it to cudaStream_t.
    void* stream() const;
    int   device_id() const;

    // CUDA's current device is per-thread; every entry point that allocates or
    // launches calls this first.
    void activate() const;

    // The host drivers are synchronous: every launch is followed by wait(), so
    // callers never observe a half-computed tensor.
    void wait() const;
};
