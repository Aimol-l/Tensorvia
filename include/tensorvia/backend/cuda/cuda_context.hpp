#pragma once

#include <cstddef>
#include <memory>

// CUDA-free interface: the device index, stream and driver calls live in the
// pimpl defined by src/backend/cuda/device/runtime.cu. The generated header of
// this partition is included by the CUDA translation units, so keeping it clear
// of <cuda_runtime.h> means the same declaration is also compilable by a plain
// host compiler (and by CMake, where only the header form is built).
#include "tensorvia/types.hpp"
#include "tensorvia/context.hpp"

class CUDAContext : public ContextImpl {
private:
    struct Impl;
    Impl* m_impl = nullptr;

public:
    // Value handle for a stream. Two handles name the same stream iff their
    // `handle` values match; `device` is carried along so a mismatch is
    // detectable without asking the driver.
    struct CUDAStream {
        void* handle = nullptr;   // cudaStream_t
        int   device = -1;
    };

    // Picks the device with the largest global memory.
    CUDAContext();
    explicit CUDAContext(int device_index);
    ~CUDAContext() override;

    // How many GPUs this process can see, and which one a caller that names no
    // index should get (the largest-memory one, matching the default ctor).
    // Both are static and CUDA-only, so the init TU can probe without compiling
    // against the runtime itself.
    static int device_count();
    static int preferred_device_index();

    CUDAContext(const CUDAContext&) = delete;
    CUDAContext& operator=(const CUDAContext&) = delete;
    CUDAContext(CUDAContext&&) = delete;
    CUDAContext& operator=(CUDAContext&&) = delete;

    // Opaque stream handle; the pimpl owns the streams.
    void* ctx_raw_ptr() override;
    // The calling thread's current stream, for launches. Callers in .cu files
    // cast it to cudaStream_t.
    void* stream() const;
    int   device_index() const;

    // CUDA's current device is per-thread; every entry point that allocates or
    // launches calls this first.
    void activate() const;

    // ---------------------------------------------------------------------
    // Streams. The current stream is per thread: two threads sharing one
    // context do not queue behind each other. A thread that never sets one
    // uses the context's default stream.
    // ---------------------------------------------------------------------

    // The calling thread's current stream, or the default until it sets one.
    CUDAStream current_stream() const;
    CUDAStream default_stream() const;
    // Round-robin from the context's stream pool (32 lazily created,
    // non-blocking streams), for callers that want a stream of their own.
    CUDAStream stream_from_pool();
    // Thread-local: affects only the calling thread. Const because it changes
    // no context state.
    void set_current_stream(CUDAStream stream) const;

    // Runs `stream` as the current stream for the enclosing scope, restoring
    // the previous one on exit — the counterpart of torch.cuda.stream(s).
    class StreamGuard {
    public:
        StreamGuard(const CUDAContext& context, CUDAStream stream);
        ~StreamGuard();
        StreamGuard(const StreamGuard&) = delete;
        StreamGuard& operator=(const StreamGuard&) = delete;

    private:
        const CUDAContext* m_context = nullptr;
        CUDAStream         m_previous;
    };

    // The host drivers are synchronous: every launch is followed by wait(), so
    // callers never observe a half-computed tensor. wait() and
    // synchronize_current_stream() wait for the calling thread's current
    // stream; synchronize_all() covers the context's other streams as well.
    // The device-wide entry point the rest of the library calls is
    // ops::synchronize(), which routes to synchronize_all().
    void wait() const;
    void synchronize_current_stream() const;
    void synchronize_all() const;

    // Device memory goes through the context's caching allocator: repeated
    // tensor creation reuses blocks instead of paying a synchronizing
    // cudaMalloc each time. The stream argument is the one the memory is used
    // on, so a block handed to a different stream waits for its release event.
    // Both make this device current first; deallocate() is noexcept because
    // tensors release from their destructor.
    void* allocate(size_t bytes, CUDAStream stream);
    void  deallocate(void* ptr, CUDAStream stream) noexcept;

    // Returns every free device block to the driver. Tensors still holding
    // memory are untouched. Called by the backend initializer at process exit:
    // the factory's static registry (which keeps the context alive) is
    // destroyed after the CUDA runtime's own teardown, and cudaFree from there
    // fails silently, so the cache has to go earlier.
    void empty_cache() noexcept;
};
