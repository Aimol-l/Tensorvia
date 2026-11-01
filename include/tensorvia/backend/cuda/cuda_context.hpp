#pragma once

#include <cstddef>
#include <memory>

#include "tensorvia/types.hpp"
#include "tensorvia/context.hpp"

class CUDAContext : public ContextImpl {
private:
    struct Impl;
    Impl* m_impl = nullptr;

public:
    struct CUDAStream {
        void* handle = nullptr;   // cudaStream_t
        int   device = -1;
    };

    CUDAContext();
    explicit CUDAContext(int device_index);
    ~CUDAContext() override;

    static int device_count();
    static int preferred_device_index();

    CUDAContext(const CUDAContext&) = delete;
    CUDAContext& operator=(const CUDAContext&) = delete;
    CUDAContext(CUDAContext&&) = delete;
    CUDAContext& operator=(CUDAContext&&) = delete;

    void* ctx_raw_ptr() override;
    void* stream() const;
    int   device_index() const;

    void activate() const;

    // ---------------------------------------------------------------------
    // Stream。每条（线程, 设备）只有一条流，首次使用时惰性创建；没有换流的入口，
    // 也没有池。跨线程使用张量前必须由原线程 synchronize()。
    // ---------------------------------------------------------------------

    // 调用线程在本设备上的流；首次调用时创建。
    CUDAStream current_stream() const;

    void synchronize_all() const;
    void synchronize_current_stream() const;

    void empty_cache() noexcept;
    void* allocate(size_t bytes, CUDAStream stream);
    void  deallocate(void* ptr, CUDAStream stream) noexcept;
};
