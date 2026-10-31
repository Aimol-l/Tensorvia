#pragma once

#include <cstddef>
#include <memory>

// 不含 CUDA 的接口：设备索引、stream 与驱动调用都在 src/backend/cuda/device/
// runtime.cu 定义的 pimpl 里。这个分区的生成头会被 CUDA 翻译单元包含，因此不让它
// 沾 <cuda_runtime.h>，同一份声明就能被普通宿主编译器（以及只构建头文件形态的
// CMake 路径）编译。
#include "tensorvia/types.hpp"
#include "tensorvia/context.hpp"

class CUDAContext : public ContextImpl {
private:
    struct Impl;
    Impl* m_impl = nullptr;

public:
    // 表示一个 stream 的值句柄。两个句柄当且仅当 `handle` 相同才指同一条流；
    // 同时带上 `device`，无需询问驱动也能发现不匹配。
    struct CUDAStream {
        void* handle = nullptr;   // cudaStream_t
        int   device = -1;
    };

    // 选择全局内存最大的那张卡。
    CUDAContext();
    explicit CUDAContext(int device_index);
    ~CUDAContext() override;

    // 本进程能看到的 GPU 数量，以及不带索引的调用方应得的默认卡（全局内存最大的
    // 那张，与默认构造函数一致）。两者都是静态且只在 CUDA TU 里实现，初始化单元
    // 无需直接编译 CUDA runtime 也能探测。
    static int device_count();
    static int preferred_device_index();

    CUDAContext(const CUDAContext&) = delete;
    CUDAContext& operator=(const CUDAContext&) = delete;
    CUDAContext(CUDAContext&&) = delete;
    CUDAContext& operator=(CUDAContext&&) = delete;

    // 不透明的 stream 句柄；stream 由 pimpl 持有。
    void* ctx_raw_ptr() override;
    // 调用线程的 current stream，供 launch 使用。.cu 文件里的调用方把它转成
    // cudaStream_t。
    void* stream() const;
    int   device_index() const;

    // CUDA 的 current device 是线程私有的；所有分配或 launch 的入口都先调用它。
    void activate() const;

    // ---------------------------------------------------------------------
    // Stream。每条（线程, 设备）只有一条流，首次使用时惰性创建；没有换流的入口，
    // 也没有池。跨线程使用张量前必须由原线程 synchronize()。
    // ---------------------------------------------------------------------

    // 调用线程在本设备上的流；首次调用时创建。
    CUDAStream current_stream() const;

    // synchronize_current_stream() 等调用线程的流；synchronize_all() 覆盖本 context
    // 已创建的所有流（即所有用过它的线程）。库其它部分使用的设备级入口是
    // ops::synchronize()，它路由到 synchronize_all()。
    void synchronize_current_stream() const;
    void synchronize_all() const;

    // 设备内存走 context 的缓存分配器：反复创建张量会复用块，而不是每次都付一次
    // 同步 cudaMalloc 的代价。块只在它被释放的那条流上复用，同流按序执行即可保证
    // 安全，热路径不需要事件。两者都先把本设备设为 current；deallocate() 是
    // noexcept，因为张量在析构函数里释放。
    void* allocate(size_t bytes, CUDAStream stream);
    void  deallocate(void* ptr, CUDAStream stream) noexcept;

    // 把所有空闲设备块还给驱动。仍持有内存的张量不受影响。由后端初始化器在进程
    // 退出时调用：工厂的静态注册表（它让 context 保持存活）在 CUDA runtime 自身
    // 拆除之后才析构，从那时调用 cudaFree 会静默失败，所以缓存必须提前释放。
    void empty_cache() noexcept;
};
