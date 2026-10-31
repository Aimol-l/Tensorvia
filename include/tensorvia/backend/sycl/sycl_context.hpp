#pragma once

#include <cstddef>
#include <memory>
#include <string>

// 不含 SYCL 的接口：queue 与 USM 分配器都在 src/backend/sycl/device/runtime.sycl
// 创建的不透明句柄里。宿主驱动（.cpp）与设备单元都包含这个头，因此它必须避开
// <sycl/sycl.hpp>。
#include "tensorvia/types.hpp"
#include "tensorvia/context.hpp"

class SYCLContext : public ContextImpl {
private:
    struct Impl;
    Impl* m_impl = nullptr;

public:
    // 用 sycl::default_selector_v 选设备（优先 GPU，回退到 OpenCL CPU）。
    // 没有可用设备时抛错。
    SYCLContext();
    // 取 sycl::device::get_devices() 的第 `device_index` 个；探测器的默认索引就是
    // 默认构造函数会选中的那个。
    explicit SYCLContext(int device_index);
    ~SYCLContext() override;

    SYCLContext(const SYCLContext&) = delete;
    SYCLContext& operator=(const SYCLContext&) = delete;
    SYCLContext(SYCLContext&&) = delete;
    SYCLContext& operator=(SYCLContext&&) = delete;

    // 不透明的 queue 句柄；queue 由 pimpl 持有。所有 sycl_c_* seam 函数的 `ctx`
    // 参数就是它。
    void* ctx_raw_ptr() override;
    void* handle() const;

    // queue 自带设备，没有可切换的线程级 current device。保留此函数只是为了与其它
    // 后端的驱动保持对称。
    void activate() const {}

    // 等队列清空，即设备级同步点（ops::synchronize()）。算子入队后不在这里等待；
    // 宿主读数据前由 to_host/item<T>() 负责同步。
    void wait() const;

    // 所选设备的诊断名称，例如 "Intel(R) Core(TM) ..."。
    std::string device_name() const;
};
