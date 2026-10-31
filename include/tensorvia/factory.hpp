#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"

using TensorImplFactory =
    std::function<std::shared_ptr<TensorImpl>(void* ptr, size_t numel, DataType)>;
// 参数是正在启动的实例序号，因此一个后端可以持有多个 context（每张卡一个）。
using BackendInitializer = std::function<void(int device_index)>;
// 后端有几个实例、新工作默认落到哪个实例。惰性探测并缓存：答案取决于硬件，
// 而探测本身会把驱动加载起来。
struct DeviceProbe {
    int count = 0;
    int default_index = 0;
};
using BackendProbe = std::function<DeviceProbe()>;

void register_tensor_impl(DeviceId device, TensorImplFactory factory);

// 注册一个惰性回调：某设备实例第一次被请求、却还没有为它注册实现时调用。
// 这样可选后端（Vulkan）在纯 CPU 运行时完全处于休眠状态。
void register_backend_initializer(Device device, BackendInitializer init);

// 注册“如何发现该后端的实例”。每个后端至多调用一次，发生在第一次询问它的设备
// 数量或 current device 时。
void register_device_probe(Device device, BackendProbe probe);
// 该后端的探测结果；首次使用时才真正执行探测。后端没有探测器（不存在或未编入）
// 时返回 {0, 0}。
DeviceProbe device_probe(Device device);

// `kind` 后端上调用线程的 current 实例序号，首次使用时解析为探测器的默认值。
// 后端没有可用设备时抛错：询问一个不存在的后端的 current device 是调用方错误，
// 不能悄悄返回 0。
int  current_device_index(Device kind);
// 设置调用线程的 current 序号，越出 [0, device_probe(kind).count) 的一律拒绝。
void set_current_device_index(Device kind, int index);

// 作用于设备整体、但不是张量计算的算子，因此不放在 Tensor 或后端张量实现上。
// 两个字段都是可选的：后端留空就不会被调用。
struct DeviceControl {
    // 等待已提交到该设备的所有工作——所有流，而不只是调用线程的 current stream。
    // 对应 torch.cuda.synchronize() / torch.accelerator.synchronize()。
    std::function<void()> synchronize;
    // 把可回收的缓存内存还给驱动。仍在使用中的 buffer 不受影响。
    // 对应 torch.cuda.empty_cache()。
    std::function<void()> empty_cache;
};

// 注册方式与张量实现相同：要么立即注册（CPU），要么在设备的惰性初始化里注册
// （加速器）。
void register_device_control(DeviceId device, DeviceControl control);

// 所有已启动设备实例的控制接口。后端从未初始化的实例不在其中，因此请求等待或
// 释放内存不会把休眠中的后端（及其驱动）拉起来。
std::vector<std::pair<DeviceId, DeviceControl>> live_device_controls();

std::shared_ptr<TensorImpl> create_tensor_impl(size_t numel,
                                                      DataType dtype,
                                                      DeviceId device);
std::shared_ptr<TensorImpl> create_tensor_impl(void* ptr,
                                                     size_t numel,
                                                     DataType dtype,
                                                     DeviceId device);

void copy_device_to_host(std::shared_ptr<TensorImpl> src,
                               std::shared_ptr<TensorImpl> dst,
                               DataType dtype);
void copy_host_to_device(std::shared_ptr<TensorImpl> src,
                               std::shared_ptr<TensorImpl> dst,
                               DataType dtype);
