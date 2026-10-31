#include <condition_variable>
#include <exception>
#include <format>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <cstddef>
#include <functional>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/factory.hpp"

namespace {

struct Registry {
    std::mutex                                     mtx;
    std::condition_variable                        cv;
    // 按 (kind, index) 作键：每个设备实例一条。
    std::unordered_map<DeviceId, TensorImplFactory, DeviceIdHash>  factories;
    std::unordered_map<DeviceId, DeviceControl, DeviceIdHash>      controls;
    // 按 kind 作键：无论谁请求，初始化器只构建一个实例。
    std::unordered_map<Device, BackendInitializer> initializers;
    std::unordered_map<Device, BackendProbe>       probes;
    // 已调用过初始化器的实例。没有它，某个拒绝某索引且什么也不注册的后端会让
    // find_factory 死循环；有了它，“试过但没拿到”就是最终结论。
    std::unordered_set<DeviceId, DeviceIdHash>     attempted;
    std::unordered_set<DeviceId, DeviceIdHash>     in_flight;
    std::unordered_map<Device, DeviceProbe>        probe_cache;
    std::unordered_set<Device>                     probe_in_flight;
};

Registry& registry() {
    static Registry r;
    return r;
}

// 返回 `device` 的张量工厂，首次请求时为该实例运行后端的惰性初始化器。后端缺席
// 或该实例不存在时返回空。
TensorImplFactory find_factory(DeviceId device) {
    auto& reg = registry();
    std::unique_lock<std::mutex> lock(reg.mtx);
    for (;;) {
        if (auto it = reg.factories.find(device); it != reg.factories.end()) {
            return it->second;
        }
        if (reg.in_flight.contains(device)) {
            reg.cv.wait(lock);
            continue;
        }
        // 之前初始化过却没注册任何东西：该实例不存在。
        if (reg.attempted.contains(device)) return {};

        auto init = reg.initializers.find(device.kind);
        if (init == reg.initializers.end()) return {};

        // 拷贝而非移动：索引 1 和索引 2 仍需要运行它。
        BackendInitializer fn = init->second;
        reg.attempted.insert(device);
        reg.in_flight.insert(device);

        std::exception_ptr failure;
        lock.unlock();
        try {
            fn(device.index);
        } catch (...) {
            failure = std::current_exception();
        }
        lock.lock();

        reg.in_flight.erase(device);
        reg.cv.notify_all();
        if (failure) {
            // 让下一次调用重试，而不是把暂时性失败（比如驱动忙）固化成“该设备
            // 不存在”。
            reg.attempted.erase(device);
            std::rethrow_exception(failure);
        }
    }
}

}  // namespace

void register_tensor_impl(DeviceId device, TensorImplFactory factory) {
    auto& reg = registry();
    try {
        std::lock_guard<std::mutex> lock(reg.mtx);
        reg.factories[device] = std::move(factory);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        throw std::runtime_error("register_tensor_impl: failed to register backend");
    }
}

void register_backend_initializer(Device device, BackendInitializer init) {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mtx);
    reg.initializers[device] = std::move(init);
}

void register_device_probe(Device device, BackendProbe probe) {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mtx);
    reg.probes[device] = std::move(probe);
}

DeviceProbe device_probe(Device device) {
    auto& reg = registry();
    std::unique_lock<std::mutex> lock(reg.mtx);
    for (;;) {
        if (auto it = reg.probe_cache.find(device); it != reg.probe_cache.end()) {
            return it->second;
        }
        if (reg.probe_in_flight.contains(device)) {
            reg.cv.wait(lock);
            continue;
        }
        auto probe = reg.probes.find(device);
        if (probe == reg.probes.end()) return {};

        BackendProbe fn = probe->second;
        reg.probe_in_flight.insert(device);

        std::optional<DeviceProbe> result;
        std::exception_ptr          failure;
        lock.unlock();
        try {
            result = fn();
        } catch (...) {
            failure = std::current_exception();
        }
        lock.lock();

        reg.probe_in_flight.erase(device);
        reg.cv.notify_all();
        if (failure) std::rethrow_exception(failure);
        // 探测只答一次：设备集合不会在运行中变化。
        reg.probe_cache[device] = *result;
        return *result;
    }
}

namespace {
// 线程私有，按后端种类作键：两个线程可以各自操作同一后端的不同 GPU 而无需协调。
// 条目在首次询问该索引时出现，随后被钉在探测得到的默认值上，这样从不调用
// set_device 的程序始终得到同一个稳定答案。
thread_local std::unordered_map<Device, int> tls_current_device;
}  // namespace

int current_device_index(Device kind) {
    if (auto it = tls_current_device.find(kind); it != tls_current_device.end()) {
        return it->second;
    }
    const DeviceProbe probe = device_probe(kind);
    if (probe.count <= 0 || probe.default_index < 0) {
        throw std::runtime_error(std::format(
            "no {} device is available", device_to_string(kind)));
    }
    tls_current_device[kind] = probe.default_index;
    return probe.default_index;
}

void set_current_device_index(Device kind, int index) {
    const DeviceProbe probe = device_probe(kind);
    if (probe.count <= 0) {
        throw std::runtime_error(std::format(
            "no {} device is available", device_to_string(kind)));
    }
    if (index < 0 || index >= probe.count) {
        throw std::out_of_range(std::format(
            "{}: device index {} out of range ({} device(s) available)",
            device_to_string(kind), index, probe.count));
    }
    tls_current_device[kind] = index;
}

void register_device_control(DeviceId device, DeviceControl control) {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mtx);
    reg.controls[device] = std::move(control);
}

std::vector<std::pair<DeviceId, DeviceControl>> live_device_controls() {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mtx);
    // 后端在惰性初始化器内部注册自己的 control，所以某条目存在恰好意味着
    // “该设备实例已启动”。
    return {reg.controls.begin(), reg.controls.end()};
}

// numel == 0 是空张量：后端仍会给出有效的（1 字节占位）分配，使 data() 永不为空，
// 统一的析构/拷贝路径得以继续工作。
std::shared_ptr<TensorImpl> create_tensor_impl(size_t numel, DataType dtype, DeviceId device) {
    auto factory = find_factory(device);
    if (!factory) {
        throw std::runtime_error(std::format(
            "No tensor implementation registered for device: {}", device_id_to_string(device)));
    }
    return factory(nullptr, numel, dtype);
}

std::shared_ptr<TensorImpl> create_tensor_impl(void* ptr, size_t numel, DataType dtype, DeviceId device) {
    if (ptr == nullptr) {
        throw std::runtime_error("create_tensor_impl: ptr is null");
    }
    auto factory = find_factory(device);
    if (!factory) {
        throw std::runtime_error(std::format(
            "No tensor implementation registered for device: {}", device_id_to_string(device)));
    }
    return factory(ptr, numel, dtype);
}
