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
    // Keyed by (kind, index): one entry per device instance.
    std::unordered_map<DeviceId, TensorImplFactory, DeviceIdHash>  factories;
    std::unordered_map<DeviceId, DeviceControl, DeviceIdHash>      controls;
    // Keyed by kind: an initializer builds one instance, whoever asks for it.
    std::unordered_map<Device, BackendInitializer> initializers;
    std::unordered_map<Device, BackendProbe>       probes;
    // Instances whose initializer has already been invoked. Without this, a
    // backend that declines an index without registering anything would make
    // find_factory loop forever; with it, "tried and got nothing" is final.
    std::unordered_set<DeviceId, DeviceIdHash>     attempted;
    std::unordered_set<DeviceId, DeviceIdHash>     in_flight;
    std::unordered_map<Device, DeviceProbe>        probe_cache;
    std::unordered_set<Device>                     probe_in_flight;
};

Registry& registry() {
    static Registry r;
    return r;
}

// Returns the tensor factory for `device`, running the backend's lazy
// initializer for that instance the first time it is asked for. Empty when the
// backend is absent or the instance does not exist.
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
        // Initialized before, registered nothing: this instance does not exist.
        if (reg.attempted.contains(device)) return {};

        auto init = reg.initializers.find(device.kind);
        if (init == reg.initializers.end()) return {};

        // Copied, not moved: index 1 and index 2 still need to run it.
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
            // Let the next call try again rather than latching a transient
            // failure (a busy driver, say) as "this device does not exist".
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
        // A probe answers once: the device set does not change under us.
        reg.probe_cache[device] = *result;
        return *result;
    }
}

namespace {
// Thread-local, keyed by backend kind: two threads may be working on different
// GPUs of the same backend without coordinating. An entry appears the first
// time the index is asked for, and is then pinned to the probe's default so a
// program that never calls set_device keeps one stable answer.
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
    // A backend registers its control from inside its lazy initializer, so the
    // presence of an entry is exactly "this device instance is up".
    return {reg.controls.begin(), reg.controls.end()};
}

std::shared_ptr<TensorImpl> create_tensor_impl(size_t numel, DataType dtype, DeviceId device) {
    if (numel <= 0) {
        throw std::invalid_argument("numel must be bigger than 0");
    }
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
    if (numel <= 0) {
        throw std::invalid_argument("numel must be bigger than 0");
    }
    auto factory = find_factory(device);
    if (!factory) {
        throw std::runtime_error(std::format(
            "No tensor implementation registered for device: {}", device_id_to_string(device)));
    }
    return factory(ptr, numel, dtype);
}
