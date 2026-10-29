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

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/factory.hpp"

namespace {

struct Registry {
    std::mutex                                     mtx;
    std::condition_variable                        cv;
    std::unordered_map<Device, TensorImplFactory>  factories;
    std::unordered_map<Device, BackendInitializer> initializers;
    std::unordered_set<Device>                     in_flight;
};

Registry& registry() {
    static Registry r;
    return r;
}

TensorImplFactory find_factory(Device device) {
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

        auto init = reg.initializers.find(device);
        if (init == reg.initializers.end()) return {};

        BackendInitializer fn = std::move(init->second);
        reg.initializers.erase(init);
        reg.in_flight.insert(device);

        std::exception_ptr failure;
        lock.unlock();
        try {
            fn();
        } catch (...) {
            failure = std::current_exception();
        }
        lock.lock();

        reg.in_flight.erase(device);
        reg.cv.notify_all();
        if (failure) std::rethrow_exception(failure);
    }
}

}  // namespace

void register_tensor_impl(Device device, TensorImplFactory factory) {
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

std::shared_ptr<TensorImpl> create_tensor_impl(size_t numel, DataType dtype, Device device) {
    if (numel <= 0) {
        throw std::invalid_argument("numel must be bigger than 0");
    }
    auto factory = find_factory(device);
    if (!factory) {
        throw std::runtime_error(
            std::format("No tensor implementation registered for device: {}", device_to_string(device)));
    }
    return factory(nullptr, numel, dtype);
}

std::shared_ptr<TensorImpl> create_tensor_impl(void* ptr, size_t numel, DataType dtype, Device device) {
    if (ptr == nullptr) {
        throw std::runtime_error("create_tensor_impl: ptr is null");
    }
    if (numel <= 0) {
        throw std::invalid_argument("numel must be bigger than 0");
    }
    auto factory = find_factory(device);
    if (!factory) {
        throw std::runtime_error(
            std::format("No tensor implementation registered for device: {}", device_to_string(device)));
    }
    return factory(ptr, numel, dtype);
}
