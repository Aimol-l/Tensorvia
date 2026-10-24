#include <format>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <memory>
#include <cstddef>
#include <functional>

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/factory.hpp"

namespace {
auto& get_factories() {
    static std::unordered_map<Device, TensorImplFactory> factories;
    return factories;
}
auto& get_initializers() {
    static std::unordered_map<Device, BackendInitializer> initializers;
    return initializers;
}
}  // namespace

void register_tensor_impl(Device device, TensorImplFactory factory) {
    try {
        get_factories()[device] = std::move(factory);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        throw std::runtime_error("register_tensor_impl: failed to register backend");
    }
}

void register_backend_initializer(Device device, BackendInitializer init) {
    get_initializers()[device] = std::move(init);
}

namespace {
// Invoke the lazy initializer (if any) once, then return the factory for
// `device` or nullptr when the backend is not compiled in / unavailable.
TensorImplFactory* find_factory(Device device) {
    auto& factories = get_factories();
    auto it = factories.find(device);
    if (it != factories.end()) return &it->second;

    auto& initializers = get_initializers();
    auto init = initializers.find(device);
    if (init == initializers.end()) return nullptr;
    // Run the initializer exactly once; erase it so a backend that fails to
    // come up is not retried on every allocation.
    BackendInitializer fn = std::move(init->second);
    initializers.erase(init);
    fn();

    it = factories.find(device);
    return it == factories.end() ? nullptr : &it->second;
}
}  // namespace

std::shared_ptr<TensorImpl> create_tensor_impl(size_t numel, DataType dtype, Device device) {
    if (numel <= 0) {
        throw std::invalid_argument("numel must be bigger than 0");
    }
    auto* factory = find_factory(device);
    if (!factory) {
        throw std::runtime_error(
            std::format("No tensor implementation registered for device: {}", device_to_string(device)));
    }
    return (*factory)(nullptr, numel, dtype);
}

std::shared_ptr<TensorImpl> create_tensor_impl(void* ptr, size_t numel, DataType dtype, Device device) {
    if (ptr == nullptr) {
        throw std::runtime_error("create_tensor_impl: ptr is null");
    }
    if (numel <= 0) {
        throw std::invalid_argument("numel must be bigger than 0");
    }
    auto* factory = find_factory(device);
    if (!factory) {
        throw std::runtime_error(
            std::format("No tensor implementation registered for device: {}", device_to_string(device)));
    }
    return (*factory)(ptr, numel, dtype);
}
