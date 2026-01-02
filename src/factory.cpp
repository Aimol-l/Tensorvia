#include <stdexcept>
#include "context.h"
#include "core/factory.h"
using namespace via;

// 替换全局 factories 为函数返回引用
static auto& get_factories() {
    static std::unordered_map<Device, TensorImplFactory> factories;
    return factories;
}

void register_tensor_impl(Device device, TensorImplFactory factory) {
    try {
        get_factories()[device] = std::move(factory);  // ← 使用 get_factories()
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        throw std::runtime_error("register_tensor_impl: failed to register backend");
    }
    std::cout << "Register backend " << device_to_string(device) << std::endl;
}

std::shared_ptr<TensorImpl> create_tensor_impl(size_t numel, DataType dtype, Device device) {
    if (numel <= 0) {
        throw std::invalid_argument("numel must be bigger than 0");
    }
    auto& factories = get_factories();  // ←
    auto it = factories.find(device);
    if (it == factories.end()) {
        throw std::runtime_error(std::format("No tensor implementation registered for device: {}", device_to_string(device)));
    }
    return it->second(nullptr, numel, dtype);
}

std::shared_ptr<TensorImpl> create_tensor_impl(void* ptr, size_t numel, DataType dtype, Device device) {
    if (ptr == nullptr) {
        throw std::runtime_error("create_tensor_impl: ptr is null");
    }
    if (numel <= 0) {
        throw std::invalid_argument("numel must be bigger than 0");
    }
    auto& factories = get_factories();  // ←
    auto it = factories.find(device);
    if (it == factories.end()) {
        throw std::runtime_error(std::format("No tensor implementation registered for device: {}", device_to_string(device)));
    }
    return it->second(ptr, numel, dtype);
}