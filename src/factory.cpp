#include <stdexcept>
#include "context.h"
#include "core/factory.h"

// using TensorImplFactory = std::function<std::shared_ptr<TensorImpl>(void* ptr,std::vector<int> shape, DataType)>;
// static std::unordered_map<Device, TensorImplFactory> factories;

void register_tensor_impl(Device device, TensorImplFactory factory) {
    if(device == Device::CPU){
        std::cout<<("Register backend CPU")<<std::endl;
    }else if(device == Device::CUDA){
        std::cout<<("Register backend CUDA")<<std::endl;
    }else if(device == Device::SYCL){
        std::cout<<("Register backend SYCL")<<std::endl;
    }else if(device == Device::VULKAN){
        std::cout<<("Register backend VULKAN")<<std::endl;
    }else{
        std::cout<<("Register backend unknown")<<std::endl;
    }
    factories[device] = std::move(factory);
}
std::shared_ptr<TensorImpl> create_tensor_impl(std::vector<int> shape, DataType dtype, Device device) {
    auto it = factories.find(device);

    // shape必须存在元素,且都大于0
    if (shape.empty()) throw std::invalid_argument("shape must not be empty");
    for (auto i : shape) {
        if (i <= 0) throw std::invalid_argument("shape must be positive");
    }
    if (it == factories.end()) {
        throw std::runtime_error(std::format("No tensor implementation registered for this device: {}",device_to_string(device)));
    }
    return it->second(nullptr,shape, dtype);
}
std::shared_ptr<TensorImpl> create_tensor_impl(void*ptr,std::vector<int> shape, DataType dtype, Device device) {
    if (ptr == nullptr) 
        throw std::runtime_error("create_tensor_impl: ptr is null");
    auto it = factories.find(device);
    if (shape.empty()) 
        throw std::invalid_argument("shape must not be empty");
    for (auto i : shape) {
        if (i <= 0) throw std::invalid_argument("shape must be positive");
    }
    if (it == factories.end()) {
        throw std::runtime_error(std::format("No tensor implementation registered for this device:{}",device_to_string(device)));
    }
    return it->second(ptr,shape, dtype);
}