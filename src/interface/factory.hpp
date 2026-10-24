#pragma once

#include <cstddef>
#include <functional>
#include <memory>

#include "interface/types.hpp"
#include "interface/tensor.hpp"

using TensorImplFactory =
    std::function<std::shared_ptr<TensorImpl>(void* ptr, size_t numel, DataType)>;
using BackendInitializer = std::function<void()>;

void register_tensor_impl(Device device, TensorImplFactory factory);

// Registers a callback that is invoked lazily, the first time a tensor on
// `device` is requested without an implementation having been registered.
// This keeps optional backends (Vulkan) completely dormant in CPU-only runs.
void register_backend_initializer(Device device, BackendInitializer init);

std::shared_ptr<TensorImpl> create_tensor_impl(size_t numel,
                                                      DataType dtype,
                                                      Device device);
std::shared_ptr<TensorImpl> create_tensor_impl(void* ptr,
                                                     size_t numel,
                                                     DataType dtype,
                                                     Device device);

void copy_device_to_host(std::shared_ptr<TensorImpl> src,
                               std::shared_ptr<TensorImpl> dst,
                               DataType dtype);
void copy_host_to_device(std::shared_ptr<TensorImpl> src,
                               std::shared_ptr<TensorImpl> dst,
                               DataType dtype);
