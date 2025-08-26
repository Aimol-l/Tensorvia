#pragma once
#include <memory>
#include <functional>

#include "tensor.h"

using TensorImplFactory = std::function<std::shared_ptr<TensorImpl>(void* ptr,std::vector<int> shape, DataType)>;

static std::unordered_map<Device, TensorImplFactory> factories;
void register_tensor_impl(Device device, TensorImplFactory factory);

std::shared_ptr<TensorImpl> create_tensor_impl(std::vector<int> shape, DataType dtype, Device device);
std::shared_ptr<TensorImpl> create_tensor_impl(void*ptr,std::vector<int> shape, DataType dtype, Device device);

void copy_device_to_host(std::shared_ptr<TensorImpl> src,std::shared_ptr<TensorImpl> dst,DataType dtype);
void copy_host_to_device(std::shared_ptr<TensorImpl> src,std::shared_ptr<TensorImpl> dst,DataType dtype);