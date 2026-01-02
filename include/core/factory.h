#pragma once
#include <memory>
#include <functional>
#include <unordered_map>
#include <iostream>
#include <format>

#include "tensor.h"

using TensorImplFactory = std::function<std::shared_ptr<TensorImpl>(void* ptr,size_t numel, via::DataType)>;

void register_tensor_impl(via::Device device, TensorImplFactory factory);

std::shared_ptr<TensorImpl> create_tensor_impl(size_t numel, via::DataType dtype, via::Device device);
std::shared_ptr<TensorImpl> create_tensor_impl(void*ptr,size_t numel, via::DataType dtype, via::Device device);

void copy_device_to_host(std::shared_ptr<TensorImpl> src,std::shared_ptr<TensorImpl> dst,via::DataType dtype);
void copy_host_to_device(std::shared_ptr<TensorImpl> src,std::shared_ptr<TensorImpl> dst,via::DataType dtype);