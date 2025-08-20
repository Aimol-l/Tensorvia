
#include <typeinfo> 
#include <sycl/sycl.hpp>
#include <print>
#include "sycl_tensor.h"
#include "cpu_tensor.h"
#include "core/factory.h"


// sycl --> cpu
void copy_device_to_host(std::shared_ptr<TensorImpl> src,std::shared_ptr<TensorImpl> dst,DataType dtype) {
    auto dst_impl = std::dynamic_pointer_cast<CPUTensor>(dst); // cpu
    auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(src); // sycl
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src->context());
    auto& queue_ = ctx_impl->get_queue(); 
    const size_t bytes = src->numel() * calc_dtype_size(dtype);
    queue_.memcpy(dst_impl->data(),src_impl->data(),bytes).wait();
    // LOG_INFO("SYCL ---> CPU");
}
// cpu --> sycl
void copy_host_to_device(std::shared_ptr<TensorImpl> src,std::shared_ptr<TensorImpl> dst,DataType dtype) {
    auto src_impl =  std::dynamic_pointer_cast<CPUTensor>(src); // cpu
    auto dst_impl = std::dynamic_pointer_cast<SYCLTensor>(dst); // sycl

    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(dst->context());

    auto& queue_ = ctx_impl->get_queue(); 
    const size_t bytes = src_impl->numel() * calc_dtype_size(dtype);
    queue_.memcpy(dst_impl->data(),src_impl->data(),bytes).wait();
    // LOG_INFO("CPU ---> SYCL");
}
