#include <typeinfo> 
#include "cpu_tensor.h"
#include "cuda_tensor.h"
#include "core/factory.h"


// cuda --> cpu
void copy_device_to_host(std::shared_ptr<TensorImpl> src,std::shared_ptr<TensorImpl> dst,DataType dtype) {

    auto dst_impl = std::dynamic_pointer_cast<CPUTensor>(dst); // cpu
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(src); // cuda

    size_t num_bytes = src_impl->numel() * calc_dtype_size(dtype);
    cudaMemcpy(dst_impl->data(), src_impl->data(), num_bytes, cudaMemcpyDeviceToHost);
}
// cpu --> cuda
void copy_host_to_device(std::shared_ptr<TensorImpl> src,std::shared_ptr<TensorImpl> dst,DataType dtype) {

    auto src_impl =  std::dynamic_pointer_cast<CPUTensor>(src); // cpu
    auto dst_impl = std::dynamic_pointer_cast<CUDATensor>(dst); // cuda

    const size_t bytes = src_impl->numel() * calc_dtype_size(dtype);

    cudaMemcpy(dst_impl->data(), src_impl->data(), bytes, cudaMemcpyHostToDevice);
}
