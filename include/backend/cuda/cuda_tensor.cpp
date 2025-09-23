#include "cuda_tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <numeric>


void CUDATensor::init(void* ptr, std::vector<int64_t> shape, DataType dtype, std::shared_ptr<CUDAContext> context) {
    m_shape = std::move(shape);
    m_dtype = dtype;
    m_context = context;
    m_numel = calc_numel(m_shape);
    void* allocated_ptr = nullptr;
    size_t total_bytes = m_numel * calc_dtype_size(m_dtype);
    cudaError_t err = cudaMalloc(&allocated_ptr, total_bytes);

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed: " + std::string(cudaGetErrorString(err)));
    }
    // 显存创建完成后，将ptr的数据拷贝过去
    if(ptr != nullptr){
        err = cudaMemcpy(allocated_ptr, ptr, total_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(allocated_ptr);  // Clean up allocated memory before throwing
            throw std::runtime_error("CUDA memory copy failed: " + std::string(cudaGetErrorString(err)));
        }
    }
    m_data.reset(allocated_ptr);
}

CUDATensor::CUDATensor(std::vector<int64_t> shape, DataType dtype, std::shared_ptr<CUDAContext> context):
    m_data(nullptr,FreeDeleter(dtype)){
    this->init(nullptr, shape, dtype, context);
}

CUDATensor::CUDATensor(void* ptr, std::vector<int64_t> shape, DataType dtype, std::shared_ptr<CUDAContext> context):
    m_data(nullptr,FreeDeleter(dtype)){
    this->init(nullptr, shape, dtype, context);
}

std::unique_ptr<TensorImpl> CUDATensor::clone() const {
    auto cloned = std::make_unique<CUDATensor>(m_shape, m_dtype, m_context);
    size_t size_bytes = m_numel * calc_dtype_size(m_dtype);
    cudaMemcpy(cloned->m_data.get(), m_data.get(), size_bytes, cudaMemcpyDeviceToDevice);
    return cloned;
}
//  cuda设备内存 -> cuda设备内存
void CUDATensor::copy_to(void* dst) const {
    if(dst == nullptr)
        throw std::runtime_error("[ERROR] CUDATensor::copy_to: dst is nullptr");

    if(!m_data.get())
        throw std::runtime_error("[ERROR] CUDATensor::copy_to: m_data is nullptr");
    
    size_t size_bytes = m_numel * calc_dtype_size(m_dtype);

    cudaMemcpy(dst, m_data.get(), size_bytes, cudaMemcpyDeviceToDevice);
}
std::shared_ptr<ContextImpl> CUDATensor::context() const{
    return m_context;
}
void CUDATensor::reshape(std::vector<int64_t>& newshape) {
    m_shape.clear();
    m_shape.assign(newshape.begin(), newshape.end());
}
void CUDATensor::reshape(std::initializer_list<int64_t> newshape) {
    m_shape.clear();
    m_shape.assign(newshape.begin(), newshape.end());
}