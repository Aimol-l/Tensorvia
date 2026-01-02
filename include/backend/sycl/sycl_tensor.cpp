#include <print>
#include "sycl_tensor.h"
#include "core/types.h"
#include "ops/repack.h"

using namespace via;

void SYCLTensor::init(void *ptr,size_t numel, DataType dtype, std::shared_ptr<SYCLContext> context){
    m_dtype = dtype;
    m_numel = numel;
    m_context = context;
    auto& queue = m_context->get_queue();
    void* allocated_ptr = nullptr;
    size_t total_bytes = m_numel * calc_dtype_size(m_dtype);
    switch (dtype) {
        case DataType::INT8:    allocated_ptr = sycl::malloc_device<int8_t>(m_numel, queue); break;
        case DataType::INT16:   allocated_ptr = sycl::malloc_device<int16_t>(m_numel, queue); break;
        case DataType::INT32:   allocated_ptr = sycl::malloc_device<int32_t>(m_numel, queue); break;
        case DataType::INT64:   allocated_ptr = sycl::malloc_device<int64_t>(m_numel, queue); break;
        case DataType::FLOAT16: allocated_ptr = sycl::malloc_device<float16>(m_numel, queue); break;
        case DataType::FLOAT32: allocated_ptr = sycl::malloc_device<float32>(m_numel, queue); break;
        case DataType::FLOAT64: allocated_ptr = sycl::malloc_device<float64>(m_numel, queue); break;
        case DataType::BFLOAT16:allocated_ptr = sycl::malloc_device<bfloat16>(m_numel, queue); break;
        default: throw std::runtime_error("Unsupported data type");
    }
    queue.wait();
    if(ptr != nullptr) 
        queue.memcpy(allocated_ptr, ptr, total_bytes).wait();
    m_data.reset(allocated_ptr);
}

SYCLTensor::SYCLTensor(size_t numel, via::DataType dtype,std::shared_ptr<SYCLContext> context):
    m_data(nullptr,FreeDeleter(context->get_queue(), dtype)){
    // LOG_INFO("Create SYCL tensor");
    this->init(nullptr, numel, dtype, context);
}
SYCLTensor::SYCLTensor(void* ptr,size_t numel, via::DataType dtype,std::shared_ptr<SYCLContext> context):
    m_data(nullptr,FreeDeleter(context->get_queue(), dtype))
{
    // LOG_INFO("Create SYCL tensor from raw pointer");
    this->init(ptr, numel, dtype, context);
}

std::unique_ptr<TensorImpl> SYCLTensor::clone() const
{
    auto& queue = m_context->get_queue();
    auto cloned = std::make_unique<SYCLTensor>(m_numel, m_dtype,m_context);
    size_t bytes = this->m_numel * calc_dtype_size(this->m_dtype);
    queue.memcpy(cloned->m_data.get(), m_data.get(), bytes).wait();
    return cloned;
}
/**
 * SYCL 设备内存 → 另一块 SYCL 设备内存
 * @param dst 目标设备指针（必须为 SYCL 分配的 USM 设备内存）
 */
void SYCLTensor::copy_to(std::shared_ptr<TensorImpl> dst) const {
    if (!dst->data()) throw std::invalid_argument("Destination pointer is null");
    if (!m_data.get()) throw std::runtime_error("Source tensor data is null");
    const size_t bytes = m_numel * calc_dtype_size(m_dtype);
    auto& queue = m_context->get_queue();
    auto dst_ptr_type = sycl::get_pointer_type(dst->data(), queue.get_context());
    // 检查目标指针类型
    if (dst_ptr_type != sycl::usm::alloc::device) {
        throw std::runtime_error("Destination pointer is not SYCL device memory");
    }
    queue.memcpy(dst->data(), m_data.get(), bytes).wait();
}
std::shared_ptr<ContextImpl> SYCLTensor::context() const{
    return m_context;
}
void* SYCLTensor::data() { return m_data.get(); }
const void* SYCLTensor::data() const  { return m_data.get();}
size_t SYCLTensor::numel() const  { return m_numel; }

std::unique_ptr<TensorImpl> SYCLTensor::clone_as_contiguous(const Metadata& meta) const{
    auto cloned = std::make_unique<SYCLTensor>(meta.numel, this->m_dtype, this->m_context);
    RepackImpl<via::Device::SYCL>::execute(meta,this->m_data.get(),cloned->m_data.get(),this->m_context);
    return cloned;
}