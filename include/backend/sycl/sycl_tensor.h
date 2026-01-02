#pragma once

#include <memory>
#include <cstdlib>
#include <cstring>
#include <print>
#include "core/tensor.h"
#include <sycl/sycl.hpp>
#include "sycl_context.h"

class SYCLTensor : public TensorImpl {
struct FreeDeleter {
    via::DataType m_dtype;
    sycl::queue& m_queue;  // 需要保存 queue 用于释放内存
    explicit FreeDeleter(sycl::queue& q,via::DataType dtype) : m_dtype(dtype),m_queue(q) {}
    void operator()(void* ptr) {
        if (ptr == nullptr) return;
        switch (m_dtype) {
            case via::DataType::INT8:    sycl::free(static_cast<int8_t*>(ptr),  m_queue); break;
            case via::DataType::INT16:   sycl::free(static_cast<int16_t*>(ptr), m_queue); break;
            case via::DataType::INT32:   sycl::free(static_cast<int32_t*>(ptr), m_queue); break;
            case via::DataType::INT64:   sycl::free(static_cast<int64_t*>(ptr), m_queue); break;
            case via::DataType::FLOAT16: sycl::free(static_cast<float16*>(ptr), m_queue); break;
            case via::DataType::FLOAT32: sycl::free(static_cast<float32*>(ptr), m_queue); break;
            case via::DataType::FLOAT64: sycl::free(static_cast<float64*>(ptr), m_queue); break;
            case via::DataType::BFLOAT16:sycl::free(static_cast<bfloat16*>(ptr),m_queue); break;
            default: throw std::runtime_error("Unsupported data type");
        } 
    }
};
private:
    size_t m_numel;
    via::DataType m_dtype;
    std::shared_ptr<SYCLContext> m_context;
    std::unique_ptr<void, FreeDeleter> m_data;
public:
    ~SYCLTensor(){};
    SYCLTensor(size_t numel, via::DataType dtype,std::shared_ptr<SYCLContext> context);
    SYCLTensor(void* ptr,size_t numel, via::DataType dtype,std::shared_ptr<SYCLContext> context);

    void init(void* ptr,size_t numel, via::DataType dtype,std::shared_ptr<SYCLContext> context);

    void* data() override;
    size_t numel() const override;
    const void* data() const override;
    void copy_to(std::shared_ptr<TensorImpl> dst) const override;           // 同设备
    std::unique_ptr<TensorImpl> clone() const override;
    std::shared_ptr<ContextImpl> context() const override;
    std::unique_ptr<TensorImpl> clone_as_contiguous(const Metadata&) const override;

};
