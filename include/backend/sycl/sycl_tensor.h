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
    DataType m_dtype;
    sycl::queue& m_queue;  // 需要保存 queue 用于释放内存
    explicit FreeDeleter(sycl::queue& q,DataType dtype) : m_dtype(dtype),m_queue(q) {}
    void operator()(void* ptr) {
        if (ptr == nullptr) return;
        switch (m_dtype) {
            case DataType::INT8:    sycl::free(static_cast<int8_t*>(ptr),  m_queue); break;
            case DataType::INT16:   sycl::free(static_cast<int16_t*>(ptr), m_queue); break;
            case DataType::INT32:   sycl::free(static_cast<int32_t*>(ptr), m_queue); break;
            case DataType::INT64:   sycl::free(static_cast<int64_t*>(ptr), m_queue); break;
            case DataType::FLOAT16: sycl::free(static_cast<float16*>(ptr), m_queue); break;
            case DataType::FLOAT32: sycl::free(static_cast<float32*>(ptr), m_queue); break;
            case DataType::FLOAT64: sycl::free(static_cast<float64*>(ptr), m_queue); break;
            case DataType::BFLOAT16:sycl::free(static_cast<bfloat16*>(ptr),m_queue); break;
            default: throw std::runtime_error("Unsupported data type");
        } 
    }
};
private:
    size_t m_numel;
    DataType m_dtype;
    std::vector<int> m_shape;
    std::shared_ptr<SYCLContext> m_context;
    std::unique_ptr<void, FreeDeleter> m_data;
public:
    ~SYCLTensor(){};
    SYCLTensor(std::vector<int> shape, DataType dtype,std::shared_ptr<SYCLContext> context);
    SYCLTensor(void* ptr,std::vector<int> shape, DataType dtype,std::shared_ptr<SYCLContext> context);

    void init(void* ptr,std::vector<int> shape, DataType dtype,std::shared_ptr<SYCLContext> context);

    void* data() override;
    size_t numel() const override;
    const void* data() const override;
    void copy_to(void* dst) const override;           // 同设备
    std::unique_ptr<TensorImpl> clone() const override;
    std::shared_ptr<ContextImpl> context() const override;
    void reshape(std::vector<int>& newshape)override;
    void reshape(std::initializer_list<int> newshape)override;
};
