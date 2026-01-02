#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include <print>
#include "core/tensor.h"
#include "core/types.h"
#include "core/context.h"
#include "cuda_context.h" 

class CUDATensor : public TensorImpl {
private:
    struct FreeDeleter {
        via::DataType m_dtype;
        FreeDeleter(via::DataType dtype): m_dtype(dtype){}
        void operator()(void* ptr) const {
            if (ptr == nullptr) return;
            switch (m_dtype) {
                case via::DataType::INT8:    cudaFree(static_cast<int8_t*>(ptr)); break;
                case via::DataType::INT16:   cudaFree(static_cast<int16_t*>(ptr)); break;
                case via::DataType::INT32:   cudaFree(static_cast<int32_t*>(ptr)); break;
                case via::DataType::INT64:   cudaFree(static_cast<int64_t*>(ptr)); break;
                case via::DataType::FLOAT16: cudaFree(static_cast<float16*>(ptr)); break;
                case via::DataType::FLOAT32: cudaFree(static_cast<float32*>(ptr)); break;
                case via::DataType::FLOAT64: cudaFree(static_cast<float64*>(ptr)); break;
                case via::DataType::BFLOAT16:cudaFree(static_cast<bfloat16*>(ptr)); break;
                default: throw std::runtime_error("Unsupported data type for CUDATensor");
            }
        }
        
    };
private:
    size_t m_numel;
    via::DataType m_dtype;
    std::shared_ptr<CUDAContext> m_context;
    std::unique_ptr<void, FreeDeleter> m_data;

public:
    ~CUDATensor() = default;
    CUDATensor(size_t numel, via::DataType dtype, std::shared_ptr<CUDAContext> context);
    CUDATensor(void* ptr, size_t numel, via::DataType dtype, std::shared_ptr<CUDAContext> context);
    
    void init(void* ptr, size_t numel, via::DataType dtype, std::shared_ptr<CUDAContext> context);
    void* data() override {return m_data.get(); }
    size_t numel() const override{return m_numel;}
    const void* data() const override{ return m_data.get();}
    void copy_to(std::shared_ptr<TensorImpl> dst) const override;
    std::unique_ptr<TensorImpl> clone() const override;
    std::shared_ptr<ContextImpl> context() const override;
    std::unique_ptr<TensorImpl> clone_as_contiguous(const Metadata&) const override;

};