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
        DataType m_dtype;
        FreeDeleter(DataType dtype): m_dtype(dtype){}
        void operator()(void* ptr) const {
            if (ptr == nullptr) return;
            switch (m_dtype) {
                case DataType::INT8:    cudaFree(static_cast<int8_t*>(ptr)); break;
                case DataType::INT16:   cudaFree(static_cast<int16_t*>(ptr)); break;
                case DataType::INT32:   cudaFree(static_cast<int32_t*>(ptr)); break;
                case DataType::INT64:   cudaFree(static_cast<int64_t*>(ptr)); break;
                case DataType::FLOAT16: cudaFree(static_cast<float16*>(ptr)); break;
                case DataType::FLOAT32: cudaFree(static_cast<float32*>(ptr)); break;
                case DataType::FLOAT64: cudaFree(static_cast<float64*>(ptr)); break;
                case DataType::BFLOAT16:cudaFree(static_cast<bfloat16*>(ptr)); break;
                default: throw std::runtime_error("Unsupported data type for CUDATensor");
            }
        }
        
    };
private:
    size_t m_numel;
    DataType m_dtype;
    std::vector<int64_t> m_shape;
    std::shared_ptr<CUDAContext> m_context;
    std::unique_ptr<void, FreeDeleter> m_data;

public:
    ~CUDATensor() = default;
    CUDATensor(std::vector<int64_t> shape, DataType dtype, std::shared_ptr<CUDAContext> context);
    CUDATensor(void* ptr, std::vector<int64_t> shape, DataType dtype, std::shared_ptr<CUDAContext> context);
    
    void init(void* ptr, std::vector<int64_t> shape, DataType dtype, std::shared_ptr<CUDAContext> context);

    void* data() override {return m_data.get(); }
    const void* data() const override{ return m_data.get();}
    size_t numel() const override{return m_numel;}

    void copy_to(void* dst) const override;
    std::unique_ptr<TensorImpl> clone() const override;
    std::shared_ptr<ContextImpl> context() const override;
    void reshape(std::vector<int64_t>& newshape) override;
    void reshape(std::initializer_list<int64_t> newshape) override;
};