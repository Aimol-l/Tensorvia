#pragma once

#include <cstddef>
#include <memory>

#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/context.hpp"
#include "tensorvia/backend/cuda/cuda_context.hpp"

class CUDATensor : public TensorImpl {
private:
    void*                         m_data = nullptr;
    size_t                        m_numel = 0;
    DataType                      m_dtype = DataType::FLOAT32;
    CUDAContext::CUDAStream       m_stream;
    std::shared_ptr<CUDAContext>  m_context;

    void init(void* ptr, size_t numel, DataType dtype,
              std::shared_ptr<CUDAContext> context);

public:
    CUDATensor(size_t numel, DataType dtype, std::shared_ptr<CUDAContext> context);
    CUDATensor(void* ptr, size_t numel, DataType dtype,
               std::shared_ptr<CUDAContext> context);
    ~CUDATensor() override;

    CUDATensor(const CUDATensor&) = delete;
    CUDATensor& operator=(const CUDATensor&) = delete;

    void*       data() override { return m_data; }
    const void* data() const override { return m_data; }
    size_t      numel() const override { return m_numel; }
    DataType    dtype() const { return m_dtype; }

    void copy_to(std::shared_ptr<TensorImpl> dst) const override;
    std::unique_ptr<TensorImpl> clone() const override;
    std::shared_ptr<ContextImpl> context() const override;
    std::unique_ptr<TensorImpl> clone_as_contiguous(const Metadata& meta) const override;

    // 不能沿用默认实现：基类实现用 memcpy，会在宿主上解引用设备指针。
    void copy_to_host(void* dst, DataType dtype) const override;
    void copy_from_host(const void* src, DataType dtype) override;
};
