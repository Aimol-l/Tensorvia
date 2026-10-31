#pragma once

#include <cstddef>
#include <memory>

// 只有声明：buffer 管理与拷贝路径定义在 src/backend/cuda/device/runtime.cu。
// data() 返回真正的设备指针（Vulkan 那边返回的是 buffer 句柄），但在宿主上仍然
// 不能解引用：请用 copy_to_host / Tensor::to_host。
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/context.hpp"
#include "tensorvia/backend/cuda/cuda_context.hpp"

class CUDATensor : public TensorImpl {
private:
    void*                         m_data = nullptr;
    size_t                        m_numel = 0;
    DataType                      m_dtype = DataType::FLOAT32;
    // buffer 是在哪条流上分配的；释放记录也记在那条流上，因此复用该块的另一个
    // 线程只需等待这条流，而不需要设备级屏障。
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
