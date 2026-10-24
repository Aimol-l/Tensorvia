#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/context.hpp"
#include "interface/cpu_context.hpp"

namespace cpu_tensor_detail {
// Stateful deleter: carries the byte size so the pool knows which size-class
// freelist to return the block to. operator() body lives in cpu_tensor.cpp.
struct PoolDeleter {
    size_t bytes = 0;
    PoolDeleter() = default;
    explicit PoolDeleter(size_t b) noexcept : bytes(b) {}
    void operator()(void* p) const noexcept;
};
}  // namespace cpu_tensor_detail

class CPUTensor : public TensorImpl {
private:
    size_t                                                m_numel;
    DataType                                              m_dtype;
    CPUContext*                                           m_ctx_raw;   // 非所有：CPURegistrar 单例持有
    std::unique_ptr<void, cpu_tensor_detail::PoolDeleter> m_data;

public:
    ~CPUTensor() override = default;

    // 入参仍接收 shared_ptr 保持 API 兼容；内部只取裸指针，避免每次构造/析构一对原子操作。
    CPUTensor(std::shared_ptr<CPUContext> ctx, size_t numel, DataType dtype);
    CPUTensor(std::shared_ptr<CPUContext> ctx, void* ptr, size_t numel, DataType dtype);
    void init(void* ptr, size_t numel, DataType dtype);

    void* data() override { return m_data.get(); }
    const void* data() const override { return m_data.get(); }
    size_t numel() const override;
    void copy_to(std::shared_ptr<TensorImpl> dst) const override;
    std::unique_ptr<TensorImpl> clone() const override;
    std::shared_ptr<ContextImpl> context() const override;
    std::unique_ptr<TensorImpl> clone_as_contiguous(const Metadata&) const override;
};
