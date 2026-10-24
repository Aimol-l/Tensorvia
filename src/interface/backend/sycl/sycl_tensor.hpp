#pragma once

#include <cstddef>
#include <memory>

// Declarations only: buffer management and the copy paths are defined by
// src/backend/sycl/host/runtime.cpp. data() returns a real USM device pointer,
// but it must still not be dereferenced on the host: use copy_to_host /
// Tensor::to_host.
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/context.hpp"
#include "interface/backend/sycl/sycl_context.hpp"

class SYCLTensor : public TensorImpl {
private:
    void*                        m_data = nullptr;
    size_t                       m_numel = 0;
    DataType                     m_dtype = DataType::FLOAT32;
    std::shared_ptr<SYCLContext> m_context;

    void init(void* ptr, size_t numel, DataType dtype,
              std::shared_ptr<SYCLContext> context);

public:
    SYCLTensor(size_t numel, DataType dtype, std::shared_ptr<SYCLContext> context);
    SYCLTensor(void* ptr, size_t numel, DataType dtype,
               std::shared_ptr<SYCLContext> context);
    ~SYCLTensor() override;

    SYCLTensor(const SYCLTensor&) = delete;
    SYCLTensor& operator=(const SYCLTensor&) = delete;

    void*       data() override { return m_data; }
    const void* data() const override { return m_data; }
    size_t      numel() const override { return m_numel; }
    DataType    dtype() const { return m_dtype; }

    void copy_to(std::shared_ptr<TensorImpl> dst) const override;
    std::unique_ptr<TensorImpl> clone() const override;
    std::shared_ptr<ContextImpl> context() const override;
    std::unique_ptr<TensorImpl> clone_as_contiguous(const Metadata& meta) const override;

    // Not optional: the base implementation memcpy's, which would dereference a
    // device pointer on the host.
    void copy_to_host(void* dst, DataType dtype) const override;
    void copy_from_host(const void* src, DataType dtype) override;
};
