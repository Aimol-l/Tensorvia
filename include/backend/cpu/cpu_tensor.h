#pragma once
#include <cstdlib>
#include <cstring>
#include <memory>
#include <print>
#include "core/tensor.h"
#include "core/types.h"
#include "core/context.h"

class CPUTensor : public TensorImpl {
struct FreeDeleter {
    void operator()(void* ptr) const {
        std::free(ptr);
    }
};
private:
    size_t m_numel;
    DataType m_dtype;
    std::unique_ptr<void, FreeDeleter> m_data;

public:
    ~CPUTensor(){};

    CPUTensor(size_t numel, DataType dtype);
    CPUTensor(void* ptr,size_t numel, DataType dtype);
    void init(void* ptr,size_t numel, DataType dtype);
    
    void* data() override {return m_data.get(); }
    const void* data() const override{ return m_data.get();}
    size_t numel() const override;
    void copy_to(TensorImpl& dst) const override;           // 同设备
    std::unique_ptr<TensorImpl> clone() const override;
    std::shared_ptr<ContextImpl> context() const override;
    std::unique_ptr<TensorImpl> clone_as_contiguous(const Metadata&) const override;

};
