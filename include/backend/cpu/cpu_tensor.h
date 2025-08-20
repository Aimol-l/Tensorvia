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
    std::vector<int> m_shape;
    std::unique_ptr<void, FreeDeleter> m_data;

public:
    ~CPUTensor(){};

    CPUTensor(std::vector<int> shape, DataType dtype);
    CPUTensor(void* ptr,std::vector<int> shape, DataType dtype);
    void init(void* ptr,std::vector<int> shape, DataType dtype);
    
    void* data() override {return m_data.get(); }
    const void* data() const override{ return m_data.get();}
    size_t numel() const override;
    void copy_to(void* dst) const override;           // 同设备
    std::unique_ptr<TensorImpl> clone() const override;
    std::shared_ptr<ContextImpl> context() const override;
    void reshape(std::vector<int>& newshape)override;
    void reshape(std::initializer_list<int> newshape)override;

};
