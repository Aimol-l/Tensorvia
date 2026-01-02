#pragma once

class ContextImpl {
public:
    virtual ~ContextImpl() = default;
    virtual void* ctx_raw_ptr() = 0;  // 返回后端特定的上下文指针（如 sycl::queue*, CUDA stream 等）
};
