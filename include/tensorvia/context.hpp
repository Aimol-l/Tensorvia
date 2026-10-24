#pragma once

#include <memory>


class ContextImpl {
public:
    virtual ~ContextImpl() = default;
    virtual void* ctx_raw_ptr() = 0;
};
