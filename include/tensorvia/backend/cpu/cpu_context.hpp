#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "tensorvia/context.hpp"

class CPUContext : public ContextImpl {
private:
    int         m_num_cores;
    std::string m_model_name;
    size_t      m_memory_total_bytes;

public:
    CPUContext();
    ~CPUContext() override = default;

    void* ctx_raw_ptr() override { return nullptr; }

    int         num_cores()          const { return m_num_cores; }
    const std::string& model_name() const { return m_model_name; }
    size_t      memory_total_bytes() const { return m_memory_total_bytes; }
};
