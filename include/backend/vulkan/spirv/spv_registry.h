#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
namespace vkspv {
    std::vector<uint32_t> get_spv_code(const std::string& key);
}
