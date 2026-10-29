#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace vkspv {

struct spv_view {
    const std::uint32_t* data;
    std::size_t          words;
};

spv_view get_spv_code(std::string_view key);

}  // namespace vkspv
