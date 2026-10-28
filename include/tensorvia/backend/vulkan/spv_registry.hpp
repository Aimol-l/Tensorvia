#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace vkspv {

// 编译进产物的 SPIR-V 的一处视图，不是副本。
//
// 表本身由 build.mcpp 生成（每个 shader 一条），里面是指向静态数组的指针：
// 这里返回视图就够，vkCreateShaderModule 当场用完，没有理由为每次建 pipeline
// 复制几十 KB 字节。
struct spv_view {
    const std::uint32_t* data;
    std::size_t          words;
};

// key 是 make_pipeline_key(OpType, DataType) 的产物（"add_float32"）。
// 找不到就抛 std::runtime_error("SPIR-V not found: <key>")。
spv_view get_spv_code(std::string_view key);

}  // namespace vkspv
