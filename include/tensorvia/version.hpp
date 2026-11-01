#pragma once

#include <string_view>

// 使用预编译 `mcpp pack` 包的一方会自己编译这个接口单元，拿不到库构建时的 -D
// 宏，所以回退值必须就是真实版本（与 mcpp.toml / CMakeLists.txt 保持一致）。
#ifndef TENSORVIA_VERSION
    #define TENSORVIA_VERSION "0.3.0"
#endif


namespace tensorvia {
inline constexpr std::string_view version = TENSORVIA_VERSION;
}
