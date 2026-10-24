#pragma once

#include <string_view>

// Consumers of a prebuilt `mcpp pack` package compile this interface unit
// themselves and do not inherit the library's -D flags, so the fallback must
// carry the real version (keep it in sync with mcpp.toml / CMakeLists.txt).
#ifndef TENSORVIA_VERSION
    #define TENSORVIA_VERSION "0.3.0"
#endif


namespace tensorvia {
inline constexpr std::string_view version = TENSORVIA_VERSION;
}
