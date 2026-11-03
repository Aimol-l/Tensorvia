#pragma once

#include <string_view>

#ifndef TENSORVIA_VERSION
    #define TENSORVIA_VERSION "0.3.0"
#endif


namespace tensorvia {
inline constexpr std::string_view version = TENSORVIA_VERSION;
}
