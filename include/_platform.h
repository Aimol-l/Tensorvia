#pragma once

#ifndef TENSORVIA_PLATFORM_H
#define TENSORVIA_PLATFORM_H

#if !defined(RESTRICT)
    #if defined(_MSC_VER)
        #define RESTRICT __restrict
    #elif defined(__GNUC__) || defined(__clang__)
        #define RESTRICT __restrict__
    #else
        #define RESTRICT
    #endif
#endif

// C++23 的 PDP14 废弃了 std::float_denorm_style 和它的枚举值，但三个标准库实现
// （MSVC STL、libc++、libstdc++）都仍然用这个类型声明 numeric_limits<T>::has_denorm，
// 所以可移植的特化只能继续写它。MSVC STL 和 libc++ 会对这种写法告警
// （MSVC: STL4042），这里按 MSVC 自己 <limits> 里的做法，把告警就地压掉：
//
//     TENSORVIA_DENORM_DEPRECATION_PUSH
//     static constexpr float_denorm_style has_denorm = denorm_present;
//     TENSORVIA_DENORM_DEPRECATION_POP
//
// 用点压而不是在包含 <limits> 之前定义 _SILENCE_CXX23_DENORM_DEPRECATION_WARNING：
// 那个宏只对 MSVC 有效，而且要求在本 TU 第一次包含 <limits> 之前就定义，
// 消费者先包含了 <limits> 就失效了。
#if defined(__clang__)
    #define TENSORVIA_DENORM_DEPRECATION_PUSH                                    \
        _Pragma("clang diagnostic push")                                         \
        _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"")
    #define TENSORVIA_DENORM_DEPRECATION_POP _Pragma("clang diagnostic pop")
#elif defined(_MSC_VER)
    #define TENSORVIA_DENORM_DEPRECATION_PUSH                                    \
        _Pragma("warning(push)")                                                 \
        _Pragma("warning(disable : 4996)")
    #define TENSORVIA_DENORM_DEPRECATION_POP _Pragma("warning(pop)")
#else
    #define TENSORVIA_DENORM_DEPRECATION_PUSH
    #define TENSORVIA_DENORM_DEPRECATION_POP
#endif

#endif
