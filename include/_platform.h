// 极小的辅助头，被每个 .cpp 的全局模块片段包含。宏不会跨模块 import 传播，
// 所以每个实现单元都要直接 #include 它，才能拿到与编译器无关的关键字宏。
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

#endif
