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
