#pragma once
#include <cstdint>

// 需要POD类型，小于128字节

struct NumelParams{
    int64_t numel;
};
struct ValueParams{
    float value;
    int64_t numel;
};

struct MatmulParams{
    int32_t B;
    int32_t M;
    int32_t N;
    int32_t K;
};

struct SoftmaxParams{
    int32_t axis;
    int32_t outer_dim;
    int32_t inner_dim;
};

struct ClampParams{
    float min;
    float max;
    int64_t numel;
};
