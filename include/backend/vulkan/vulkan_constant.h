#pragma once
#include <stdfloat>   // for std::float16_t, std::bfloat16_t
#include <cstdint>    // for int8_t, etc.
#include <concepts>

// 需要POD类型，小于128字节

// for: relu,silu,tanh,sigmoid,sin,cos,tanh,exp,sqrt,abs,
// for: equal,not_equal,greater,greater_equal,less_equal,non_zero
// for: sum,min,max
// for: typecast
// int64_t numel;


// for: add,sub,dot,div,pow,log,fill
// for: any,all
template<class T>
requires(
    std::same_as<T,int8_t>      ||
    std::same_as<T,int16_t>     ||
    std::same_as<T,int32_t>     ||
    std::same_as<T,int64_t>     ||
    std::same_as<T,std::float16_t> ||
    std::same_as<T,std::bfloat16_t>||
    std::same_as<T,double>      ||
    std::same_as<T,float>
)
struct ValueParams{
    T value;
    int64_t numel;
};


// for:random
struct RandomParams{
    float min;
    float max;
    uint32_t seed;
    int64_t numel;
};

// for: clamp
struct ClampParams{
    float min;
    float max;
    int64_t numel;
};
struct MatmulParams{
    uint32_t batch;
    uint32_t N;
    uint32_t K;
    uint32_t M;
};

// for softmax
struct SoftmaxParams{
    int32_t axis_dim;
    int32_t outer_dim;
    int32_t inner_dim;
};
// for: max_reduce,mean_reduce,argmax,argmin
// struct SoftmaxParams{
//     int32_t numel;
//     int32_t axis_size;
//     int32_t inner_size;
// };

// for: transpose_2d
struct Trans2DParams{
    uint32_t rows;
    uint32_t cols;
};

// for: transpose_nd
struct TransNDParams{
    int32_t dims;
    int32_t numel;
    int32_t axes[8];
    int32_t in_strides[8];  
    int32_t out_strides[8];
}; // 102kb


struct SliceParams{
    int32_t input_shape[8];
    int32_t slice_starts[8];
    int32_t output_shape[8];
    int32_t input_strides[8];
};  // 4*8*4 = 128kb


// for: concat
struct CopyParams{
    uint32_t subnumel; // 子张量元素数量
    uint32_t offset; // 目标张量偏移
};
struct ConcatParams {
    uint32_t  num;
    uint32_t  axis;
    uint32_t offsets[8];
    uint32_t prefix_sum[9];
    uint32_t input_sizes[8];
    uint32_t output_strides[8];
    uint32_t input_strides[8][8];
};
