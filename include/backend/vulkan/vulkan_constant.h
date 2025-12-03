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
    int32_t rows;
    int32_t cols;
};

// for: transpose_nd
struct TransNDParams{
    int32_t dims;
    int32_t numel;
    int32_t axes[4];
    int32_t in_shape[4];
    int32_t in_strides[4];
    int32_t out_strides[4];
}; // 72kb


// for: concat
struct ConcatParams{
    int32_t axis;           //合并维度
    int32_t dims;           //张量维度
    int32_t numel;          //张量数量
    int32_t offsets[4];     //输入张量的偏移量
    int32_t all_strides[4];
    int32_t all_shapes[4];
    int32_t res_coord_weights[4];
};

