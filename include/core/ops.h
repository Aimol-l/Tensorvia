#pragma once
#include <vector>
#include "types.h"
#include "tensor.h"

#ifdef _WIN32
    #define OPS_API __declspec(dllexport)
    #define NOMINMAX 1 // prevent windows redefining min/max
#else
    #define OPS_API // Linux or macOS
#endif

namespace ops{ 

    // 非CPU后端Tensor会先 to_host(),然后再打印
    OPS_API void println(Tensor & a);
    OPS_API void println(Tensor && a);

    OPS_API Tensor Ones(const std::vector<int64_t>& shape, DataType dtype);
    OPS_API Tensor Zeros(const std::vector<int64_t>& shape, DataType dtype);
    OPS_API Tensor Fill(const std::vector<int64_t>& shape, DataType dtype, float value);
    OPS_API Tensor Random(const std::vector<int64_t>& shape, DataType dtype,float min,float max);
    
    OPS_API Tensor Slice(const Tensor& t, const std::vector<std::pair<int, int>>& ranges);
    
    // inplace
    OPS_API void Add(Tensor& a,float b);
    OPS_API void Sub(Tensor& a,float b);
    OPS_API void Dot(Tensor& a,float b);
    OPS_API void Div(Tensor& a,float b);

    // uninplace
    OPS_API Tensor Add(const Tensor& a, float b);
    OPS_API Tensor Sub(const Tensor& a, float b);
    OPS_API Tensor Dot(const Tensor& a, float b);
    OPS_API Tensor Div(const Tensor& a, float b);
    
    OPS_API Tensor Add(const Tensor& a,const Tensor& b);
    OPS_API Tensor Sub(const Tensor& a,const Tensor& b);
    OPS_API Tensor Dot(const Tensor& a,const Tensor& b);
    OPS_API Tensor Div(const Tensor& a,const Tensor& b);

    // [w,k] @ [k,h] --> [w,h]
    // [b,w,k] @ [b,k,h] --> [b,w,h]
    OPS_API Tensor Mul(const Tensor& a,const Tensor& b);

    // 激活函数
    OPS_API Tensor Abs(const Tensor& a);
    OPS_API Tensor Sin(const Tensor& a);
    OPS_API Tensor Cos(const Tensor& a);
    OPS_API Tensor Tan(const Tensor& a);
    OPS_API Tensor Exp(const Tensor& a);
    OPS_API Tensor Relu(const Tensor& a);
    OPS_API Tensor Silu(const Tensor& a);
    OPS_API Tensor Tanh(const Tensor& a);
    OPS_API Tensor Sqrt(const Tensor& a);
    OPS_API Tensor Sigmoid(const Tensor& a);
    OPS_API Tensor Pow(const Tensor& a,float val);
    OPS_API Tensor Log(const Tensor& a,float val);
    OPS_API Tensor Softmax(const Tensor& a,int axis);
    OPS_API Tensor Clamp(const Tensor& a,float min,float max);

    OPS_API void Abs(Tensor& a);
    OPS_API void Sin(Tensor& a);    // 只支持浮点数
    OPS_API void Cos(Tensor& a);    // 只支持浮点数
    OPS_API void Tan(Tensor& a);    // 只支持浮点数
    OPS_API void Relu(Tensor& a);
    OPS_API void Silu(Tensor& a);    // 只支持浮点数
    OPS_API void Tanh(Tensor& a);    // 只支持浮点数
    OPS_API void Sigmoid(Tensor& a); // 只支持浮点数
    OPS_API void Clamp(Tensor& a,float min,float max);

    OPS_API float Sum(const Tensor& a);
    OPS_API float Min(const Tensor& a);
    OPS_API float Max(const Tensor& a);
    OPS_API float Mean(const Tensor& a);
    OPS_API Tensor Sum(const Tensor& a,int axis);
    OPS_API Tensor Min(const Tensor& a,int axis);
    OPS_API Tensor Max(const Tensor& a,int axis);
    OPS_API Tensor Mean(const Tensor& a,int axis);

    OPS_API Tensor Typecast(const Tensor& a,DataType dst_type);
    OPS_API Tensor Concat(const std::vector<Tensor> &tensors, int dim);

    OPS_API void Transpose(Tensor& a); // for 2-d (inplace)
    OPS_API Tensor Transpose(Tensor& a,std::initializer_list<int64_t> axes); // for n-d

    OPS_API Tensor Equal(const Tensor& a,const Tensor& b);
    OPS_API Tensor NotEqual(const Tensor& a,const Tensor& b);
    OPS_API Tensor Greater(const Tensor& a,const Tensor& b);
    OPS_API Tensor Less(const Tensor& a,const Tensor& b);
    OPS_API Tensor GreaterEqual(const Tensor& a,const Tensor& b);
    OPS_API Tensor LessEqual(const Tensor& a,const Tensor& b);

    OPS_API bool All(const Tensor& t,float val); // 判断张量 t 中是否所有元素都非val（true）。
    OPS_API bool Any(const Tensor& t,float val); // 判断张量 t 中是否存在val（true）。
    OPS_API size_t Nonzero(const Tensor& t); // 返回张量 t 中非零元素的数量。
    OPS_API Tensor Argmax(const Tensor& t, int axis); // 沿指定维度 axis 查找最大值的位置索引（不是值）。
    OPS_API Tensor Argmin(const Tensor& t, int axis); //  沿 axis 查找最小值的位置。
}