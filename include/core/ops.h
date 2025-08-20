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


    OPS_API Tensor ones(const std::vector<int>& shape, DataType dtype);
    OPS_API Tensor zeros(const std::vector<int>& shape, DataType dtype);
    OPS_API Tensor fill(const std::vector<int>& shape, DataType dtype, float value);
    OPS_API Tensor random(const std::vector<int>& shape, DataType dtype,float min,float max);
    
    OPS_API Tensor slice(const Tensor& t, const std::vector<std::pair<int, int>>& ranges);
    
    // inplace
    OPS_API void add(Tensor& a,float b);
    OPS_API void sub(Tensor& a,float b);
    OPS_API void dot(Tensor& a,float b);
    OPS_API void div(Tensor& a,float b);

    // uninplace
    OPS_API Tensor add(const Tensor& a, float b);
    OPS_API Tensor sub(const Tensor& a, float b);
    OPS_API Tensor dot(const Tensor& a, float b);
    OPS_API Tensor div(const Tensor& a, float b);
    
    OPS_API Tensor add(const Tensor& a,const Tensor& b);
    OPS_API Tensor sub(const Tensor& a,const Tensor& b);
    OPS_API Tensor dot(const Tensor& a,const Tensor& b);
    OPS_API Tensor div(const Tensor& a,const Tensor& b);

    // [w,k] @ [k,h] --> [w,h]
    // [b,w,k] @ [b,k,h] --> [b,w,h]
    OPS_API Tensor mul(const Tensor& a,const Tensor& b);

    // 激活函数
    OPS_API Tensor abs(const Tensor& a);
    OPS_API Tensor sin(const Tensor& a);
    OPS_API Tensor cos(const Tensor& a);
    OPS_API Tensor tan(const Tensor& a);
    OPS_API Tensor exp(const Tensor& a);
    OPS_API Tensor relu(const Tensor& a);
    OPS_API Tensor silu(const Tensor& a);
    OPS_API Tensor tanh(const Tensor& a);
    OPS_API Tensor sqrt(const Tensor& a);
    OPS_API Tensor sigmoid(const Tensor& a);
    OPS_API Tensor pow(const Tensor& a,float val);
    OPS_API Tensor log(const Tensor& a,float val);
    OPS_API Tensor softmax(const Tensor& a,int axis);
    OPS_API Tensor clamp(const Tensor& a,float min,float max);

    OPS_API void abs(Tensor& a);
    OPS_API void sin(Tensor& a);    // 只支持浮点数
    OPS_API void cos(Tensor& a);    // 只支持浮点数
    OPS_API void tan(Tensor& a);    // 只支持浮点数
    OPS_API void relu(Tensor& a);
    OPS_API void silu(Tensor& a);    // 只支持浮点数
    OPS_API void tanh(Tensor& a);    // 只支持浮点数
    OPS_API void sigmoid(Tensor& a); // 只支持浮点数
    OPS_API void clamp(Tensor& a,float min,float max);

    OPS_API float sum(const Tensor& a);
    OPS_API float min(const Tensor& a);
    OPS_API float max(const Tensor& a);
    OPS_API float mean(const Tensor& a);
    OPS_API Tensor sum(const Tensor& a,int axis);
    OPS_API Tensor min(const Tensor& a,int axis);
    OPS_API Tensor max(const Tensor& a,int axis);
    OPS_API Tensor mean(const Tensor& a,int axis);

    OPS_API Tensor typecast(const Tensor& a,DataType dst_type);
    OPS_API Tensor concat(const std::vector<Tensor> &tensors, int dim);

    OPS_API void transpose(Tensor& a); // for 2-d (inplace)
    OPS_API Tensor transpose(Tensor& a,std::initializer_list<int> axes); // for n-d

    OPS_API Tensor equal(const Tensor& a,const Tensor& b);
    OPS_API Tensor not_equal(const Tensor& a,const Tensor& b);
    OPS_API Tensor greater(const Tensor& a,const Tensor& b);
    OPS_API Tensor less(const Tensor& a,const Tensor& b);
    OPS_API Tensor greater_equal(const Tensor& a,const Tensor& b);
    OPS_API Tensor less_equal(const Tensor& a,const Tensor& b);

    OPS_API bool all(const Tensor& t,float val); // 判断张量 t 中是否所有元素都非val（true）。
    OPS_API bool any(const Tensor& t,float val); // 判断张量 t 中是否存在val（true）。
    OPS_API size_t nonzero(const Tensor& t); // 返回张量 t 中非零元素的数量。
    OPS_API Tensor argmax(const Tensor& t, int axis); // 沿指定维度 axis 查找最大值的位置索引（不是值）。
    OPS_API Tensor argmin(const Tensor& t, int axis); //  沿 axis 查找最小值的位置。
}