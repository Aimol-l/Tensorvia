#include "backend/vulkan/ops/reduce.h"

namespace ops {

float SumImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    
    int64_t numel = a.numel();
    Tensor res({256},a.dtype(),Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());

    ctx_impl->submitCompute(
        OpType::Sum,
        a.dtype(),
        {src_impl->buffer(), res_impl->buffer()},
        (numel + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    // 转到host
    res.to_host();

    float sum = 0.0f;
    float* res_ptr = static_cast<float*>(res.data());
    for (int i = 0; i < 256; i++) {
        sum += res_ptr[i];
    }

    return sum;
}
Tensor SumImpl<Device::VULKAN>::execute(const Tensor& a,int axis){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    // 移除 a.shape(axis) 所在的轴
    int dims = a.shape().size();
    int32_t numel = a.numel();
    std::vector<int64_t> new_shape;
    for (int i = 0; i < dims; i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    int32_t inner_dim = 1;
    for(int i = axis + 1; i < dims; ++i){
        inner_dim *= a.shape(i);
    }
    int32_t axis_dim = a.shape(axis);
    int32_t outer_dim = 1;
    for(int i = 0; i < axis; ++i){
        outer_dim *= a.shape(i);
    }
    Tensor result(new_shape, a.dtype(), Device::VULKAN);  // 再考虑一下a 是int8_t的时候res应该是int8_t还是int32_t
    auto result_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());

    // 因为实际上要传入的参数刚好和softmax的参数所需要开辟的空间大小一致，所以这里先直接用softmax的params
    SoftmaxParams params{
        .axis_dim = axis_dim,
        .outer_dim = outer_dim * inner_dim, 
        .inner_dim = inner_dim
    };

    ctx_impl->submitCompute(
        OpType::SumVec,
        a.dtype(),
        {src_impl->buffer(), result_impl->buffer()},
        ((outer_dim * inner_dim) + 255) / 256, 1, 1,
        &params,
        sizeof(SoftmaxParams)
    );
    return result;
}
float MeanImpl<Device::VULKAN>::execute(const Tensor& a) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    float sum_val = 0.0f;
    return sum_val / a.numel();
}
Tensor MeanImpl<Device::VULKAN>::execute(const Tensor& a,int axis){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    // 移除 a.shape(axis) 所在的轴
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    int dims = a.shape().size();
    int32_t numel = a.numel();
    int32_t inner_dim = 1;
    for(int i = axis + 1; i < dims; ++i){
        inner_dim *= a.shape(i);
    }
    int32_t axis_dim = a.shape(axis);
    int32_t outer_dim = 1;
    for(int i = 0; i < dims; ++i){
        outer_dim *= a.shape(i);
    }

    SoftmaxParams params{
        .axis_dim = axis_dim,
        .outer_dim = outer_dim * inner_dim, // 因为实际上要传入的参数刚好和softmax的参数所需要开辟的空间大小一致，所以这里先直接用softmax的params
        .inner_dim = inner_dim
    };
    Tensor result(new_shape,a.dtype(),Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    ctx_impl->submitCompute(
        OpType::Mean,
        a.dtype(),
        {src_impl->buffer(), res_impl->buffer()},
        ((outer_dim * inner_dim) + 255) / 256, 1, 1,
        &params,
        sizeof(SoftmaxParams)
    );
    return result;
}
float MinImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    int64_t numel = a.numel();
    Tensor res({256},a.dtype(),Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());

    ctx_impl->submitCompute(
        OpType::Min,
        a.dtype(),
        {src_impl->buffer(), res_impl->buffer()},
        (numel + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    res.to_host();
    float min_val = MAXFLOAT;
    float* res_ptr = static_cast<float*>(res.data());
    for (int i = 0; i < 256; i++) {
        min_val = std::min(min_val, res_ptr[i]);
    }
    return min_val;
}
Tensor MinImpl<Device::VULKAN>::execute(const Tensor& a,int axis){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    // 移除 a.shape(axis) 所在的轴
    int dims = a.shape().size();
    int32_t numel = a.numel();
    std::vector<int64_t> new_shape;
    for (int i = 0; i < dims; i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    int32_t inner_dim = 1;
    for(int i = axis + 1; i < dims; ++i){
        inner_dim *= a.shape(i);
    }
    int32_t axis_dim = a.shape(axis);
    int32_t outer_dim = 1;
    for(int i = 0; i < axis; ++i){
        outer_dim *= a.shape(i);
    }
    Tensor result(new_shape, a.dtype(), Device::VULKAN);  // 再考虑一下a 是int8_t的时候res应该是int8_t还是int32_t
    auto result_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());

    SoftmaxParams params{
        .axis_dim = axis_dim,
        .outer_dim = outer_dim * inner_dim, // 因为实际上要传入的参数刚好和softmax的参数所需要开辟的空间大小一致，所以这里先直接用softmax的params
        .inner_dim = inner_dim
    };

    ctx_impl->submitCompute(
        OpType::MinVec,
        a.dtype(),
        {src_impl->buffer(), result_impl->buffer()},
        ((outer_dim * inner_dim) + 255) / 256, 1, 1,
        &params,
        sizeof(SoftmaxParams)
    );
    return result;
}
float MaxImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    int64_t numel = a.numel();
    Tensor res({256},a.dtype(),Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());

    ctx_impl->submitCompute(
        OpType::Max,
        a.dtype(),
        {src_impl->buffer(), res_impl->buffer()},
        (numel + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    res.to_host();
    float max_val = -MAXFLOAT;
    float* res_ptr = static_cast<float*>(res.data());
    for (int i = 0; i < 256; i++) {
        max_val = std::max(max_val, res_ptr[i]);
    }
    return max_val;
}
Tensor MaxImpl<Device::VULKAN>::execute(const Tensor& a,int axis){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    // 移除 a.shape(axis) 所在的轴
    int dims = a.shape().size();
    int32_t numel = a.numel();
    std::vector<int64_t> new_shape;
    for (int i = 0; i < dims; i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    int32_t inner_dim = 1;
    for(int i = axis + 1; i < dims; ++i){
        inner_dim *= a.shape(i);
    }
    int32_t axis_dim = a.shape()[axis];
    int32_t outer_dim = 1;
    for(int i = 0; i < axis; ++i){
        outer_dim *= a.shape(i);
    }
    Tensor result(new_shape, a.dtype(), Device::VULKAN);  // 再考虑一下a 是int8_t的时候res应该是int8_t还是int32_t
    auto result_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());

    SoftmaxParams params{
        .axis_dim = axis_dim,
        .outer_dim = outer_dim * inner_dim, // 因为实际上要传入的参数刚好和softmax的参数所需要开辟的空间大小一致，所以这里先直接用softmax的params
        .inner_dim = inner_dim
    };

    ctx_impl->submitCompute(
        OpType::MaxVec,
        a.dtype(),
        {src_impl->buffer(), result_impl->buffer()},
        ((outer_dim * inner_dim) + 255) / 256, 1, 1,
        &params,
        sizeof(SoftmaxParams)
    );
    return result;
}

Tensor ArgMaxImpl<Device::VULKAN>::execute(const Tensor &a, int axis){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    auto a_shape = a.shape();
    int dim = a_shape.size();
    int numel = a.numel();
    int32_t outer_dim = 1;
    for(int i = 0; i < axis; ++i) outer_dim *= a_shape[i];
    int32_t axis_dim = a.shape()[axis];
    int32_t inner_dim = 1;
    for(int i = axis+1; i < dim; ++i) inner_dim *= a_shape[i];
    // 移除 a.shape(axis) 所在的轴
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    Tensor result(new_shape,DataType::INT32,Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());

    SoftmaxParams params{
        .axis_dim = axis_dim,
        .outer_dim = numel, // 因为实际上要传入的参数刚好和softmax的参数所需要开辟的空间大小一致，所以这里先直接用softmax的params
        .inner_dim = inner_dim
    };

    ctx_impl->submitCompute(
        OpType::Argmax,
        a.dtype(),
        {src_impl->buffer(), res_impl->buffer()},
        ((numel + 255) / 256), 1, 1,
        &params,
        sizeof(SoftmaxParams)
    );

    return result;
}

Tensor ArgMinImpl<Device::VULKAN>::execute(const Tensor &a, int axis) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    auto a_shape = a.shape();
    int dim = a_shape.size();
    int numel = a.numel();
    int32_t outer_dim = 1;
    for(int i = 0; i < axis; ++i) outer_dim *= a_shape[i];
    int32_t axis_dim = a.shape()[axis];
    int32_t inner_dim = 1;
    for(int i = axis+1; i < dim; ++i) inner_dim *= a_shape[i];
    // 移除 a.shape(axis) 所在的轴
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    Tensor result(new_shape,DataType::INT32,Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());

    SoftmaxParams params{
        .axis_dim = axis_dim,
        .outer_dim = numel, // 因为实际上要传入的参数刚好和softmax的参数所需要开辟的空间大小一致，所以这里先直接用softmax的params
        .inner_dim = inner_dim
    };

    ctx_impl->submitCompute(
        OpType::Argmin,
        a.dtype(),
        {src_impl->buffer(), res_impl->buffer()},
        ((numel + 255) / 256), 1, 1,
        &params,
        sizeof(SoftmaxParams)
    );

    return result;
}
bool AnyImpl<Device::VULKAN>::execute(const Tensor& a,float val) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    // 无法在函数接口中传 bool 指针，用 Tensor 的 INT8 代替，但是shader中的原子操作不支持int8，暂时这里用int32代替
    // Tensor result({1},DataType::INT32,Device::VULKAN);
    auto result = Tensor::Fill({1}, 0, DataType::INT32);

    // 初始化为0
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    int32_t numel = a.numel();

    ValueParams<float> params{
        .value = val,
        .numel = numel
    };

    ctx_impl->submitCompute(
        OpType::Any,
        a.dtype(),
        {src_impl->buffer(), res_impl->buffer()},
        ((numel + 255) / 256), 1, 1,
        &params,
        sizeof(params)
    );

    result.to_host();

    int res = static_cast<int*>(result.data())[0];


    return res == 1;
}
bool AllImpl<Device::VULKAN>::execute(const Tensor& a,float val) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    // 无法在函数接口中传 bool 指针，用 Tensor 的 INT8 代替
    Tensor result = Tensor::Fill({1}, 1, DataType::INT32);

    auto res_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    int32_t numel = a.numel();

    ValueParams<float> params{
        .value = val,
        .numel = numel
    };

    ctx_impl->submitCompute(
        OpType::All,
        a.dtype(),
        {src_impl->buffer(), res_impl->buffer()},
        ((numel + 255) / 256), 1, 1,
        &params,
        sizeof(params)
    );

    result.to_host();

    int res = static_cast<int*>(result.data())[0];

    return res == 1;
}

 template struct SumImpl<Device::VULKAN>;
 template struct MeanImpl<Device::VULKAN>;
 template struct MinImpl<Device::VULKAN>;
 template struct MaxImpl<Device::VULKAN>;
 template struct ArgMaxImpl<Device::VULKAN>;
 template struct ArgMinImpl<Device::VULKAN>;
 template struct AnyImpl<Device::VULKAN>;
 template struct AllImpl<Device::VULKAN>;

}