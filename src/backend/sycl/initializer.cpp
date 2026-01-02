#include "backend/sycl/ops/initializer.h"
using namespace via;

namespace ops {
    
template <typename T>
inline void fill_value_sycl(T* typed_ptr,float val,size_t numel,sycl::queue& queue_) {
    T target_val = T(val);
    queue_.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(numel),[=](sycl::id<1> idx){
            size_t i = idx[0];
            typed_ptr[i] = target_val;
        });
    }).wait();
}
template <typename T>
void fill_random_sycl(T* data_ptr, size_t numel, float min, float max, sycl::queue& q) {
    uint64_t seed = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
    q.submit([=](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            oneapi::dpl::experimental::philox4x32 engine(seed);
            engine.discard(i);  // 手动推进状态，模拟 subsequence
            oneapi::dpl::uniform_real_distribution<float> dist(min, max);
            float r = dist(engine);
            data_ptr[i] = static_cast<T>(r);
        });
    }).wait();
}

Tensor ZerosImpl<Device::SYCL>::execute(const std::vector<int64_t>& shape, DataType dtype){
    Tensor temp(shape, dtype, Device::SYCL);
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(temp.get_impl()->context());
    auto& queue_ = ctx_impl->get_queue();
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        fill_value_sycl<T>(static_cast<T*>(temp.data()),0,temp.numel(),queue_);
    });
    return  temp;
}
Tensor OnesImpl<Device::SYCL>::execute(const std::vector<int64_t>& shape, DataType dtype){
    Tensor temp(shape, dtype, Device::SYCL);
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(temp.get_impl()->context());
    auto& queue_ = ctx_impl->get_queue();
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        fill_value_sycl<T>(static_cast<T*>(temp.data()),1,temp.numel(),queue_);
    });
    return  temp;
}
Tensor FillImpl<Device::SYCL>::execute(const std::vector<int64_t>& shape, DataType dtype, float value){
    Tensor temp(shape, dtype, Device::SYCL);
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(temp.get_impl()->context());
    auto& queue_ = ctx_impl->get_queue();
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        fill_value_sycl<T>(static_cast<T*>(temp.data()),value,temp.numel(),queue_);
    });
    return  temp;
}

Tensor RandomImpl<Device::SYCL>::execute(const std::vector<int64_t>& shape, DataType dtype,float min,float max){
    Tensor temp(shape, dtype, Device::SYCL);
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(temp.get_impl()->context());
    auto& queue_ = ctx_impl->get_queue();
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        fill_random_sycl<T>(static_cast<T*>(temp.data()), temp.numel(),min,max,queue_);
    });
    return  temp;
}

template struct ZerosImpl<Device::SYCL>;
template struct OnesImpl<Device::SYCL>;
template struct FillImpl<Device::SYCL>;
template struct RandomImpl<Device::SYCL>;
}