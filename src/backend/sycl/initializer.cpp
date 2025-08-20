#include "backend/sycl/ops/initializer.h"

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


Tensor ZerosImpl<Device::SYCL>::execute(const std::vector<int>& shape, DataType dtype){
        Tensor temp(shape, dtype, Device::SYCL);
        size_t numel = temp.numel();
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(temp.get_impl()->context());
        auto& queue_ = ctx_impl->get_queue();
        switch (dtype) {
            case DataType::INT8:    fill_value_sycl<int8_t>(static_cast<int8_t*>(temp.data()), 0,numel,queue_);break;
            case DataType::INT16:   fill_value_sycl<int16_t>(static_cast<int16_t*>(temp.data()), 0,numel,queue_); break;
            case DataType::INT32:   fill_value_sycl<int32_t>(static_cast<int32_t*>(temp.data()), 0,numel,queue_); break;
            case DataType::INT64:   fill_value_sycl<int64_t>(static_cast<int64_t*>(temp.data()), 0,numel,queue_); break;
            case DataType::FLOAT16: fill_value_sycl<float16>(static_cast<float16*>(temp.data()), 0,numel,queue_); break;
            case DataType::FLOAT32: fill_value_sycl<float32>(static_cast<float32*>(temp.data()), 0,numel,queue_); break;
            case DataType::FLOAT64: fill_value_sycl<float64>(static_cast<float64*>(temp.data()), 0,numel,queue_); break;
            case DataType::BFLOAT16:fill_value_sycl<bfloat16>(static_cast<bfloat16*>(temp.data()), 0,numel,queue_); break;
        }
        return  temp;
    }


Tensor OnesImpl<Device::SYCL>::execute(const std::vector<int>& shape, DataType dtype){
        Tensor temp(shape, dtype, Device::SYCL);
        size_t numel = temp.numel();
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(temp.get_impl()->context());
        auto& queue_ = ctx_impl->get_queue();
        switch (dtype) {
            case DataType::INT8:    fill_value_sycl<int8_t>(static_cast<int8_t*>(temp.data()), 1,numel,queue_);break;
            case DataType::INT16:   fill_value_sycl<int16_t>(static_cast<int16_t*>(temp.data()), 1,numel,queue_); break;
            case DataType::INT32:   fill_value_sycl<int32_t>(static_cast<int32_t*>(temp.data()), 1,numel,queue_); break;
            case DataType::INT64:   fill_value_sycl<int64_t>(static_cast<int64_t*>(temp.data()), 1,numel,queue_); break;
            case DataType::FLOAT16: fill_value_sycl<float16>(static_cast<float16*>(temp.data()), 1,numel,queue_); break;
            case DataType::FLOAT32: fill_value_sycl<float32>(static_cast<float32*>(temp.data()), 1,numel,queue_); break;
            case DataType::FLOAT64: fill_value_sycl<float64>(static_cast<float64*>(temp.data()), 1,numel,queue_); break;
            case DataType::BFLOAT16:fill_value_sycl<bfloat16>(static_cast<bfloat16*>(temp.data()), 1,numel,queue_); break;
        }
        return  temp;
    }

 Tensor FillImpl<Device::SYCL>::execute(const std::vector<int>& shape, DataType dtype, float value){
        Tensor temp(shape, dtype, Device::SYCL);
        size_t numel = temp.numel();
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(temp.get_impl()->context());
        auto& queue_ = ctx_impl->get_queue();
        switch (dtype) {
            case DataType::INT8:    fill_value_sycl<int8_t>(static_cast<int8_t*>(temp.data()), value,numel,queue_);break;
            case DataType::INT16:   fill_value_sycl<int16_t>(static_cast<int16_t*>(temp.data()), value,numel,queue_); break;
            case DataType::INT32:   fill_value_sycl<int32_t>(static_cast<int32_t*>(temp.data()), value,numel,queue_); break;
            case DataType::INT64:   fill_value_sycl<int64_t>(static_cast<int64_t*>(temp.data()), value,numel,queue_); break;
            case DataType::FLOAT16: fill_value_sycl<float16>(static_cast<float16*>(temp.data()), value,numel,queue_); break;
            case DataType::FLOAT32: fill_value_sycl<float32>(static_cast<float32*>(temp.data()), value,numel,queue_); break;
            case DataType::FLOAT64: fill_value_sycl<float64>(static_cast<float64*>(temp.data()), value,numel,queue_); break;
            case DataType::BFLOAT16:fill_value_sycl<bfloat16>(static_cast<bfloat16*>(temp.data()), value,numel,queue_); break;
        }
        return  temp;
    }


Tensor RandomImpl<Device::SYCL>::execute(const std::vector<int>& shape, DataType dtype,double min,double max){
        Tensor temp(shape, dtype, Device::SYCL);
        size_t numel = temp.numel();
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(temp.get_impl()->context());
        auto& queue_ = ctx_impl->get_queue();
        switch (dtype) {
            case DataType::INT8:     fill_random_sycl<int8_t>(static_cast<int8_t*>(temp.data()), numel,  min,max,queue_);break;
            case DataType::INT16:    fill_random_sycl<int16_t>(static_cast<int16_t*>(temp.data()), numel, min,max,queue_);break;
            case DataType::INT32:    fill_random_sycl<int32_t>(static_cast<int32_t*>(temp.data()), numel, min,max,queue_);break;
            case DataType::INT64:    fill_random_sycl<int64_t>(static_cast<int64_t*>(temp.data()), numel, min,max,queue_);break;
            case DataType::FLOAT16:  fill_random_sycl<float16>(static_cast<float16*>(temp.data()), numel, min,max,queue_);break;
            case DataType::FLOAT32:  fill_random_sycl<float32>(static_cast<float32*>(temp.data()), numel, min,max,queue_);break;
            case DataType::FLOAT64:  fill_random_sycl<float64>(static_cast<float64*>(temp.data()), numel, min,max,queue_);break;
            case DataType::BFLOAT16: fill_random_sycl<bfloat16>(static_cast<bfloat16*>(temp.data()), numel, min,max,queue_);break;
            default: throw std::runtime_error("Unsupported dtype for random initializer.");
        }
        return  temp;
    }

     template struct ZerosImpl<Device::SYCL>;
 template struct OnesImpl<Device::SYCL>;
 template struct FillImpl<Device::SYCL>;
 template struct RandomImpl<Device::SYCL>;

}