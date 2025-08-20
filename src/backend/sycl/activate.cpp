
#include "backend/sycl/ops/activate.h"

namespace ops {

template <typename T>
void relu__sycl(const T* src_ptr,T* dst_ptr,size_t size,sycl::queue& q){
    q.submit([&](sycl::handler& h){
        h.parallel_for(sycl::range<1>(size),[=](auto idx){
            dst_ptr[idx] = (src_ptr[idx] > T(0)) ? src_ptr[idx] : T(0);
        });
    }).wait();
}
template <typename T,typename R>
void silu__sycl(const T* src_ptr,R* dst_ptr,size_t size,sycl::queue& q){
    q.submit([&](sycl::handler& h){
        h.parallel_for(sycl::range<1>(size),[=](auto idx){
            if constexpr(std::is_same_v<T,float16>){
                dst_ptr[idx] = R(src_ptr[idx] / (R(1)+sycl::exp(-src_ptr[idx])));
            }else if constexpr(std::is_same_v<T,bfloat16>){
                dst_ptr[idx] = R(src_ptr[idx] / (R(1)+ R(sycl::exp(float(-src_ptr[idx])))));
            }else{
                dst_ptr[idx] = R(src_ptr[idx] / (R(1)+std::exp(-src_ptr[idx])));
            }
        });
    }).wait();
}
template <typename T,typename R>
void tanh__sycl(const T* src_ptr, R* dst_ptr, size_t size, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](auto idx) {
            if constexpr(std::is_same_v<T,bfloat16>){
                dst_ptr[idx] = R(sycl::tanh(float(src_ptr[idx])));
            }else{
                dst_ptr[idx] = R(std::tanh(src_ptr[idx]));
            }
        });
    }).wait();
}
template <typename T,typename R>
void sigmoid__sycl(const T* src_ptr,R* dst_ptr,size_t size,sycl::queue& q){
    q.submit([&](sycl::handler& h){
            h.parallel_for(sycl::range<1>(size),[=](auto idx){
                if constexpr(std::is_same_v<T,bfloat16>){
                    dst_ptr[idx] = R(1) / R(1+ sycl::exp(-float(src_ptr[idx])));
                }else{
                    dst_ptr[idx] = R(1) / R( 1 + std::exp(-src_ptr[idx]));
                }
            });
        }).wait();
    }
template <typename T>
void softmax__sycl(const T* src_ptr,float* dst,size_t outer_dim ,size_t axis_dim ,size_t inner_dim,sycl::queue& q){ 
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(outer_dim * inner_dim), [=](sycl::id<1> idx) {
            size_t outer_idx = idx[0] / inner_dim;
            size_t inner_idx = idx[0] % inner_dim;
            // 找出当前切片的最大值（数值稳定性）
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < axis_dim; ++i) {
                size_t pos = (outer_idx * axis_dim + i) * inner_dim + inner_idx;
                if (src_ptr[pos] > max_val) {
                    max_val = src_ptr[pos];
                }
            }
            // 计算指数和
            float exp_sum = 0.0f;
            for (size_t i = 0; i < axis_dim; ++i) {
                size_t pos = (outer_idx * axis_dim + i) * inner_dim + inner_idx;
                float exp_val = sycl::exp(static_cast<float>(src_ptr[pos] - max_val));
                exp_sum += exp_val;
            }
            // 计算 softmax 并写入结果
            for (size_t i = 0; i < axis_dim; ++i) {
                size_t pos = (outer_idx * axis_dim + i) * inner_dim + inner_idx;
                float exp_val = sycl::exp(static_cast<float>(src_ptr[pos] - max_val));
                dst[pos] = exp_val / exp_sum;
            }
        });
    }).wait();  // 等待计算完成
}

//****************************************************
 void ReluImpl<Device::SYCL>::execute(Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        void* dst = a.data();
        void* src = a.data();
        switch (a.dtype()) {
            case DataType::INT8:            relu__sycl<int8_t>(static_cast<int8_t*>(src),static_cast<int8_t*>(dst),a.numel(),q);break;
            case DataType::INT16:           relu__sycl<int16_t>(static_cast<int16_t*>(src),static_cast<int16_t*>(dst),a.numel(),q);break;
            case DataType::INT32:           relu__sycl<int32_t>(static_cast<int32_t*>(src),static_cast<int32_t*>(dst),a.numel(),q);break;
            case DataType::INT64:           relu__sycl<int64_t>(static_cast<int64_t*>(src),static_cast<int64_t*>(dst),a.numel(),q);break;
            case DataType::FLOAT16:         relu__sycl<float16>(static_cast<float16*>(src),static_cast<float16*>(dst),a.numel(),q);break;
            case DataType::BFLOAT16:        relu__sycl<bfloat16>(static_cast<bfloat16*>(src),static_cast<bfloat16*>(dst),a.numel(),q);break;
            case DataType::FLOAT32:         relu__sycl<float32>(static_cast<float32*>(src),static_cast<float32*>(dst),a.numel(),q);break;
            case DataType::FLOAT64:         relu__sycl<float64>(static_cast<float64*>(src),static_cast<float64*>(dst),a.numel(),q);break;
            default:throw std::runtime_error("Unsupported dtype for sigmoid");
        }
    }
     Tensor ReluImpl<Device::SYCL>::execute(const Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        Tensor result(a.shape(),a.dtype(),a.device());
        void* dst = result.data();
        const void* src = a.data();
        switch (a.dtype()) {
            case DataType::INT8:            relu__sycl<int8_t>(static_cast<const int8_t*>(src),static_cast<int8_t*>(dst),a.numel(),q);break;
            case DataType::INT16:           relu__sycl<int16_t>(static_cast<const int16_t*>(src),static_cast<int16_t*>(dst),a.numel(),q);break;
            case DataType::INT32:           relu__sycl<int32_t>(static_cast<const int32_t*>(src),static_cast<int32_t*>(dst),a.numel(),q);break;
            case DataType::INT64:           relu__sycl<int64_t>(static_cast<const int64_t*>(src),static_cast<int64_t*>(dst),a.numel(),q);break;
            case DataType::FLOAT16:         relu__sycl<float16>(static_cast<const float16*>(src),static_cast<float16*>(dst),a.numel(),q);break;
            case DataType::BFLOAT16:        relu__sycl<bfloat16>(static_cast<const bfloat16*>(src),static_cast<bfloat16*>(dst),a.numel(),q);break;
            case DataType::FLOAT32:         relu__sycl<float32>(static_cast<const float32*>(src),static_cast<float32*>(dst),a.numel(),q);break;
            case DataType::FLOAT64:         relu__sycl<float64>(static_cast<const float64*>(src),static_cast<float64*>(dst),a.numel(),q);break;
            default:throw std::runtime_error("Unsupported dtype for sigmoid");
        }
        return result;
    }

 void SiluImpl<Device::SYCL>::execute(Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        void* src = a.data();
        void* dst = a.data();
        switch (a.dtype()) {
            // case DataType::INT8:            silu__sycl<int8_t>(static_cast<int8_t*>(src),static_cast<int8_t*>(dst),a.numel(),q);break;
            // case DataType::INT16:           silu__sycl<int16_t>(static_cast<int16_t*>(src),static_cast<int16_t*>(dst),a.numel(),q);break;
            // case DataType::INT32:           silu__sycl<int32_t>(static_cast<int32_t*>(src),static_cast<int32_t*>(dst),a.numel(),q);break;
            // case DataType::INT64:           silu__sycl<int64_t>(static_cast<int64_t*>(src),static_cast<int64_t*>(dst),a.numel(),q);break;
            case DataType::FLOAT16:         silu__sycl<float16,float16>(static_cast<float16*>(src),static_cast<float16*>(dst),a.numel(),q);break;
            case DataType::BFLOAT16:        silu__sycl<bfloat16,bfloat16>(static_cast<bfloat16*>(src),static_cast<bfloat16*>(dst),a.numel(),q);break;
            case DataType::FLOAT32:         silu__sycl<float32,float32>(static_cast<float32*>(src),static_cast<float32*>(dst),a.numel(),q);break;
            case DataType::FLOAT64:         silu__sycl<float64,float64>(static_cast<float64*>(src),static_cast<float64*>(dst),a.numel(),q);break;
            default:throw std::runtime_error("Unsupported dtype for sigmoid");
        }
    }
    Tensor SiluImpl<Device::SYCL>::execute(const Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        const void* src = a.data();

        Tensor result;
        switch (a.dtype()) {
            case DataType::INT8:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                silu__sycl<int8_t,float32>(static_cast<const int8_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT16:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                silu__sycl<int16_t,float32>(static_cast<const int16_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT32:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                silu__sycl<int32_t,float32>(static_cast<const int32_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT64:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                silu__sycl<int64_t,float32>(static_cast<const int64_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT16:{
                result = Tensor(a.shape(), DataType::FLOAT16,a.device());
                silu__sycl<float16,float16>(static_cast<const float16*>(src),static_cast<float16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::BFLOAT16:{
                result = Tensor(a.shape(), DataType::BFLOAT16,a.device());
                silu__sycl<bfloat16,bfloat16>(static_cast<const bfloat16*>(src),static_cast<bfloat16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT32: {
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                silu__sycl<float32,float32>(static_cast<const float32*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT64:{
                result = Tensor(a.shape(), DataType::FLOAT64,a.device());
                silu__sycl<float64,float64>(static_cast<const float64*>(src),static_cast<float64*>(result.data()),a.numel(),q);
                break;
            }
            default:throw std::runtime_error("Unsupported dtype for sigmoid");
        }
        return result;
    }

void TanhImpl<Device::SYCL>::execute(Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        void* src = a.data();
        void* dst = a.data();
        switch (a.dtype()) {
            // case DataType::INT8:            tanh__sycl<int8_t>(static_cast<int8_t*>(src),static_cast<int8_t*>(dst),a.numel(),q);break;
            // case DataType::INT16:           tanh__sycl<int16_t>(static_cast<int16_t*>(src),static_cast<int16_t*>(dst),a.numel(),q);break;
            // case DataType::INT32:           tanh__sycl<int32_t>(static_cast<int32_t*>(src),static_cast<int32_t*>(dst),a.numel(),q);break;
            // case DataType::INT64:           tanh__sycl<int64_t>(static_cast<int64_t*>(src),static_cast<int64_t*>(dst),a.numel(),q);break;
            case DataType::FLOAT16:         tanh__sycl<float16,float16>(static_cast<float16*>(src),static_cast<float16*>(dst),a.numel(),q);break;
            case DataType::BFLOAT16:        tanh__sycl<bfloat16,bfloat16>(static_cast<bfloat16*>(src),static_cast<bfloat16*>(dst),a.numel(),q);break;
            case DataType::FLOAT32:         tanh__sycl<float32,float32>(static_cast<float32*>(src),static_cast<float32*>(dst),a.numel(),q);break;
            case DataType::FLOAT64:         tanh__sycl<float64,float64>(static_cast<float64*>(src),static_cast<float64*>(dst),a.numel(),q);break;
            default:throw std::runtime_error("Unsupported dtype for sigmoid");
        }
    }
    Tensor TanhImpl<Device::SYCL>::execute(const Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        const void* src = a.data();

        Tensor result;
        switch (a.dtype()) {
            case DataType::INT8:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                tanh__sycl<int8_t,float32>(static_cast<const int8_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT16:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                tanh__sycl<int16_t,float32>(static_cast<const int16_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT32:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                tanh__sycl<int32_t,float32>(static_cast<const int32_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT64:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                tanh__sycl<int64_t,float32>(static_cast<const int64_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT16:{
                result = Tensor(a.shape(), DataType::FLOAT16,a.device());
                tanh__sycl<float16,float16>(static_cast<const float16*>(src),static_cast<float16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::BFLOAT16:{
                result = Tensor(a.shape(), DataType::BFLOAT16,a.device());
                tanh__sycl<bfloat16,bfloat16>(static_cast<const bfloat16*>(src),static_cast<bfloat16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT32: {
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                tanh__sycl<float32,float32>(static_cast<const float32*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT64:{
                result = Tensor(a.shape(), DataType::FLOAT64,a.device());
                tanh__sycl<float64,float64>(static_cast<const float64*>(src),static_cast<float64*>(result.data()),a.numel(),q);
                break;
            }
            default:throw std::runtime_error("Unsupported dtype for sigmoid");
        }
        return result;
    }
void SigmoidImpl<Device::SYCL>::execute(Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        void* src = a.data();
        void* dst = a.data();
        switch (a.dtype()) {
            // case DataType::INT8:            sigmoid__sycl<int8_t>(static_cast<int8_t*>(src),static_cast<int8_t*>(dst),a.numel(),q);break;
            // case DataType::INT16:           sigmoid__sycl<int16_t>(static_cast<int16_t*>(src),static_cast<int16_t*>(dst),a.numel(),q);break;
            // case DataType::INT32:           sigmoid__sycl<int32_t>(static_cast<int32_t*>(src),static_cast<int32_t*>(dst),a.numel(),q);break;
            // case DataType::INT64:           sigmoid__sycl<int64_t>(static_cast<int64_t*>(src),static_cast<int64_t*>(dst),a.numel(),q);break;
            case DataType::FLOAT16:         sigmoid__sycl<float16,float16>(static_cast<float16*>(src),static_cast<float16*>(dst),a.numel(),q);break;
            case DataType::BFLOAT16:        sigmoid__sycl<bfloat16,bfloat16>(static_cast<bfloat16*>(src),static_cast<bfloat16*>(dst),a.numel(),q);break;
            case DataType::FLOAT32:         sigmoid__sycl<float32,float32>(static_cast<float32*>(src),static_cast<float32*>(dst),a.numel(),q);break;
            case DataType::FLOAT64:         sigmoid__sycl<float64,float64>(static_cast<float64*>(src),static_cast<float64*>(dst),a.numel(),q);break;
            default:throw std::runtime_error("Unsupported dtype for sigmoid");
        }
    }
    Tensor SigmoidImpl<Device::SYCL>::execute(const Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        const void* src = a.data();

        Tensor result;
        switch (a.dtype()) {
            case DataType::INT8:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                sigmoid__sycl<int8_t,float32>(static_cast<const int8_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT16:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                sigmoid__sycl<int16_t,float32>(static_cast<const int16_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT32:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                sigmoid__sycl<int32_t,float32>(static_cast<const int32_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT64:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                sigmoid__sycl<int64_t,float32>(static_cast<const int64_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT16:{
                result = Tensor(a.shape(), DataType::FLOAT16,a.device());
                sigmoid__sycl<float16,float16>(static_cast<const float16*>(src),static_cast<float16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::BFLOAT16:{
                result = Tensor(a.shape(), DataType::BFLOAT16,a.device());
                sigmoid__sycl<bfloat16,bfloat16>(static_cast<const bfloat16*>(src),static_cast<bfloat16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT32: {
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                sigmoid__sycl<float32,float32>(static_cast<const float32*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT64:{
                result = Tensor(a.shape(), DataType::FLOAT64,a.device());
                sigmoid__sycl<float64,float64>(static_cast<const float64*>(src),static_cast<float64*>(result.data()),a.numel(),q);
                break;
            }
            default:throw std::runtime_error("Unsupported dtype for sigmoid");
        }
        return result;
    }

Tensor SoftmaxImpl<Device::SYCL>::execute(const Tensor& a,int axis){
        int dims = a.shape().size();
        if (axis < 0) axis += dims;  // 支持负轴索引
        // 计算沿指定轴的维度信息
        size_t outer_dim = 1;
        for (int i = 0; i < axis; ++i) {
            outer_dim *= a.shape(i);
        }
        size_t axis_dim = a.shape(axis);
        size_t inner_dim = 1;
        for (int i = axis + 1; i < dims; ++i) {
            inner_dim *= a.shape()[i];
        }
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        Tensor result(a.shape(),DataType::FLOAT32,a.device());
        const void* src = a.data();
        float* dst = static_cast<float*>(result.data());
        switch (a.dtype()) {
            case DataType::INT8:            softmax__sycl<int8_t>(static_cast<const int8_t*>(src),dst,outer_dim,axis_dim,inner_dim,q);break;
            case DataType::INT16:           softmax__sycl<int16_t>(static_cast<const int16_t*>(src),dst,outer_dim,axis_dim,inner_dim,q);break;
            case DataType::INT32:           softmax__sycl<int32_t>(static_cast<const int32_t*>(src),dst,outer_dim,axis_dim,inner_dim,q);break;
            case DataType::INT64:           softmax__sycl<int64_t>(static_cast<const int64_t*>(src),dst,outer_dim,axis_dim,inner_dim,q);break;
            case DataType::FLOAT16:         softmax__sycl<float16>(static_cast<const float16*>(src),dst,outer_dim,axis_dim,inner_dim,q);break;
            case DataType::BFLOAT16:        softmax__sycl<bfloat16>(static_cast<const bfloat16*>(src),dst,outer_dim,axis_dim,inner_dim,q);break;
            case DataType::FLOAT32:         softmax__sycl<float32>(static_cast<const float32*>(src),dst,outer_dim,axis_dim,inner_dim,q);break;
            case DataType::FLOAT64:         softmax__sycl<float64>(static_cast<const float64*>(src),dst,outer_dim,axis_dim,inner_dim,q);break;
            default:throw std::runtime_error("Unsupported dtype for softmax");
        }
        return result;
    }

template struct ReluImpl<Device::SYCL>;
template struct SiluImpl<Device::SYCL>;
template struct SigmoidImpl<Device::SYCL>;
template struct TanhImpl<Device::SYCL>;
template struct SoftmaxImpl<Device::SYCL>;

}