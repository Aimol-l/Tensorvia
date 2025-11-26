#include "backend/sycl/ops/arithmetic.h"

namespace ops {
template <typename T>
void add_sycl(const Tensor& a, const Tensor& b, Tensor& out, size_t size, sycl::queue& q) {
    const T* pa = static_cast<const T*>(a.data());
    const T* pb = static_cast<const T*>(b.data());
    T* pr = static_cast<T*>(out.data());
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
            pr[i] = pa[i] + pb[i];
        });
    }).wait();
}
template <typename T>
void add_sycl(Tensor& a,float b ,size_t size, sycl::queue& q) {
    T* pa = static_cast< T*>(a.data());
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
            pa[i] += T(b);
        });
    }).wait();
}
template <typename T>
void sub_sycl(const Tensor& a, const Tensor& b, Tensor& out, size_t size, sycl::queue& q) {
    const T* pa = static_cast<const T*>(a.data());
    const T* pb = static_cast<const T*>(b.data());
    T* pr = static_cast<T*>(out.data());
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
            pr[i] = pa[i] - pb[i];
        });
    }).wait();
}
template <typename T>
void sub_sycl(Tensor& a,float b ,size_t size, sycl::queue& q) {
    T* pa = static_cast< T*>(a.data());
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
            pa[i] -= T(b);
        });
    }).wait();
}
template <typename T>
void dot_sycl(const Tensor& a, const Tensor& b, Tensor& out, size_t size, sycl::queue& q) {
    const T* pa = static_cast<const T*>(a.data());
    const T* pb = static_cast<const T*>(b.data());
    T* pr = static_cast<T*>(out.data());
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
            pr[i] = pa[i] * pb[i];
        });
    }).wait();
}
template <typename T>
void dot_sycl(Tensor& a,float b ,size_t size, sycl::queue& q) {
    T* pa = static_cast<T*>(a.data());
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
            pa[i] *= T(b);
        });
    }).wait();
}
template <typename T>
void div_sycl(const Tensor& a, const Tensor& b, Tensor& out, size_t size, sycl::queue& q) {
    const T* pa = static_cast<const T*>(a.data());
    const T* pb = static_cast<const T*>(b.data());
    T* pr = static_cast<T*>(out.data());
    T nan = std::numeric_limits<T>::quiet_NaN();
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
            if( pb[i] != 0)
                pr[i] = pa[i] / pb[i];
            else
                pr[i] = nan; // NAN
        });
    }).wait();
}
template <typename T>
void div_sycl(Tensor& a,float b ,size_t size, sycl::queue& q) {
    T* pa = static_cast<T*>(a.data());
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
            pa[i] /= T(b);
        });
    }).wait();
}

template <typename T>
void abs_sycl(const T* src_ptr,T* dst_ptr,size_t size,sycl::queue& q){
    q.submit([&](sycl::handler& h){
        h.parallel_for(sycl::range<1>(size),[=](auto idx){
            dst_ptr[idx] = std::abs(src_ptr[idx]);
        });
    }).wait();
}

template <typename T,typename R = T>
void sin_sycl(const T* src_ptr, R* dst_ptr, size_t size, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            if constexpr (std::is_same_v<T, sycl::half>) {
                // 处理half类型
                dst_ptr[idx] = sycl::sin(src_ptr[idx]);
            } 
            else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
                // 处理bfloat16类型
                dst_ptr[idx] = sycl::ext::oneapi::bfloat16(std::sin(float(src_ptr[idx])));
            }
            else {
                // 处理float/double等标准类型
                dst_ptr[idx] = R(std::sin(src_ptr[idx]));
            }
        });
    }).wait();  // 确保同步
}
template <typename T,typename R = T>
void cos_sycl(const T* src_ptr, R* dst_ptr, size_t size, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            if constexpr (std::is_same_v<T, sycl::half>) {
                // 处理half类型
                dst_ptr[idx] = sycl::cos(src_ptr[idx]);
            } 
            else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
                // 处理bfloat16类型
                dst_ptr[idx] = sycl::ext::oneapi::bfloat16(std::cos(float(src_ptr[idx])));
            }
            else {
                // 处理float/double等标准类型
                dst_ptr[idx] = R(std::cos(src_ptr[idx]));
            }
        });
    }).wait();  // 确保同步
}
template <typename T,typename R = T>
void tan_sycl(const T* src_ptr, R* dst_ptr, size_t size, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            if constexpr (std::is_same_v<T, sycl::half>) {
                // 处理half类型
                dst_ptr[idx] = sycl::tan(src_ptr[idx]);
            }else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
                // 处理bfloat16类型
                dst_ptr[idx] = sycl::ext::oneapi::bfloat16(std::tan(float(src_ptr[idx])));
            }
            else {
                // 处理float/double等标准类型
                dst_ptr[idx] = R(std::tan(src_ptr[idx]));
            }
        });
    }).wait();  // 确保同步
}
template <typename T,typename R = T>
void exp_sycl(const T* src_ptr,R* dst_ptr,size_t size,sycl::queue& q){
   q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            if constexpr (std::is_same_v<T, sycl::half>) {
                // 处理half类型
                dst_ptr[idx] = sycl::exp(src_ptr[idx]);
            } 
            else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
                // 处理bfloat16类型
                dst_ptr[idx] = sycl::ext::oneapi::bfloat16(std::exp(float(src_ptr[idx])));
            }
            else {
                // 处理float/double等标准类型
                dst_ptr[idx] = R(std::exp(src_ptr[idx]));
            }
        });
    }).wait();  // 确保同步
}
template <typename T,typename R = T>
void sqrt_sycl(const T* src_ptr,R* dst_ptr,size_t size,sycl::queue& q){
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            if constexpr (std::is_same_v<T, sycl::half>) {
                // 处理half类型
                dst_ptr[idx] = sycl::sqrt(src_ptr[idx]);
            } 
            else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
                // 处理bfloat16类型
                dst_ptr[idx] = sycl::ext::oneapi::bfloat16(std::sqrt(float(src_ptr[idx])));
            }
            else {
                // 处理float/double等标准类型
                dst_ptr[idx] = std::sqrt(R(src_ptr[idx]));
            }
        });
    }).wait();  // 确保同步
}
template <typename T,typename R=T>
void pow_sycl(const T* src_ptr,R* dst_ptr,size_t size,float val,sycl::queue& q){
    q.submit([&](sycl::handler& h){
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            if constexpr (std::is_same_v<T, sycl::half>) {
                // 处理half类型
                dst_ptr[idx] = R(sycl::pow(src_ptr[idx],T(val)));
            } else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
                // 处理bfloat16类型
                dst_ptr[idx] = sycl::ext::oneapi::bfloat16(std::pow(float(src_ptr[idx]),val));
            }
            else {
                // 处理float/double等标准类型
                dst_ptr[idx] = std::pow(R(src_ptr[idx]),val);
            }
        });
    }).wait();
}
template <typename T,typename R = T>
void log_sycl(const T* src_ptr,R* dst_ptr,size_t size,float val,sycl::queue& q){
    q.submit([&](sycl::handler& h){
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            if constexpr (std::is_same_v<T, sycl::half>) {
                // 处理half类型
                dst_ptr[idx] = sycl::log(float(src_ptr[idx])) / sycl::log(val);
            } 
            else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
                // 处理bfloat16类型
                dst_ptr[idx] = sycl::ext::oneapi::bfloat16(sycl::log(float(src_ptr[idx]))/sycl::log(val));
            }
            else {
                // 处理float/double等标准类型
                dst_ptr[idx] = R(sycl::log(R(src_ptr[idx])) / sycl::log(val));
            }
        });
    }).wait();
}
template <typename T>
void clamp_sycl(const T* src_ptr,T* dst_ptr,float min,float max,size_t size,sycl::queue& q){
    q.submit([&](sycl::handler& h){
        h.parallel_for(sycl::range<1>(size),[=](auto idx){
            T val = src_ptr[idx];
            if constexpr (std::is_integral_v<T> || std::is_same_v<T,float16> || std::is_same_v<T,bfloat16>) {
                float float_val = static_cast<float>(val);
                float float_min = static_cast<float>(min);
                float float_max = static_cast<float>(max);
                float clamped = sycl::clamp(float_val, float_min, float_max);
                dst_ptr[idx] = static_cast<T>(clamped);
            } else {
                dst_ptr[idx] = sycl::clamp(val, T(min), T(max));
            }
        });
    }).wait();
}


void AddImpl<Device::SYCL>::execute(Tensor& a,float b){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue(); 
        dispatch_dtype(a.dtype(), [&](auto type_id) {
            using T = typename decltype(type_id)::type;
            add_sycl<T>(static_cast<T*>(a.data()), b, a.numel(),q);
        });
    }
// uninplace
Tensor AddImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b) {
    // 避免自加修改：a + a 返回新 tensor
    if (&a == &b) ops::Add(a.clone(), b.clone());
    // 计算公共类别
    DataType res_type = std::max(a.dtype(),b.dtype()); // 全是int 或 全是 float 
    if(a.dtype() <= DataType::INT64 && b.dtype() > DataType::INT64){
        res_type = std::max(b.dtype(),DataType::FLOAT32);
    }else if(a.dtype() > DataType::INT64 && b.dtype() <= DataType::INT64){
        res_type = std::max(a.dtype(),DataType::FLOAT32);
    }
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);

    auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
    auto& q = ctx_impl->get_queue(); 

    size_t size = a.numel();
    Tensor result(a.shape(), res_type, Device::SYCL);

    switch (res_type) {
        case DataType::INT8:            add_sycl<int8_t>(A, B, result, size, q);break;
        case DataType::INT16:           add_sycl<int16_t>(A, B, result, size, q);break;
        case DataType::INT32:           add_sycl<int32_t>(A, B, result, size, q);break;
        case DataType::INT64:           add_sycl<int64_t>(A, B, result, size, q);break;
        case DataType::FLOAT16:         add_sycl<float16>(A, B, result, size, q);break;
        case DataType::BFLOAT16:        add_sycl<bfloat16>(A, B, result, size, q);break;
        case DataType::FLOAT32:         add_sycl<float32>(A, B, result, size, q);break;
        case DataType::FLOAT64:         add_sycl<float64>(A, B, result, size, q);break;
        default:throw std::runtime_error("Unsupported dtype for add");
    }
    return result;
}
Tensor AddImpl<Device::SYCL>::execute(const Tensor& a, float b){
    Tensor t = ops::Fill(a.shape(),a.dtype(),b);
    return ops::Add(a, t);
}

void SubImpl<Device::SYCL>::execute(Tensor& a,float b){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue(); 
        size_t size = a.numel();
        // 分发到模板 kernel（根据 dtype 决定类型）
        switch (a.dtype()) {
            case DataType::INT8:            sub_sycl<int8_t>(a, b,size, q);break;
            case DataType::INT16:           sub_sycl<int16_t>(a, b,size, q);break;
            case DataType::INT32:           sub_sycl<int32_t>(a, b,size, q);break;
            case DataType::INT64:           sub_sycl<int64_t>(a, b,size, q);break;
            case DataType::FLOAT16:         sub_sycl<float16>(a, b,size, q);break;
            case DataType::BFLOAT16:        sub_sycl<bfloat16>(a, b,size, q);break;
            case DataType::FLOAT32:         sub_sycl<float32>(a, b,size, q);break;
            case DataType::FLOAT64:         sub_sycl<float64>(a, b,size, q);break;
            default:throw std::runtime_error("Unsupported dtype for add");
        }
    }
    // uninplace
     Tensor SubImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b) {
        // 避免自加修改：a + a 返回新 tensor
        if (&a == &b) ops::Sub(a.clone(), b.clone());
       // 计算公共类别
        DataType res_type = std::max(a.dtype(),b.dtype()); // 全是int 或 全是 float 
        if(a.dtype() <= DataType::INT64 && b.dtype() > DataType::INT64){
            res_type = std::max(b.dtype(),DataType::FLOAT32);
        }else if(a.dtype() > DataType::INT64 && b.dtype() <= DataType::INT64){
            res_type = std::max(a.dtype(),DataType::FLOAT32);
        }
        const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
        const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);

        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue(); 

        size_t size = a.numel();
        Tensor result(a.shape(), res_type, Device::SYCL);

        switch (res_type) {
            case DataType::INT8:            sub_sycl<int8_t>(A, B, result, size, q);break;
            case DataType::INT16:           sub_sycl<int16_t>(A, B, result, size, q);break;
            case DataType::INT32:           sub_sycl<int32_t>(A, B, result, size, q);break;
            case DataType::INT64:           sub_sycl<int64_t>(A, B, result, size, q);break;
            case DataType::FLOAT16:         sub_sycl<float16>(A, B, result, size, q);break;
            case DataType::BFLOAT16:        sub_sycl<bfloat16>(A, B, result, size, q);break;
            case DataType::FLOAT32:         sub_sycl<float32>(A, B, result, size, q);break;
            case DataType::FLOAT64:         sub_sycl<float64>(A, B, result, size, q);break;
            default:throw std::runtime_error("Unsupported dtype for add");
        }
        return result;
    }
     Tensor SubImpl<Device::SYCL>::execute(const Tensor& a, float b){
        Tensor t = ops::Fill(a.shape(),a.dtype(),b);
        return ops::Sub(a, t);
    }
 void DotImpl<Device::SYCL>::execute(Tensor& a,float b){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue(); 
        size_t size = a.numel();
        // 分发到模板 kernel（根据 dtype 决定类型）
        switch (a.dtype()) {
            case DataType::INT8:            dot_sycl<int8_t>(a, b,size, q);break;
            case DataType::INT16:           dot_sycl<int16_t>(a, b,size, q);break;
            case DataType::INT32:           dot_sycl<int32_t>(a, b,size, q);break;
            case DataType::INT64:           dot_sycl<int64_t>(a, b,size, q);break;
            case DataType::FLOAT16:         dot_sycl<float16>(a, b,size, q);break;
            case DataType::BFLOAT16:        dot_sycl<bfloat16>(a, b,size, q);break;
            case DataType::FLOAT32:         dot_sycl<float32>(a, b,size, q);break;
            case DataType::FLOAT64:         dot_sycl<float64>(a, b,size, q);break;
            default:throw std::runtime_error("Unsupported dtype for add");
        }
    }
    // uninplace
     Tensor DotImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b) {
        // 避免自加修改：a + a 返回新 tensor
        if (&a == &b) ops::Dot(a.clone(), b.clone());
       // 计算公共类别
        DataType res_type = std::max(a.dtype(),b.dtype()); // 全是int 或 全是 float 
        if(a.dtype() <= DataType::INT64 && b.dtype() > DataType::INT64){
            res_type = std::max(b.dtype(),DataType::FLOAT32);
        }else if(a.dtype() > DataType::INT64 && b.dtype() <= DataType::INT64){
            res_type = std::max(a.dtype(),DataType::FLOAT32);
        }
        const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
        const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);

        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue(); 

        size_t size = a.numel();
        Tensor result(a.shape(), res_type, Device::SYCL);

        switch (res_type) {
            case DataType::INT8:            dot_sycl<int8_t>(A, B, result, size, q);break;
            case DataType::INT16:           dot_sycl<int16_t>(A, B, result, size, q);break;
            case DataType::INT32:           dot_sycl<int32_t>(A, B, result, size, q);break;
            case DataType::INT64:           dot_sycl<int64_t>(A, B, result, size, q);break;
            case DataType::FLOAT16:         dot_sycl<float16>(A, B, result, size, q);break;
            case DataType::BFLOAT16:        dot_sycl<bfloat16>(A, B, result, size, q);break;
            case DataType::FLOAT32:         dot_sycl<float32>(A, B, result, size, q);break;
            case DataType::FLOAT64:         dot_sycl<float64>(A, B, result, size, q);break;
            default:throw std::runtime_error("Unsupported dtype for add");
        }
        return result;
    }
     Tensor DotImpl<Device::SYCL>::execute(const Tensor& a, float b){
        Tensor t = ops::Fill(a.shape(),a.dtype(),b);
        return ops::Dot(a, t);
    }
void DivImpl<Device::SYCL>::execute(Tensor& a,float b){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue(); 
        size_t size = a.numel();
        // 分发到模板 kernel（根据 dtype 决定类型）
        switch (a.dtype()) {
            case DataType::INT8:            div_sycl<int8_t>(a, b,size, q);break;
            case DataType::INT16:           div_sycl<int16_t>(a, b,size, q);break;
            case DataType::INT32:           div_sycl<int32_t>(a, b,size, q);break;
            case DataType::INT64:           div_sycl<int64_t>(a, b,size, q);break;
            case DataType::FLOAT16:         div_sycl<float16>(a, b,size, q);break;
            case DataType::BFLOAT16:        div_sycl<bfloat16>(a, b,size, q);break;
            case DataType::FLOAT32:         div_sycl<float32>(a, b,size, q);break;
            case DataType::FLOAT64:         div_sycl<float64>(a, b,size, q);break;
            default:throw std::runtime_error("Unsupported dtype for add");
        }
    }
     // uninplace
    Tensor DivImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b) {
        // 避免自加修改：a + a 返回新 tensor
        if (&a == &b) ops::Div(a.clone(), b.clone());
       // 计算公共类别
        DataType res_type = std::max(a.dtype(),b.dtype()); // 全是int 或 全是 float 
        if(a.dtype() <= DataType::INT64 && b.dtype() > DataType::INT64){
            res_type = std::max(b.dtype(),DataType::FLOAT32);
        }else if(a.dtype() > DataType::INT64 && b.dtype() <= DataType::INT64){
            res_type = std::max(a.dtype(),DataType::FLOAT32);
        }
        const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
        const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);

        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue(); 

        size_t size = a.numel();
        Tensor result(a.shape(), res_type, Device::SYCL);

        switch (res_type) {
            case DataType::INT8:            div_sycl<int8_t>(A, B, result, size, q);break;
            case DataType::INT16:           div_sycl<int16_t>(A, B, result, size, q);break;
            case DataType::INT32:           div_sycl<int32_t>(A, B, result, size, q);break;
            case DataType::INT64:           div_sycl<int64_t>(A, B, result, size, q);break;
            case DataType::FLOAT16:         div_sycl<float16>(A, B, result, size, q);break;
            case DataType::BFLOAT16:        div_sycl<bfloat16>(A, B, result, size, q);break;
            case DataType::FLOAT32:         div_sycl<float32>(A, B, result, size, q);break;
            case DataType::FLOAT64:         div_sycl<float64>(A, B, result, size, q);break;
            default:throw std::runtime_error("Unsupported dtype for add");
        }
        return result;
    }
     Tensor DivImpl<Device::SYCL>::execute(const Tensor& a, float b){
        Tensor t = ops::Fill(a.shape(),a.dtype(),b);
        return ops::Div(a, t);
    }


void SinImpl<Device::SYCL>::execute(Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        void* dst = a.data();
        void* src = a.data();
        switch (a.dtype()) {
            // case DataType::INT8:            sin_sycl<int8_t>(static_cast<int8_t*>(src),static_cast<int8_t*>(dst),a.numel(),q);break;
            // case DataType::INT16:           sin_sycl<int16_t>(static_cast<int16_t*>(src),static_cast<int16_t*>(dst),a.numel(),q);break;
            // case DataType::INT32:           sin_sycl<int32_t>(static_cast<int32_t*>(src),static_cast<int32_t*>(dst),a.numel(),q);break;
            // case DataType::INT64:           sin_sycl<int64_t>(static_cast<int64_t*>(src),static_cast<int64_t*>(dst),a.numel(),q);break;
            case DataType::FLOAT16:         sin_sycl<float16>(static_cast<float16*>(src),static_cast<float16*>(dst),a.numel(),q);break;
            case DataType::BFLOAT16:        sin_sycl<bfloat16>(static_cast<bfloat16*>(src),static_cast<bfloat16*>(dst),a.numel(),q);break;
            case DataType::FLOAT32:         sin_sycl<float32>(static_cast<float32*>(src),static_cast<float32*>(dst),a.numel(),q);break;
            case DataType::FLOAT64:         sin_sycl<float64>(static_cast<float64*>(src),static_cast<float64*>(dst),a.numel(),q);break;
            default:throw std::runtime_error("Unsupported dtype for sin");
        }
    }
     Tensor SinImpl<Device::SYCL>::execute(const Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        const void* src = a.data();

        Tensor result;
        switch (a.dtype()) {
            case DataType::INT8:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                sin_sycl<int8_t,float32>(static_cast<const int8_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT16:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                sin_sycl<int16_t,float32>(static_cast<const int16_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT32:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                sin_sycl<int32_t,float32>(static_cast<const int32_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT64:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                sin_sycl<int64_t,float32>(static_cast<const int64_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT16:{
                result = Tensor(a.shape(), DataType::FLOAT16,a.device());
                sin_sycl<float16,float16>(static_cast<const float16*>(src),static_cast<float16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::BFLOAT16:{
                result = Tensor(a.shape(), DataType::BFLOAT16,a.device());
                sin_sycl<bfloat16,bfloat16>(static_cast<const bfloat16*>(src),static_cast<bfloat16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT32: {
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                sin_sycl<float32,float32>(static_cast<const float32*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT64:{
                result = Tensor(a.shape(), DataType::FLOAT64,a.device());
                sin_sycl<float64,float64>(static_cast<const float64*>(src),static_cast<float64*>(result.data()),a.numel(),q);
                break;
            }
            default:throw std::runtime_error("Unsupported dtype for sin");
        }
        return result;
    }
void CosImpl<Device::SYCL>::execute(Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        void* dst = a.data();
        void* src = a.data();
        switch (a.dtype()) {
            // case DataType::INT8:            cos_sycl<int8_t>(static_cast<int8_t*>(src),static_cast<int8_t*>(dst),a.numel(),q);break;
            // case DataType::INT16:           cos_sycl<int16_t>(static_cast<int16_t*>(src),static_cast<int16_t*>(dst),a.numel(),q);break;
            // case DataType::INT32:           cos_sycl<int32_t>(static_cast<int32_t*>(src),static_cast<int32_t*>(dst),a.numel(),q);break;
            // case DataType::INT64:           cos_sycl<int64_t>(static_cast<int64_t*>(src),static_cast<int64_t*>(dst),a.numel(),q);break;
            case DataType::FLOAT16:         cos_sycl<float16>(static_cast<float16*>(src),static_cast<float16*>(dst),a.numel(),q);break;
            case DataType::BFLOAT16:        cos_sycl<bfloat16>(static_cast<bfloat16*>(src),static_cast<bfloat16*>(dst),a.numel(),q);break;
            case DataType::FLOAT32:         cos_sycl<float32>(static_cast<float32*>(src),static_cast<float32*>(dst),a.numel(),q);break;
            case DataType::FLOAT64:         cos_sycl<float64>(static_cast<float64*>(src),static_cast<float64*>(dst),a.numel(),q);break;
            default:throw std::runtime_error("Unsupported dtype for cos");
        }
    }
     Tensor CosImpl<Device::SYCL>::execute(const Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        const void* src = a.data();

        Tensor result;
        switch (a.dtype()) {
            case DataType::INT8:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                cos_sycl<int8_t,float32>(static_cast<const int8_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT16:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                cos_sycl<int16_t,float32>(static_cast<const int16_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT32:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                cos_sycl<int32_t,float32>(static_cast<const int32_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT64:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                cos_sycl<int64_t,float32>(static_cast<const int64_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT16:{
                result = Tensor(a.shape(), DataType::FLOAT16,a.device());
                cos_sycl<float16,float16>(static_cast<const float16*>(src),static_cast<float16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::BFLOAT16:{
                result = Tensor(a.shape(), DataType::BFLOAT16,a.device());
                cos_sycl<bfloat16,bfloat16>(static_cast<const bfloat16*>(src),static_cast<bfloat16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT32: {
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                cos_sycl<float32,float32>(static_cast<const float32*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT64:{
                result = Tensor(a.shape(), DataType::FLOAT64,a.device());
                cos_sycl<float64,float64>(static_cast<const float64*>(src),static_cast<float64*>(result.data()),a.numel(),q);
                break;
            }
            default:throw std::runtime_error("Unsupported dtype for cos");
        }
        return result;
    }

void TanImpl<Device::SYCL>::execute(Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        void* dst = a.data();
        void* src = a.data();
        switch (a.dtype()) {
            // case DataType::INT8:            tan_sycl<int8_t>(static_cast<int8_t*>(src),static_cast<int8_t*>(dst),a.numel(),q);break;
            // case DataType::INT16:           tan_sycl<int16_t>(static_cast<int16_t*>(src),static_cast<int16_t*>(dst),a.numel(),q);break;
            // case DataType::INT32:           tan_sycl<int32_t>(static_cast<int32_t*>(src),static_cast<int32_t*>(dst),a.numel(),q);break;
            // case DataType::INT64:           tan_sycl<int64_t>(static_cast<int64_t*>(src),static_cast<int64_t*>(dst),a.numel(),q);break;
            case DataType::FLOAT16:         tan_sycl<float16>(static_cast<float16*>(src),static_cast<float16*>(dst),a.numel(),q);break;
            case DataType::BFLOAT16:        tan_sycl<bfloat16>(static_cast<bfloat16*>(src),static_cast<bfloat16*>(dst),a.numel(),q);break;
            case DataType::FLOAT32:         tan_sycl<float32>(static_cast<float32*>(src),static_cast<float32*>(dst),a.numel(),q);break;
            case DataType::FLOAT64:         tan_sycl<float64>(static_cast<float64*>(src),static_cast<float64*>(dst),a.numel(),q);break;
            default:throw std::runtime_error("Unsupported dtype for tan");
        }
    }
     Tensor TanImpl<Device::SYCL>::execute(const Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        const void* src = a.data();

        Tensor result;
        switch (a.dtype()) {
            case DataType::INT8:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                tan_sycl<int8_t,float32>(static_cast<const int8_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT16:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                tan_sycl<int16_t,float32>(static_cast<const int16_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT32:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                tan_sycl<int32_t,float32>(static_cast<const int32_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT64:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                tan_sycl<int64_t,float32>(static_cast<const int64_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT16:{
                result = Tensor(a.shape(), DataType::FLOAT16,a.device());
                tan_sycl<float16,float16>(static_cast<const float16*>(src),static_cast<float16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::BFLOAT16:{
                result = Tensor(a.shape(), DataType::BFLOAT16,a.device());
                tan_sycl<bfloat16,bfloat16>(static_cast<const bfloat16*>(src),static_cast<bfloat16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT32: {
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                tan_sycl<float32,float32>(static_cast<const float32*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT64:{
                result = Tensor(a.shape(), DataType::FLOAT64,a.device());
                tan_sycl<float64,float64>(static_cast<const float64*>(src),static_cast<float64*>(result.data()),a.numel(),q);
                break;
            }
            default:throw std::runtime_error("Unsupported dtype for tan");
        }
        return result;
    }
void ExpImpl<Device::SYCL>::execute(Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        void* dst = a.data();
        void* src = a.data();
        switch (a.dtype()) {
            // case DataType::INT8:            exp_sycl<int8_t>(static_cast<int8_t*>(src),static_cast<int8_t*>(dst),a.numel(),q);break;
            // case DataType::INT16:           exp_sycl<int16_t>(static_cast<int16_t*>(src),static_cast<int16_t*>(dst),a.numel(),q);break;
            // case DataType::INT32:           exp_sycl<int32_t>(static_cast<int32_t*>(src),static_cast<int32_t*>(dst),a.numel(),q);break;
            // case DataType::INT64:           exp_sycl<int64_t>(static_cast<int64_t*>(src),static_cast<int64_t*>(dst),a.numel(),q);break;
            case DataType::FLOAT16:         exp_sycl<float16>(static_cast<float16*>(src),static_cast<float16*>(dst),a.numel(),q);break;
            case DataType::BFLOAT16:        exp_sycl<bfloat16>(static_cast<bfloat16*>(src),static_cast<bfloat16*>(dst),a.numel(),q);break;
            case DataType::FLOAT32:         exp_sycl<float32>(static_cast<float32*>(src),static_cast<float32*>(dst),a.numel(),q);break;
            case DataType::FLOAT64:         exp_sycl<float64>(static_cast<float64*>(src),static_cast<float64*>(dst),a.numel(),q);break;
            default:throw std::runtime_error("Unsupported dtype for exp");
        }
    }
     Tensor ExpImpl<Device::SYCL>::execute(const Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        const void* src = a.data();

        Tensor result;
        switch (a.dtype()) {
            case DataType::INT8:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                exp_sycl<int8_t,float32>(static_cast<const int8_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT16:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                exp_sycl<int16_t,float32>(static_cast<const int16_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT32:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                exp_sycl<int32_t,float32>(static_cast<const int32_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT64:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                exp_sycl<int64_t,float32>(static_cast<const int64_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT16:{
                result = Tensor(a.shape(), DataType::FLOAT16,a.device());
                exp_sycl<float16,float16>(static_cast<const float16*>(src),static_cast<float16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::BFLOAT16:{
                result = Tensor(a.shape(), DataType::BFLOAT16,a.device());
                exp_sycl<bfloat16,bfloat16>(static_cast<const bfloat16*>(src),static_cast<bfloat16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT32: {
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                exp_sycl<float32,float32>(static_cast<const float32*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT64:{
                result = Tensor(a.shape(), DataType::FLOAT64,a.device());
                exp_sycl<float64,float64>(static_cast<const float64*>(src),static_cast<float64*>(result.data()),a.numel(),q);
                break;
            }
            default:throw std::runtime_error("Unsupported dtype for tan");
        }
        return result;
    }
void SqrtImpl<Device::SYCL>::execute(Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        void* dst = a.data();
        void* src = a.data();
        switch (a.dtype()) {
            // case DataType::INT8:            sqrt_sycl<int8_t>(static_cast<int8_t*>(src),static_cast<int8_t*>(dst),a.numel(),q);break;
            // case DataType::INT16:           sqrt_sycl<int16_t>(static_cast<int16_t*>(src),static_cast<int16_t*>(dst),a.numel(),q);break;
            // case DataType::INT32:           sqrt_sycl<int32_t>(static_cast<int32_t*>(src),static_cast<int32_t*>(dst),a.numel(),q);break;
            // case DataType::INT64:           sqrt_sycl<int64_t>(static_cast<int64_t*>(src),static_cast<int64_t*>(dst),a.numel(),q);break;
            case DataType::FLOAT16:         sqrt_sycl<float16>(static_cast<float16*>(src),static_cast<float16*>(dst),a.numel(),q);break;
            case DataType::BFLOAT16:        sqrt_sycl<bfloat16>(static_cast<bfloat16*>(src),static_cast<bfloat16*>(dst),a.numel(),q);break;
            case DataType::FLOAT32:         sqrt_sycl<float32>(static_cast<float32*>(src),static_cast<float32*>(dst),a.numel(),q);break;
            case DataType::FLOAT64:         sqrt_sycl<float64>(static_cast<float64*>(src),static_cast<float64*>(dst),a.numel(),q);break;
            default:throw std::runtime_error("Unsupported dtype for sigmoid");
        }
    }
    Tensor SqrtImpl<Device::SYCL>::execute(const Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        const void* src = a.data();

        Tensor result;
        switch (a.dtype()) {
            case DataType::INT8:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                sqrt_sycl<int8_t,float32>(static_cast<const int8_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT16:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                sqrt_sycl<int16_t,float32>(static_cast<const int16_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT32:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                sqrt_sycl<int32_t,float32>(static_cast<const int32_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::INT64:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                sqrt_sycl<int64_t,float32>(static_cast<const int64_t*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT16:{
                result = Tensor(a.shape(), DataType::FLOAT16,a.device());
                sqrt_sycl<float16,float16>(static_cast<const float16*>(src),static_cast<float16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::BFLOAT16:{
                result = Tensor(a.shape(), DataType::BFLOAT16,a.device());
                sqrt_sycl<bfloat16,bfloat16>(static_cast<const bfloat16*>(src),static_cast<bfloat16*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT32: {
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                sqrt_sycl<float32,float32>(static_cast<const float32*>(src),static_cast<float32*>(result.data()),a.numel(),q);
                break;
            }
            case DataType::FLOAT64:{
                result = Tensor(a.shape(), DataType::FLOAT64,a.device());
                sqrt_sycl<float64,float64>(static_cast<const float64*>(src),static_cast<float64*>(result.data()),a.numel(),q);
                break;
            }
            default:throw std::runtime_error("Unsupported dtype for tan");
        }
        return result;
    }
void PowImpl<Device::SYCL>::execute(Tensor& a,float val){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        void* dst = a.data();
        void* src = a.data();
        switch (a.dtype()) {
            // case DataType::INT8:            pow_sycl<int8_t>(static_cast<int8_t*>(src),static_cast<int8_t*>(dst),a.numel(),q);break;
            // case DataType::INT16:           pow_sycl<int16_t>(static_cast<int16_t*>(src),static_cast<int16_t*>(dst),a.numel(),q);break;
            // case DataType::INT32:           pow_sycl<int32_t>(static_cast<int32_t*>(src),static_cast<int32_t*>(dst),a.numel(),q);break;
            // case DataType::INT64:           pow_sycl<int64_t>(static_cast<int64_t*>(src),static_cast<int64_t*>(dst),a.numel(),q);break;
            case DataType::FLOAT16:         pow_sycl<float16>(static_cast<float16*>(src),static_cast<float16*>(dst),a.numel(),val,q);break;
            case DataType::BFLOAT16:        pow_sycl<bfloat16>(static_cast<bfloat16*>(src),static_cast<bfloat16*>(dst),a.numel(),val,q);break;
            case DataType::FLOAT32:         pow_sycl<float32>(static_cast<float32*>(src),static_cast<float32*>(dst),a.numel(),val,q);break;
            case DataType::FLOAT64:         pow_sycl<float64>(static_cast<float64*>(src),static_cast<float64*>(dst),a.numel(),val,q);break;
            default:throw std::runtime_error("Unsupported dtype for sigmoid");
        }
    }
     Tensor PowImpl<Device::SYCL>::execute(const Tensor& a,float val){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        const void* src = a.data();

        Tensor result;
        switch (a.dtype()) {
            case DataType::INT8:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                pow_sycl<int8_t,float32>(static_cast<const int8_t*>(src),static_cast<float32*>(result.data()),a.numel(),val,q);
                break;
            }
            case DataType::INT16:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                pow_sycl<int16_t,float32>(static_cast<const int16_t*>(src),static_cast<float32*>(result.data()),a.numel(),val,q);
                break;
            }
            case DataType::INT32:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                pow_sycl<int32_t,float32>(static_cast<const int32_t*>(src),static_cast<float32*>(result.data()),a.numel(),val,q);
                break;
            }
            case DataType::INT64:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                pow_sycl<int64_t,float32>(static_cast<const int64_t*>(src),static_cast<float32*>(result.data()),a.numel(),val,q);
                break;
            }
            case DataType::FLOAT16:{
                result = Tensor(a.shape(), DataType::FLOAT16,a.device());
                pow_sycl<float16,float16>(static_cast<const float16*>(src),static_cast<float16*>(result.data()),a.numel(),val,q);
                break;
            }
            case DataType::BFLOAT16:{
                result = Tensor(a.shape(), DataType::BFLOAT16,a.device());
                pow_sycl<bfloat16,bfloat16>(static_cast<const bfloat16*>(src),static_cast<bfloat16*>(result.data()),a.numel(),val,q);
                break;
            }
            case DataType::FLOAT32: {
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                pow_sycl<float32,float32>(static_cast<const float32*>(src),static_cast<float32*>(result.data()),a.numel(),val,q);
                break;
            }
            case DataType::FLOAT64:{
                result = Tensor(a.shape(), DataType::FLOAT64,a.device());
                pow_sycl<float64,float64>(static_cast<const float64*>(src),static_cast<float64*>(result.data()),a.numel(),val,q);
                break;
            }
            default:throw std::runtime_error("Unsupported dtype for tan");
        }
        return result;
    }
void LogImpl<Device::SYCL>::execute(Tensor& a,float val){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        void* dst = a.data();
        void* src = a.data();
        switch (a.dtype()) {
            // case DataType::INT8:            log_sycl<int8_t>(static_cast<int8_t*>(src),static_cast<int8_t*>(dst),a.numel(),q);break;
            // case DataType::INT16:           log_sycl<int16_t>(static_cast<int16_t*>(src),static_cast<int16_t*>(dst),a.numel(),q);break;
            // case DataType::INT32:           log_sycl<int32_t>(static_cast<int32_t*>(src),static_cast<int32_t*>(dst),a.numel(),q);break;
            // case DataType::INT64:           log_sycl<int64_t>(static_cast<int64_t*>(src),static_cast<int64_t*>(dst),a.numel(),q);break;
            case DataType::FLOAT16:         log_sycl<float16>(static_cast<float16*>(src),static_cast<float16*>(dst),a.numel(),val,q);break;
            case DataType::BFLOAT16:        log_sycl<bfloat16>(static_cast<bfloat16*>(src),static_cast<bfloat16*>(dst),a.numel(),val,q);break;
            case DataType::FLOAT32:         log_sycl<float32>(static_cast<float32*>(src),static_cast<float32*>(dst),a.numel(),val,q);break;
            case DataType::FLOAT64:         log_sycl<float64>(static_cast<float64*>(src),static_cast<float64*>(dst),a.numel(),val,q);break;
            default:throw std::runtime_error("Unsupported dtype for sigmoid");
        }
    }
    Tensor LogImpl<Device::SYCL>::execute(const Tensor& a,float val){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        const void* src = a.data();

        Tensor result;
        switch (a.dtype()) {
            case DataType::INT8:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                log_sycl<int8_t,float32>(static_cast<const int8_t*>(src),static_cast<float32*>(result.data()),a.numel(),val,q);
                break;
            }
            case DataType::INT16:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                log_sycl<int16_t,float32>(static_cast<const int16_t*>(src),static_cast<float32*>(result.data()),a.numel(),val,q);
                break;
            }
            case DataType::INT32:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                log_sycl<int32_t,float32>(static_cast<const int32_t*>(src),static_cast<float32*>(result.data()),a.numel(),val,q);
                break;
            }
            case DataType::INT64:{
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                log_sycl<int64_t,float32>(static_cast<const int64_t*>(src),static_cast<float32*>(result.data()),a.numel(),val,q);
                break;
            }
            case DataType::FLOAT16:{
                result = Tensor(a.shape(), DataType::FLOAT16,a.device());
                log_sycl<float16,float16>(static_cast<const float16*>(src),static_cast<float16*>(result.data()),a.numel(),val,q);
                break;
            }
            case DataType::BFLOAT16:{
                result = Tensor(a.shape(), DataType::BFLOAT16,a.device());
                log_sycl<bfloat16,bfloat16>(static_cast<const bfloat16*>(src),static_cast<bfloat16*>(result.data()),a.numel(),val,q);
                break;
            }
            case DataType::FLOAT32: {
                result = Tensor(a.shape(), DataType::FLOAT32,a.device());
                log_sycl<float32,float32>(static_cast<const float32*>(src),static_cast<float32*>(result.data()),a.numel(),val,q);
                break;
            }
            case DataType::FLOAT64:{
                result = Tensor(a.shape(), DataType::FLOAT64,a.device());
                log_sycl<float64,float64>(static_cast<const float64*>(src),static_cast<float64*>(result.data()),a.numel(),val,q);
                break;
            }
            default:throw std::runtime_error("Unsupported dtype for tan");
        }
        return result;
    }
void ClampImpl<Device::SYCL>::execute(Tensor& a,float min,float max){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        void* src = a.data();
        void* dst = a.data();
        switch (a.dtype()) {
            case DataType::INT8:            clamp_sycl<int8_t>(static_cast<int8_t*>(src),static_cast<int8_t*>(dst),min,max,a.numel(),q);break;
            case DataType::INT16:           clamp_sycl<int16_t>(static_cast<int16_t*>(src),static_cast<int16_t*>(dst),min,max,a.numel(),q);break;
            case DataType::INT32:           clamp_sycl<int32_t>(static_cast<int32_t*>(src),static_cast<int32_t*>(dst),min,max,a.numel(),q);break;
            case DataType::INT64:           clamp_sycl<int64_t>(static_cast<int64_t*>(src),static_cast<int64_t*>(dst),min,max,a.numel(),q);break;
            case DataType::FLOAT16:         clamp_sycl<float16>(static_cast<float16*>(src),static_cast<float16*>(dst),min,max,a.numel(),q);break;
            case DataType::BFLOAT16:        clamp_sycl<bfloat16>(static_cast<bfloat16*>(src),static_cast<bfloat16*>(dst),min,max,a.numel(),q);break;
            case DataType::FLOAT32:         clamp_sycl<float32>(static_cast<float32*>(src),static_cast<float32*>(dst),min,max,a.numel(),q);break;
            case DataType::FLOAT64:         clamp_sycl<float64>(static_cast<float64*>(src),static_cast<float64*>(dst),min,max,a.numel(),q);break;
            default:throw std::runtime_error("Unsupported dtype for clamp");
        }
    }
    Tensor ClampImpl<Device::SYCL>::execute(const Tensor& a,float min,float max){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        Tensor result(a.shape(),a.dtype(),a.device());
        const void* src = a.data();
        void* dst = result.data();
        switch (a.dtype()) {
            case DataType::INT8:            clamp_sycl<int8_t>(static_cast<const int8_t*>(src),static_cast<int8_t*>(dst),min,max,a.numel(),q);break;
            case DataType::INT16:           clamp_sycl<int16_t>(static_cast<const int16_t*>(src),static_cast<int16_t*>(dst),min,max,a.numel(),q);break;
            case DataType::INT32:           clamp_sycl<int32_t>(static_cast<const int32_t*>(src),static_cast<int32_t*>(dst),min,max,a.numel(),q);break;
            case DataType::INT64:           clamp_sycl<int64_t>(static_cast<const int64_t*>(src),static_cast<int64_t*>(dst),min,max,a.numel(),q);break;
            case DataType::FLOAT16:         clamp_sycl<float16>(static_cast<const float16*>(src),static_cast<float16*>(dst),min,max,a.numel(),q);break;
            case DataType::BFLOAT16:        clamp_sycl<bfloat16>(static_cast<const bfloat16*>(src),static_cast<bfloat16*>(dst),min,max,a.numel(),q);break;
            case DataType::FLOAT32:         clamp_sycl<float32>(static_cast<const float32*>(src),static_cast<float32*>(dst),min,max,a.numel(),q);break;
            case DataType::FLOAT64:         clamp_sycl<float64>(static_cast<const float64*>(src),static_cast<float64*>(dst),min,max,a.numel(),q);break;
            default:throw std::runtime_error("Unsupported dtype for clamp");
        }
        return result;
    }

void AbsImpl<Device::SYCL>::execute(Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();

        void* src = a.data();
        void* dst = a.data();

        switch (a.dtype()) {
            case DataType::INT8:            abs_sycl<int8_t>(static_cast<int8_t*>(src),static_cast<int8_t*>(dst),a.numel(),q);break;
            case DataType::INT16:           abs_sycl<int16_t>(static_cast<int16_t*>(src),static_cast<int16_t*>(dst),a.numel(),q);break;
            case DataType::INT32:           abs_sycl<int32_t>(static_cast<int32_t*>(src),static_cast<int32_t*>(dst),a.numel(),q);break;
            case DataType::INT64:           abs_sycl<int64_t>(static_cast<int64_t*>(src),static_cast<int64_t*>(dst),a.numel(),q);break;
            case DataType::FLOAT16:         abs_sycl<float16>(static_cast<float16*>(src),static_cast<float16*>(dst),a.numel(),q);break;
            case DataType::BFLOAT16:        abs_sycl<bfloat16>(static_cast<bfloat16*>(src),static_cast<bfloat16*>(dst),a.numel(),q);break;
            case DataType::FLOAT32:         abs_sycl<float32>(static_cast<float32*>(src),static_cast<float32*>(dst),a.numel(),q);break;
            case DataType::FLOAT64:         abs_sycl<float64>(static_cast<float64*>(src),static_cast<float64*>(dst),a.numel(),q);break;
            default:throw std::runtime_error("Unsupported dtype for tanh,only support float");
        }
    }
     Tensor AbsImpl<Device::SYCL>::execute(const Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        Tensor result(a.shape(),a.dtype(),a.device());
        const void* src = a.data();
        void* dst = result.data();
        switch (a.dtype()) {
            case DataType::INT8:            abs_sycl<int8_t>(static_cast<const int8_t*>(src),static_cast<int8_t*>(dst),a.numel(),q);break;
            case DataType::INT16:           abs_sycl<int16_t>(static_cast<const int16_t*>(src),static_cast<int16_t*>(dst),a.numel(),q);break;
            case DataType::INT32:           abs_sycl<int32_t>(static_cast<const int32_t*>(src),static_cast<int32_t*>(dst),a.numel(),q);break;
            case DataType::INT64:           abs_sycl<int64_t>(static_cast<const int64_t*>(src),static_cast<int64_t*>(dst),a.numel(),q);break;
            case DataType::FLOAT16:         abs_sycl<float16>(static_cast<const float16*>(src),static_cast<float16*>(dst),a.numel(),q);break;
            case DataType::BFLOAT16:        abs_sycl<bfloat16>(static_cast<const bfloat16*>(src),static_cast<bfloat16*>(dst),a.numel(),q);break;
            case DataType::FLOAT32:         abs_sycl<float32>(static_cast<const float32*>(src),static_cast<float32*>(dst),a.numel(),q);break;
            case DataType::FLOAT64:         abs_sycl<float64>(static_cast<const float64*>(src),static_cast<float64*>(dst),a.numel(),q);break;
            default:throw std::runtime_error("Unsupported dtype for tanh,only support float");
        }
        return result;
    }

 template struct AddImpl<Device::SYCL>;
 template struct SubImpl<Device::SYCL>;
 template struct DotImpl<Device::SYCL>;
 template struct DivImpl<Device::SYCL>;
 template struct SinImpl<Device::SYCL>;
 template struct CosImpl<Device::SYCL>;
 template struct TanImpl<Device::SYCL>;
 template struct ExpImpl<Device::SYCL>;
 template struct SqrtImpl<Device::SYCL>;
 template struct PowImpl<Device::SYCL>;
 template struct LogImpl<Device::SYCL>;
 template struct ClampImpl<Device::SYCL>;
 template struct AbsImpl<Device::SYCL>;

}