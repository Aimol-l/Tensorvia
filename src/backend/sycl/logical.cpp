#include "backend/sycl/ops/logical.h"

namespace ops {

template <typename T,typename R=T>
void equal_sycl(int8_t *res, const T *a_ptr, const R *b_ptr, int size, sycl::queue& q) {
    // 定义容差，逻辑保持不变
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    constexpr PromotedType abs_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-9);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-5);
        else if constexpr (std::is_same_v<T, float16> || std::is_same_v<R, float16>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<R, bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    constexpr PromotedType rel_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-12);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-6);
        else if constexpr (std::is_same_v<T, float16> || std::is_same_v<R, float16>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<R, bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
            if constexpr (std::is_integral_v<T> && std::is_integral_v<R>) {
                res[i] = a_ptr[i] == b_ptr[i] ? 1:0;
            }else{
                // 至少存在一个浮点
                const PromotedType val_a = a_ptr[i];
                const PromotedType val_b = b_ptr[i];
                res[i] = (std::abs(val_a - val_b) <= std::max(rel_tol * std::max(std::abs(val_a), std::abs(val_b)), abs_tol)) ? 1 : 0;
            }
        });
    }).wait();
}

template <typename T,typename R=T>
void not_equal_sycl(int8_t *res, const T *a_ptr, const R *b_ptr, int size, sycl::queue& q) {
    // 定义容差，逻辑保持不变
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    constexpr PromotedType abs_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-9);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-5);
        else if constexpr (std::is_same_v<T, float16> || std::is_same_v<R, float16>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<R, bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    constexpr PromotedType rel_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-12);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-6);
        else if constexpr (std::is_same_v<T, float16> || std::is_same_v<R, float16>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<R, bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
            if constexpr (std::is_integral_v<T> && std::is_integral_v<R>) {
                res[i] = a_ptr[i] != b_ptr[i] ? 1:0;
            }else{
                // 至少存在一个浮点
                const PromotedType val_a = a_ptr[i];
                const PromotedType val_b = b_ptr[i];
                res[i] = (std::abs(val_a - val_b) > std::max(rel_tol * std::max(std::abs(val_a), std::abs(val_b)), abs_tol)) ? 1 : 0;
            }
        });
    }).wait();
}

template <typename T,typename R=T>
void greater_sycl(int8_t *res, const T *a_ptr, const R *b_ptr, int size, sycl::queue& q) {
    // 定义容差，逻辑保持不变
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
            if constexpr (std::is_integral_v<T> && std::is_integral_v<R>) {
                res[i] = a_ptr[i] > b_ptr[i] ? 1:0;
            }else{
                // 至少存在一个浮点
                const PromotedType val_a = a_ptr[i];
                const PromotedType val_b = b_ptr[i];
                res[i] = val_a > val_b ? 1 : 0;
            }
        });
    }).wait();
}

template <typename T,typename R=T>
void less_sycl(int8_t *res, const T *a_ptr, const R *b_ptr, int size, sycl::queue& q) {
    // 定义容差，逻辑保持不变
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
            if constexpr (std::is_integral_v<T> && std::is_integral_v<R>) {
                res[i] = a_ptr[i] < b_ptr[i] ? 1:0;
            }else{
                // 至少存在一个浮点
                const PromotedType val_a = a_ptr[i];
                const PromotedType val_b = b_ptr[i];
                res[i] = val_a < val_b ? 1 : 0;
            }
        });
    }).wait();
}

template <typename T,typename R=T>
void greater_equal_sycl(int8_t *res, const T *a_ptr, const R *b_ptr, int size, sycl::queue& q) {
    // 定义容差，逻辑保持不变
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
            if constexpr (std::is_integral_v<T> && std::is_integral_v<R>) {
                res[i] = a_ptr[i] >= b_ptr[i] ? 1:0;
            }else{
                // 至少存在一个浮点
                const PromotedType val_a = a_ptr[i];
                const PromotedType val_b = b_ptr[i];
                res[i] = val_a >= val_b ? 1 : 0;
            }
        });
    }).wait();
}
template <typename T,typename R=T>
void less_equal_sycl(int8_t *res, const T *a_ptr, const R *b_ptr, int size, sycl::queue& q) {
    // 定义容差，逻辑保持不变
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
            if constexpr (std::is_integral_v<T> && std::is_integral_v<R>) {
                res[i] = a_ptr[i] <= b_ptr[i] ? 1:0;
            }else{
                // 至少存在一个浮点
                const PromotedType val_a = a_ptr[i];
                const PromotedType val_b = b_ptr[i];
                res[i] = val_a <= val_b ? 1 : 0;
            }
        });
    }).wait();
}
template <typename T>
size_t non_zero_sycl(const Tensor& a, size_t size, sycl::queue& q) {
    const T* a_ptr = static_cast<const T*>(a.data());
    // 1. 确定工作组大小（建议256-1024之间）
    constexpr size_t work_group_size = 256;
    const size_t num_work_groups = (size + work_group_size - 1) / work_group_size;
    // 2. 分配临时存储空间
    auto local_counts = sycl::malloc_shared<size_t>(num_work_groups, q);
    std::fill(local_counts, local_counts + num_work_groups, 0);
    // 3. 定义epsilon
    constexpr auto epsilon = []{
        if constexpr (std::is_same_v<T, float>) return 1e-5f;
        else if constexpr (std::is_same_v<T, double>) return 1e-5;
        else if constexpr (std::is_same_v<T, sycl::half>) return 1e-3f;
        else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) return 1e-2f;
        else return T{0}; // 整数类型
    }();
    // 4. 第一阶段：工作组内局部归约
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>{num_work_groups * work_group_size, work_group_size}, 
        [=](sycl::nd_item<1> item) {
            const size_t global_id = item.get_global_id(0);
            const size_t group_id = item.get_group(0);
            // 每个工作项处理自己的数据
            size_t local_count = 0;
            if (global_id < size) {
                T value = a_ptr[global_id];
                bool is_non_zero;
                if constexpr (std::is_integral_v<T>) {
                    is_non_zero = (value != T(0));
                } else {
                    is_non_zero = (sycl::fabs(float(value)) > epsilon);
                }
                local_count = is_non_zero ? 1 : 0;
            }
            // 工作组内归约
            size_t group_sum = sycl::reduce_over_group(
                item.get_group(), local_count, sycl::plus<>()
            );
            // 第一个工作项保存组内结果
            if (item.get_local_id(0) == 0) {
                local_counts[group_id] = group_sum;
            }
        });
    }).wait();
    // 5. 第二阶段：主机端汇总（小数据量）
    size_t total = 0;
    for(size_t i = 0; i < num_work_groups; ++i){
        total += local_counts[i];
    }
    // 6. 释放资源
    sycl::free(local_counts, q);
    return total;
}


//************************************
Tensor EqualImpl<Device::SYCL>::execute(const Tensor& a,const Tensor& b) {

        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();

        auto A = data_as_const_variant(a.dtype(),a.data());
        auto B = data_as_const_variant(b.dtype(),b.data());
        Tensor res(a.shape(), DataType::INT8, Device::SYCL);

        std::visit([&](auto ptr_A, auto ptr_B) {
            using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
            using BType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_B)>>;
            equal_sycl<AType,BType>(static_cast<int8_t*>(res.data()),ptr_A,ptr_B,res.numel(),q);
        }, A, B);
        return res;
    }

Tensor NotEqualImpl<Device::SYCL>::execute(const Tensor& a,const Tensor& b) {
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();

        auto A = data_as_const_variant(a.dtype(),a.data());
        auto B = data_as_const_variant(b.dtype(),b.data());
        Tensor res(a.shape(), DataType::INT8, Device::SYCL);

        std::visit([&](auto ptr_A, auto ptr_B) {
            using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
            using BType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_B)>>;
            not_equal_sycl<AType,BType>(static_cast<int8_t*>(res.data()),ptr_A,ptr_B,res.numel(),q);
        }, A, B);
        return res;
    }
Tensor GreaterImpl<Device::SYCL>::execute(const Tensor& a,const Tensor& b) {
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();

        auto A = data_as_const_variant(a.dtype(),a.data());
        auto B = data_as_const_variant(b.dtype(),b.data());
        Tensor res(a.shape(), DataType::INT8, Device::SYCL);

        std::visit([&](auto ptr_A, auto ptr_B) {
            using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
            using BType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_B)>>;
            greater_sycl<AType,BType>(static_cast<int8_t*>(res.data()),ptr_A,ptr_B,res.numel(),q);
        }, A, B);
        return res;
    }
Tensor LessImpl<Device::SYCL>::execute(const Tensor& a,const Tensor& b) {
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();

        auto A = data_as_const_variant(a.dtype(),a.data());
        auto B = data_as_const_variant(b.dtype(),b.data());
        Tensor res(a.shape(), DataType::INT8, Device::SYCL);

        std::visit([&](auto ptr_A, auto ptr_B) {
            using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
            using BType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_B)>>;
            less_sycl<AType,BType>(static_cast<int8_t*>(res.data()),ptr_A,ptr_B,res.numel(),q);
        }, A, B);
        return res;
    }
Tensor GreaterEqualImpl<Device::SYCL>::execute(const Tensor& a,const Tensor& b) {
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();

        auto A = data_as_const_variant(a.dtype(),a.data());
        auto B = data_as_const_variant(b.dtype(),b.data());
        Tensor res(a.shape(), DataType::INT8, Device::SYCL);

        std::visit([&](auto ptr_A, auto ptr_B) {
            using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
            using BType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_B)>>;
            greater_equal_sycl<AType,BType>(static_cast<int8_t*>(res.data()),ptr_A,ptr_B,res.numel(),q);
        }, A, B);
        return res;
    }
Tensor LessEqualImpl<Device::SYCL>::execute(const Tensor& a,const Tensor& b) {
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();

        auto A = data_as_const_variant(a.dtype(),a.data());
        auto B = data_as_const_variant(b.dtype(),b.data());
        Tensor res(a.shape(), DataType::INT8, Device::SYCL);

        std::visit([&](auto ptr_A, auto ptr_B) {
            using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
            using BType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_B)>>;
            less_equal_sycl<AType,BType>(static_cast<int8_t*>(res.data()),ptr_A,ptr_B,res.numel(),q);
        }, A, B);
        return res;
    }
size_t NonZeroImpl<Device::SYCL>::execute(const Tensor& a) {
        auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        size_t count = 0;
        switch (a.dtype()) {
            case DataType::INT8:     count = non_zero_sycl<int8_t>(a, a.numel(), q); break;
            case DataType::INT16:    count = non_zero_sycl<int16_t>(a, a.numel(), q); break;
            case DataType::INT32:    count = non_zero_sycl<int32_t>(a, a.numel(), q); break;
            case DataType::INT64:    count = non_zero_sycl<int64_t>(a, a.numel(), q); break;
            case DataType::FLOAT16:  count = non_zero_sycl<float16>(a, a.numel(), q); break;
            case DataType::BFLOAT16: count = non_zero_sycl<bfloat16>(a, a.numel(), q); break;
            case DataType::FLOAT32:  count = non_zero_sycl<float32>(a, a.numel(), q); break;
            case DataType::FLOAT64:  count = non_zero_sycl<float64>(a, a.numel(), q); break;
            default: throw std::runtime_error("Unsupported dtype for non-zero");
        }
        return count;
    }

 template struct EqualImpl<Device::SYCL>;
 template struct NotEqualImpl<Device::SYCL>;
 template struct GreaterImpl<Device::SYCL>;
 template struct LessImpl<Device::SYCL>;
 template struct GreaterEqualImpl<Device::SYCL>;
 template struct LessEqualImpl<Device::SYCL>;
 template struct NonZeroImpl<Device::SYCL>;

}