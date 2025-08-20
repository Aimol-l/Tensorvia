#include "backend/sycl/ops/reduce.h"

namespace ops {

template <typename T>
void argmax_sycl(const Tensor& a, Tensor& res, int axis, sycl::queue& q) {
    const T* a_ptr = static_cast<const T*>(a.data());
    int32_t* out_ptr = static_cast<int32_t*>(res.data());
    const auto& in_shape = a.shape();
    int ndim = in_shape.size();
    int axis_size = in_shape[axis];
    size_t outer_size = 1;
    size_t inner_size = 1;
    for (int i = 0; i < axis; ++i) {
        outer_size *= in_shape[i];
    }
    for (int i = axis + 1; i < ndim; ++i) {
        inner_size *= in_shape[i];
    }
    size_t total = outer_size * inner_size;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
            size_t id = idx[0];
            size_t outer_idx = id / inner_size;
            size_t inner_idx = id % inner_size;
            T max_val = a_ptr[outer_idx * axis_size * inner_size + 0 * inner_size + inner_idx];
            int max_idx = 0;
            for (int i = 1; i < axis_size; ++i) {
                T val = a_ptr[outer_idx * axis_size * inner_size + i * inner_size + inner_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }
            out_ptr[id] = max_idx;
        });
    }).wait();
}
template <typename T>
void max_sycl(const Tensor& a, Tensor& res, int axis, sycl::queue& q) {
    const T* a_ptr = static_cast<const T*>(a.data());
    T* out_ptr = static_cast<T*>(res.data());
    const auto& in_shape = a.shape();
    int ndim = in_shape.size();
    int axis_size = in_shape[axis];
    size_t outer_size = 1;
    size_t inner_size = 1;
    for (int i = 0; i < axis; ++i) {
        outer_size *= in_shape[i];
    }
    for (int i = axis + 1; i < ndim; ++i) {
        inner_size *= in_shape[i];
    }
    size_t total = outer_size * inner_size;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
            size_t id = idx[0];
            size_t outer_idx = id / inner_size;
            size_t inner_idx = id % inner_size;
            T max_val = a_ptr[outer_idx * axis_size * inner_size + 0 * inner_size + inner_idx];
            for (int i = 1; i < axis_size; ++i) {
                T val = a_ptr[outer_idx * axis_size * inner_size + i * inner_size + inner_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
            out_ptr[id] = max_val;
        });
    }).wait();
}
template <typename T>
void mean_sycl(const Tensor& a, Tensor& res, int axis, sycl::queue& q) {
    const T* a_ptr = static_cast<const T*>(a.data());
    T* out_ptr = static_cast<T*>(res.data());
    const auto& in_shape = a.shape();
    int ndim = in_shape.size();
    int axis_size = in_shape[axis];
    size_t outer_size = 1;
    size_t inner_size = 1;
    for (int i = 0; i < axis; ++i) {
        outer_size *= in_shape[i];
    }
    for (int i = axis + 1; i < ndim; ++i) {
        inner_size *= in_shape[i];
    }
    size_t total = outer_size * inner_size;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
            size_t id = idx[0];
            size_t outer_idx = id / inner_size;
            size_t inner_idx = id % inner_size;
            float sum_val = 0;
            for (int i = 1; i < axis_size; ++i) {
                T val = a_ptr[outer_idx * axis_size * inner_size + i * inner_size + inner_idx];
                sum_val += static_cast<float>(val);
            }
            out_ptr[id] = static_cast<float>(sum_val / axis_size);
        });
    }).wait();
}
template <typename T>
void sum_sycl(const Tensor& a, Tensor& res, int axis, sycl::queue& q) {
    const T* a_ptr = static_cast<const T*>(a.data());
    T* out_ptr = static_cast<T*>(res.data());
    const auto& in_shape = a.shape();
    int ndim = in_shape.size();
    int axis_size = in_shape[axis];
    size_t outer_size = 1;
    size_t inner_size = 1;
    for (int i = 0; i < axis; ++i) {
        outer_size *= in_shape[i];
    }
    for (int i = axis + 1; i < ndim; ++i) {
        inner_size *= in_shape[i];
    }
    size_t total = outer_size * inner_size;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
            size_t id = idx[0];
            size_t outer_idx = id / inner_size;
            size_t inner_idx = id % inner_size;
            T sum_val = T(0);
            for (int i = 1; i < axis_size; ++i) {
                T val = a_ptr[outer_idx * axis_size * inner_size + i * inner_size + inner_idx];
                sum_val += val;
            }
            out_ptr[id] = sum_val;
        });
    }).wait();
}
template <typename T>
void min_sycl(const Tensor& a, Tensor& res, int axis, sycl::queue& q) {
    const T* a_ptr = static_cast<const T*>(a.data());
    T* out_ptr = static_cast<T*>(res.data());
    const auto& in_shape = a.shape();
    int ndim = in_shape.size();
    int axis_size = in_shape[axis];
    size_t outer_size = 1;
    size_t inner_size = 1;
    for (int i = 0; i < axis; ++i) {
        outer_size *= in_shape[i];
    }
    for (int i = axis + 1; i < ndim; ++i) {
        inner_size *= in_shape[i];
    }
    size_t total = outer_size * inner_size;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
            size_t id = idx[0];
            size_t outer_idx = id / inner_size;
            size_t inner_idx = id % inner_size;
            T min_val = a_ptr[outer_idx * axis_size * inner_size + 0 * inner_size + inner_idx];
            for (int i = 1; i < axis_size; ++i) {
                T val = a_ptr[outer_idx * axis_size * inner_size + i * inner_size + inner_idx];
                if (val < min_val) {
                    min_val = val;
                }
            }
            out_ptr[id] = min_val;
        });
    }).wait();
}
template <typename T>
void argmin_sycl(const Tensor& a, Tensor& res, int axis, sycl::queue& q) {
    const T* a_ptr = static_cast<const T*>(a.data());
    int32_t* out_ptr = static_cast<int32_t*>(res.data());

    const auto& in_shape = a.shape();
    int ndim = in_shape.size();
    int axis_size = in_shape[axis];
    size_t outer_size = 1;
    size_t inner_size = 1;
    for (int i = 0; i < axis; ++i) {
        outer_size *= in_shape[i];
    }
    for (int i = axis + 1; i < ndim; ++i) {
        inner_size *= in_shape[i];
    }
    size_t total = outer_size * inner_size;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
            size_t id = idx[0];
            size_t outer_idx = id / inner_size;
            size_t inner_idx = id % inner_size;
            T min_val = a_ptr[outer_idx * axis_size * inner_size + 0 * inner_size + inner_idx];
            int min_idx = 0;
            for (int i = 1; i < axis_size; ++i) {
                T val = a_ptr[outer_idx * axis_size * inner_size + i * inner_size + inner_idx];
                if (val < min_val) {
                    min_val = val;
                    min_idx = i;
                }
            }
            out_ptr[id] = min_idx;
        });
    }).wait();
}
template <typename T>
float sum_sycl(const Tensor& a, sycl::queue& q) {
    const T* a_ptr = static_cast<const T*>(a.data());
    const size_t size = a.numel();
    // 1. 配置工作组参数（根据设备特性优化）
    constexpr size_t work_group_size = 256;
    const size_t num_work_groups = (size + work_group_size - 1) / work_group_size;
    // 2. 分配临时存储（使用设备内存减少数据传输）
    auto partial_sums = sycl::malloc_device<float>(num_work_groups, q);
    // 3. 第一阶段：设备端并行归约
    auto event = q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>{num_work_groups * work_group_size, work_group_size},
            [=](sycl::nd_item<1> item) {
                float val = 0.0f;
                const size_t gid = item.get_global_id(0);
                if (gid < size) val = static_cast<float>(a_ptr[gid]);
                float group_sum = sycl::reduce_over_group(item.get_group(), val, sycl::plus<>());
                if (item.get_local_id(0) == 0) {
                    partial_sums[item.get_group(0)] = group_sum;
                }
            });
    });
    // 4. 第二阶段：设备端最终归约（避免主机端计算）
    float final_sum = 0.0f;
    if (num_work_groups > 1) {
        auto final_buffer = sycl::malloc_device<float>(1, q);
        q.submit([&](sycl::handler& h) {
            h.depends_on(event);
            h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
                float total = 0.0f;
                for (size_t i = 0; i < num_work_groups; ++i) {
                    total += partial_sums[i];
                }
                final_buffer[0] = total;
            });
        }).wait();
        q.memcpy(&final_sum, final_buffer, sizeof(float)).wait();
        sycl::free(final_buffer, q);
    } else {
        q.memcpy(&final_sum, partial_sums, sizeof(float)).wait();
    }
    // 5. 清理资源
    sycl::free(partial_sums, q);
    return final_sum;
}
template <typename T>
float min_sycl(const Tensor& a, sycl::queue& q) {
    const T* a_ptr = static_cast<const T*>(a.data());
    const size_t size = a.numel();
    // 1. 配置工作组参数（自动适配设备）
    const size_t work_group_size = std::min(q.get_device().get_info<sycl::info::device::max_work_group_size>(),256ul);
    const size_t num_work_groups = (size + work_group_size - 1) / work_group_size;
    // 2. 分配设备内存存储中间结果
    auto partial_mins = sycl::malloc_device<float>(num_work_groups, q);
    // 3.分组局部规约
    q.submit([&](sycl::handler &h){
        // sycl::local_accessor<float, 1> local_min(work_group_size, h);
        h.parallel_for(sycl::nd_range<1>{num_work_groups * work_group_size, work_group_size},[=](sycl::nd_item<1> item) {
            // 初始化局部最小值
            float thread_min = std::numeric_limits<float>::max();
            // 处理有效数据（带边界检查）
            const size_t global_id = item.get_global_id(0);
            if (global_id < size) {
                thread_min = static_cast<float>(a_ptr[global_id]);
            }
            // 工作组内归约（隐式同步）
            float group_min = sycl::reduce_over_group(item.get_group(),thread_min,sycl::minimum<float>());

            // 保存组内结果
            if (item.get_local_id() == 0) {
                partial_mins[item.get_group(0)] = group_min;
            }
        });
    }).wait();
   // 2. 全局归约
    float final_min = std::numeric_limits<float>::max();
    if (num_work_groups > 1) {
        auto final_buf = sycl::malloc_device<float>(1, q);
        q.submit([&](sycl::handler& h) {
            h.single_task([=]() {
                float min_val = partial_mins[0];
                for (size_t i = 1; i < num_work_groups; ++i) {
                    min_val = sycl::min(min_val, partial_mins[i]);
                }
                final_buf[0] = min_val;
            });
        }).wait();
        q.memcpy(&final_min, final_buf, sizeof(float)).wait();
        sycl::free(final_buf, q);
    } else {
        q.memcpy(&final_min, partial_mins, sizeof(float)).wait();
    }
    sycl::free(partial_mins, q);
    return final_min;
}
template <typename T>
T max_sycl(const Tensor& a, sycl::queue& q) {
    const T* a_ptr = static_cast<const T*>(a.data());
    const size_t size = a.numel();
    // 1. 配置工作组参数（自动适配设备）
    const size_t work_group_size = std::min(q.get_device().get_info<sycl::info::device::max_work_group_size>(),256ul);
    const size_t num_work_groups = (size + work_group_size - 1) / work_group_size;
    // 2. 分配设备内存存储中间结果
    auto partial_maxs = sycl::malloc_device<float>(num_work_groups, q);
    // 3.分组局部规约
    q.submit([&](sycl::handler &h){
        // sycl::local_accessor<float, 1> local_min(work_group_size, h);
        h.parallel_for(sycl::nd_range<1>{num_work_groups * work_group_size, work_group_size},[=](sycl::nd_item<1> item) {
            // 初始化局部最小值
            float thread_max = std::numeric_limits<float>::min();
            // 处理有效数据（带边界检查）
            const size_t global_id = item.get_global_id(0);
            if (global_id < size) {
                thread_max = static_cast<float>(a_ptr[global_id]);
            }
            // 工作组内归约（隐式同步）
            float group_max = sycl::reduce_over_group(item.get_group(),thread_max,sycl::maximum<float>());
            // 保存组内结果
            if (item.get_local_id() == 0) {
                partial_maxs[item.get_group(0)] = group_max;
            }
        });
    }).wait();
   // 2. 全局归约
    float final_max = std::numeric_limits<float>::min();
    if (num_work_groups > 1) {
        auto final_buf = sycl::malloc_device<float>(1, q);
        q.submit([&](sycl::handler& h) {
            h.single_task([=]() {
                float max_val = partial_maxs[0];
                for (size_t i = 1; i < num_work_groups; ++i) {
                    max_val = sycl::max(max_val, partial_maxs[i]);
                }
                final_buf[0] = max_val;
            });
        }).wait();
        q.memcpy(&final_max, final_buf, sizeof(float)).wait();
        sycl::free(final_buf, q);
    } else {
        q.memcpy(&final_max, partial_maxs, sizeof(float)).wait();
    }
    sycl::free(partial_maxs, q);
    return final_max;
}
template <typename T>
bool any_sycl(const Tensor& a,float val,size_t size, sycl::queue& q) {
    // 判断是否存在val元素
    const T* ptr = static_cast<const T*>(a.data());
    // 1. 配置工作组参数（自动适配设备）
    const size_t work_group_size = std::min(q.get_device().get_info<sycl::info::device::max_work_group_size>(),256ul);
    const size_t num_work_groups = (size + work_group_size - 1) / work_group_size;
    // 2. 分配设备内存存储中间结果
    auto partial_anys = sycl::malloc_device<int32_t>(num_work_groups, q);

    constexpr auto epsilon = []{
        if constexpr (std::is_same_v<T, float>) return 1e-5f;
        else if constexpr (std::is_same_v<T, double>) return 1e-9;
        else if constexpr (std::is_same_v<T, sycl::half>) return 1e-3f;
        else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) return 1e-2f;
        else return T{0}; // 整数类型
    }();

    q.submit([&](sycl::handler& h){
        h.parallel_for(sycl::nd_range<1>{num_work_groups * work_group_size, work_group_size},[=](sycl::nd_item<1> item){
            const size_t gid = item.get_global_id(0);
            const size_t group_id = item.get_group(0);
            // 本地判断
            bool local_match = false;
            if (gid < size) {
                if constexpr(std::is_integral_v<T>){
                    local_match = ptr[gid] == static_cast<T>(val);;
                }else{
                    float diff = sycl::fabs(static_cast<float>(ptr[gid]) - val);
                    local_match = diff <= epsilon;
                }
            }
            // 组内归约（any of local_match）
            bool group_result = sycl::reduce_over_group(item.get_group(), local_match, sycl::logical_or<>());
            // 写入中间结果
            if (item.get_local_id(0) == 0) {
                partial_anys[group_id] = static_cast<int32_t>(group_result);
            }
        });
    }).wait();
    // 4. Host 端读取中间结果并判断
    std::vector<int32_t> host_results(num_work_groups);
    q.memcpy(host_results.data(), partial_anys, sizeof(int32_t) * num_work_groups).wait();
    sycl::free(partial_anys, q);
    for (int32_t flag : host_results) {
        if (flag != 0) return true;
    }
    return false;
}
template <typename T>
bool all_sycl(const Tensor& a,float val,size_t size, sycl::queue& q) {
    // 判断是否存在val元素
    const T* ptr = static_cast<const T*>(a.data());
    // 1. 配置工作组参数（自动适配设备）
    const size_t work_group_size = std::min(q.get_device().get_info<sycl::info::device::max_work_group_size>(),256ul);
    const size_t num_work_groups = (size + work_group_size - 1) / work_group_size;
    // 2. 分配设备内存存储中间结果
    auto partial_anys = sycl::malloc_device<int32_t>(num_work_groups, q);

    constexpr auto epsilon = []{
        if constexpr (std::is_same_v<T, float>) return 1e-5f;
        else if constexpr (std::is_same_v<T, double>) return 1e-9;
        else if constexpr (std::is_same_v<T, sycl::half>) return 1e-3f;
        else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) return 1e-2f;
        else return T{0}; // 整数类型
    }();

    q.submit([&](sycl::handler& h){
        h.parallel_for(sycl::nd_range<1>{num_work_groups * work_group_size, work_group_size},[=](sycl::nd_item<1> item){
            const size_t gid = item.get_global_id(0);
            const size_t group_id = item.get_group(0);
            // 本地判断
            bool local_match = false;
            if (gid < size) {
                if constexpr(std::is_integral_v<T>){
                    local_match = ptr[gid] == static_cast<T>(val);;
                }else{
                    float diff = sycl::fabs(static_cast<float>(ptr[gid]) - val);
                    local_match = diff <= epsilon;
                }
            }
            // 组内归约（any of local_match）
            bool group_result = sycl::reduce_over_group(item.get_group(), local_match, sycl::logical_and<>());
            // 写入中间结果
            if (item.get_local_id(0) == 0) {
                partial_anys[group_id] = static_cast<int32_t>(group_result);
            }
        });
    }).wait();
    // 4. Host 端读取中间结果并判断
    std::vector<int32_t> host_results(num_work_groups);
    q.memcpy(host_results.data(), partial_anys, sizeof(int32_t) * num_work_groups).wait();
    sycl::free(partial_anys, q);
    for (int32_t flag : host_results) {
        if (flag != 0) return true;
    }
    return false;
}

//**************************************************

//**************************************************
float SumImpl<Device::SYCL>::execute(const Tensor& a){
    auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
    auto& q = ctx_impl->get_queue();
    auto A = data_as_const_variant(a.dtype(),a.data());
    float res = 0.0f;
    std::visit([&](auto ptr_A){
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        res = sum_sycl<AType>(a,q);
    },A);
    return res;
}
Tensor SumImpl<Device::SYCL>::execute(const Tensor& a,int axis){
    auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
    auto& q = ctx_impl->get_queue();
    // 移除 a.shape(axis) 所在的轴
    std::vector<int> new_shape;
    for (int i = 0; i < a.shape().size(); i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    Tensor result(new_shape,a.dtype(),Device::SYCL);
    switch (a.dtype()) {
        case DataType::INT8:     sum_sycl<int8_t>(a,result, axis,q); break;
        case DataType::INT16:    sum_sycl<int16_t>(a,result,axis, q); break;
        case DataType::INT32:    sum_sycl<int32_t>(a,result,axis, q); break;
        case DataType::INT64:    sum_sycl<int64_t>(a,result, axis,q); break;
        case DataType::FLOAT16:  sum_sycl<float16>(a,result, axis,q); break;
        case DataType::FLOAT32:  sum_sycl<float32>(a,result, axis,q); break;
        case DataType::FLOAT64:  sum_sycl<float64>(a,result,axis, q); break;
        case DataType::BFLOAT16: sum_sycl<bfloat16>(a,result,axis, q); break;
        default: throw std::runtime_error("sum: unsupported data type");
    }
    return result;
}
float MeanImpl<Device::SYCL>::execute(const Tensor& a) {
    auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
    auto& q = ctx_impl->get_queue();

    float sum_val = 0.0f;
    switch (a.dtype()) {
        case DataType::INT8:     sum_val =  sum_sycl<int8_t>(a, q);
        case DataType::INT16:    sum_val =  sum_sycl<int16_t>(a, q);
        case DataType::INT32:    sum_val =  sum_sycl<int32_t>(a, q);
        case DataType::INT64:    sum_val =  sum_sycl<int64_t>(a, q);
        case DataType::FLOAT16:  sum_val =  sum_sycl<float16>(a, q);
        case DataType::BFLOAT16: sum_val =  sum_sycl<bfloat16>(a, q);
        case DataType::FLOAT32:  sum_val =  sum_sycl<float32>(a, q);
        case DataType::FLOAT64:  sum_val =  sum_sycl<float64>(a, q);
        default: throw std::runtime_error("mean: unsupported data type");
    }
    return sum_val / a.numel();
}
Tensor MeanImpl<Device::SYCL>::execute(const Tensor& a,int axis){
    auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
    auto& q = ctx_impl->get_queue();
    // 移除 a.shape(axis) 所在的轴
    std::vector<int> new_shape;
    for (int i = 0; i < a.shape().size(); i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    Tensor result(new_shape,a.dtype(),Device::SYCL);
    switch (a.dtype()) {
        case DataType::INT8:     mean_sycl<int8_t>(a,result, axis,q); break;
        case DataType::INT16:    mean_sycl<int16_t>(a,result,axis, q); break;
        case DataType::INT32:    mean_sycl<int32_t>(a,result,axis, q); break;
        case DataType::INT64:    mean_sycl<int64_t>(a,result, axis,q); break;
        case DataType::FLOAT16:  mean_sycl<float16>(a,result, axis,q); break;
        case DataType::FLOAT32:  mean_sycl<float>(a,result, axis,q); break;
        case DataType::FLOAT64:  mean_sycl<float64>(a,result,axis, q); break;
        case DataType::BFLOAT16: mean_sycl<bfloat16>(a,result,axis, q); break;
        default: throw std::runtime_error("mean: unsupported data type");
    }
    return result;
}
float MinImpl<Device::SYCL>::execute(const Tensor& a){
    auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
    auto& q = ctx_impl->get_queue();
    switch (a.dtype()) {
        case DataType::INT8:     return min_sycl<int8_t>(a, q);
        case DataType::INT16:    return min_sycl<int16_t>(a, q);
        case DataType::INT32:    return min_sycl<int32_t>(a, q);
        case DataType::INT64:    return min_sycl<int64_t>(a, q);
        case DataType::FLOAT16:  return min_sycl<float16>(a, q);
        case DataType::BFLOAT16: return min_sycl<bfloat16>(a, q);
        case DataType::FLOAT32:  return min_sycl<float>(a, q);
        case DataType::FLOAT64:  return min_sycl<float64>(a, q);
        default: throw std::runtime_error("min: unsupported data type");
    }
}
Tensor MinImpl<Device::SYCL>::execute(const Tensor& a,int axis){
    auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
    auto& q = ctx_impl->get_queue();
    // 移除 a.shape(axis) 所在的轴
    std::vector<int> new_shape;
    for (int i = 0; i < a.shape().size(); i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    Tensor result(new_shape,a.dtype(),Device::SYCL);
    switch (a.dtype()) {
        case DataType::INT8:     min_sycl<int8_t>(a,result, axis,q); break;
        case DataType::INT16:    min_sycl<int16_t>(a,result,axis, q); break;
        case DataType::INT32:    min_sycl<int32_t>(a,result,axis, q); break;
        case DataType::INT64:    min_sycl<int64_t>(a,result, axis,q); break;
        case DataType::FLOAT16:  min_sycl<float16>(a,result, axis,q); break;
        case DataType::FLOAT32:  min_sycl<float>(a,result, axis,q); break;
        case DataType::FLOAT64:  min_sycl<float64>(a,result,axis, q); break;
        case DataType::BFLOAT16: min_sycl<bfloat16>(a,result,axis, q); break;
        default: throw std::runtime_error("Min: unsupported data type");
    }
    return result;
}
float MaxImpl<Device::SYCL>::execute(const Tensor& a){
    auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
    auto& q = ctx_impl->get_queue();
    switch (a.dtype()) {
        case DataType::INT8:     return max_sycl<int8_t>(a, q);
        case DataType::INT16:    return max_sycl<int16_t>(a, q);
        case DataType::INT32:    return max_sycl<int32_t>(a, q);
        case DataType::INT64:    return max_sycl<int64_t>(a, q);
        case DataType::FLOAT16:  return max_sycl<float16>(a, q);
        case DataType::BFLOAT16: return max_sycl<bfloat16>(a, q);
        case DataType::FLOAT32:  return max_sycl<float>(a, q);
        case DataType::FLOAT64:  return max_sycl<float64>(a, q);
        default: throw std::runtime_error("max: unsupported data type");
    }
}
Tensor MaxImpl<Device::SYCL>::execute(const Tensor& a,int axis){
    auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
    auto& q = ctx_impl->get_queue();
    // 移除 a.shape(axis) 所在的轴
    std::vector<int> new_shape;
    for (int i = 0; i < a.shape().size(); i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    Tensor result(new_shape,a.dtype(),Device::SYCL);
    switch (a.dtype()) {
        case DataType::INT8:     max_sycl<int8_t>(a,result, axis,q); break;
        case DataType::INT16:    max_sycl<int16_t>(a,result,axis, q); break;
        case DataType::INT32:    max_sycl<int32_t>(a,result,axis, q); break;
        case DataType::INT64:    max_sycl<int64_t>(a,result, axis,q); break;
        case DataType::FLOAT16:  max_sycl<float16>(a,result, axis,q); break;
        case DataType::FLOAT32:  max_sycl<float>(a,result, axis,q); break;
        case DataType::FLOAT64:  max_sycl<float64>(a,result,axis, q); break;
        case DataType::BFLOAT16: max_sycl<bfloat16>(a,result,axis, q); break;
        default: throw std::runtime_error("max: unsupported data type");
    }
    return result;
}

Tensor ArgMaxImpl<Device::SYCL>::execute(const Tensor &a, int axis){
    auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
    auto& q = ctx_impl->get_queue();
    // 移除 a.shape(axis) 所在的轴
    std::vector<int> new_shape;
    for (int i = 0; i < a.shape().size(); i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    Tensor result(new_shape,DataType::INT32,Device::SYCL);
    switch (a.dtype()) {
        case DataType::INT8:     argmax_sycl<int8_t>(a,result, axis,q); break;
        case DataType::INT16:    argmax_sycl<int16_t>(a,result,axis, q); break;
        case DataType::INT32:    argmax_sycl<int32_t>(a,result,axis, q); break;
        case DataType::INT64:    argmax_sycl<int64_t>(a,result, axis,q); break;
        case DataType::FLOAT16:  argmax_sycl<float16>(a,result, axis,q); break;
        case DataType::FLOAT32:  argmax_sycl<float>(a,result, axis,q); break;
        case DataType::FLOAT64:  argmax_sycl<float64>(a,result,axis, q); break;
        case DataType::BFLOAT16: argmax_sycl<bfloat16>(a,result,axis, q); break;
        default: throw std::runtime_error("slice: unsupported data type");
    }
    return result;
}

Tensor ArgMinImpl<Device::SYCL>::execute(const Tensor &a, int axis) {
    auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
    auto& q = ctx_impl->get_queue();
    std::vector<int> new_shape;
    for (int i = 0; i < a.shape().size(); ++i) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    Tensor result(new_shape, DataType::INT32, Device::SYCL);
    switch (a.dtype()) {
        case DataType::INT8:     argmin_sycl<int8_t>(a, result, axis, q); break;
        case DataType::INT16:    argmin_sycl<int16_t>(a, result, axis, q); break;
        case DataType::INT32:    argmin_sycl<int32_t>(a, result, axis, q); break;
        case DataType::INT64:    argmin_sycl<int64_t>(a, result, axis, q); break;
        case DataType::FLOAT16:  argmin_sycl<float16>(a, result, axis, q); break;
        case DataType::FLOAT32:  argmin_sycl<float>(a, result, axis, q); break;
        case DataType::FLOAT64:  argmin_sycl<float64>(a, result, axis, q); break;
        case DataType::BFLOAT16: argmin_sycl<bfloat16>(a, result, axis, q); break;
        default: throw std::runtime_error("argmin: unsupported data type");
    }
    return result;
}
bool AnyImpl<Device::SYCL>::execute(const Tensor& a,float val) {
        auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();

        size_t size = a.numel();
        switch (a.dtype()) {
            case DataType::INT8:     return any_sycl<int8_t>(a,val,size, q);
            case DataType::INT16:    return any_sycl<int16_t>(a,val, size, q);
            case DataType::INT32:    return any_sycl<int32_t>(a,val, size, q);
            case DataType::INT64:    return any_sycl<int64_t>(a,val, size, q);
            case DataType::FLOAT16:  return any_sycl<float16>(a, val,size, q);
            case DataType::BFLOAT16: return any_sycl<bfloat16>(a,val, size, q);
            case DataType::FLOAT32:  return any_sycl<float32>(a,val, size, q);
            case DataType::FLOAT64:  return any_sycl<float64>(a, val,size, q);
            default: throw std::runtime_error("any: unsupported dtype");
        }
    }
bool AllImpl<Device::SYCL>::execute(const Tensor& a,float val) {
    auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
    auto& q = ctx_impl->get_queue();

    size_t size = a.numel();
    switch (a.dtype()) {
        case DataType::INT8:     return all_sycl<int8_t>(a, val,size, q);
        case DataType::INT16:    return all_sycl<int16_t>(a,val, size, q);
        case DataType::INT32:    return all_sycl<int32_t>(a,val, size, q);
        case DataType::INT64:    return all_sycl<int64_t>(a,val, size, q);
        case DataType::FLOAT16:  return all_sycl<float16>(a,val, size, q);
        case DataType::BFLOAT16: return all_sycl<bfloat16>(a,val, size, q);
        case DataType::FLOAT32:  return all_sycl<float32>(a,val, size, q);
        case DataType::FLOAT64:  return all_sycl<float64>(a,val, size, q);
        default: throw std::runtime_error("all: unsupported dtype");
    }
}

 template struct SumImpl<Device::SYCL>;
 template struct MeanImpl<Device::SYCL>;
 template struct MinImpl<Device::SYCL>;
 template struct MaxImpl<Device::SYCL>;
 template struct ArgMaxImpl<Device::SYCL>;
 template struct ArgMinImpl<Device::SYCL>;
 template struct AnyImpl<Device::SYCL>;
 template struct AllImpl<Device::SYCL>;

}