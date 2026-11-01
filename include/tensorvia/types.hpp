#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// CUDA lane 钉软件类型：nvcc 的 C++20 前端没有 std::float16_t，而 host TU
// 可能是 C++23 且有。`float16`/`bfloat16` 在 BACKEND_CUDA 构建里（host 与
// 设备单元共享本头）统一为 ops::float16_t/ops::bfloat16_t，保一个 ABI。
// `<stdfloat>` 只在编译器预定义特性宏时声明类型（GCC 会，clang 不会），
// 所以光靠 __has_include 判断不够。
#if defined(BACKEND_CUDA)
    #include "float16.hpp"
    #include "bfloat16.hpp"
    #define TENSORVIA_HAS_STDFLOAT 0
#elif !defined(_MSC_VER) && __has_include(<stdfloat>) \
    && defined(__STDCPP_FLOAT16_T__) && defined(__STDCPP_BFLOAT16_T__)
    #include <stdfloat>
    #define TENSORVIA_HAS_STDFLOAT 1
#else
    #include "float16.hpp"
    #include "bfloat16.hpp"
    #define TENSORVIA_HAS_STDFLOAT 0
#endif


using float32 = float;
using float64 = double;
#if TENSORVIA_HAS_STDFLOAT
    using float16  = std::float16_t;
    using bfloat16 = std::bfloat16_t;
#else
    using float16  = ops::float16_t;
    using bfloat16 = ops::bfloat16_t;
#endif

// 张量维度数硬上限：超过属于极端场景，直接拒绝（错误信息明确）。
// `inline` 给常量外部链接，头文件式 facade 需要用 `export using` 重新导出它。
inline constexpr size_t kMaxTensorDims = 8;

// 内联固定容量的维度数组：栈上直接容纳 8 维，消除每个 Tensor 的
// shape/strides 两次堆分配（视图创建则省 4 次）。接口对齐
// std::vector<int64_t> 的常用子集，便于存量代码迁移。
struct DimArray {
    int64_t m_data[kMaxTensorDims] = {};
    size_t  m_size = 0;

    DimArray() = default;
    explicit DimArray(size_t n) { resize(n); }
    DimArray(std::initializer_list<int64_t> il) { assign(il.begin(), il.end()); }
    DimArray(const std::vector<int64_t>& v) { assign(v.begin(), v.end()); }

    DimArray& operator=(const std::vector<int64_t>& v) {
        assign(v.begin(), v.end());
        return *this;
    }
    DimArray& operator=(std::initializer_list<int64_t> il) {
        assign(il.begin(), il.end());
        return *this;
    }

    template <typename It>
    void assign(It first, It last) {
        m_size = 0;
        for (; first != last; ++first) push_back(*first);
    }

    size_t size() const { return m_size; }
    bool   empty() const { return m_size == 0; }
    int64_t*       data()       { return m_data; }
    const int64_t* data() const { return m_data; }
    int64_t*       begin()       { return m_data; }
    int64_t*       end()         { return m_data + m_size; }
    const int64_t* begin() const { return m_data; }
    const int64_t* end()   const { return m_data + m_size; }

    int64_t&       operator[](size_t i)       { return m_data[i]; }
    const int64_t& operator[](size_t i) const { return m_data[i]; }
    int64_t&       back()       { return m_data[m_size - 1]; }
    const int64_t& back() const { return m_data[m_size - 1]; }
    int64_t&       front()       { return m_data[0]; }
    const int64_t& front() const { return m_data[0]; }

    void push_back(int64_t v) {
        if (m_size >= kMaxTensorDims)
            throw std::runtime_error("tensor dimension count exceeds the limit of 8");
        m_data[m_size++] = v;
    }
    void pop_back() { if (m_size) --m_size; }
    void clear() { m_size = 0; }
    void resize(size_t n) {
        if (n > kMaxTensorDims)
            throw std::runtime_error("tensor dimension count exceeds the limit of 8");
        for (size_t i = m_size; i < n; ++i) m_data[i] = 0;
        m_size = n;
    }
    int64_t* insert(const int64_t* pos, int64_t v) {
        const size_t idx = size_t(pos - m_data);
        if (m_size >= kMaxTensorDims)
            throw std::runtime_error("tensor dimension count exceeds the limit of 8");
        for (size_t i = m_size; i > idx; --i) m_data[i] = m_data[i - 1];
        m_data[idx] = v;
        ++m_size;
        return m_data + idx;
    }
    int64_t* erase(const int64_t* pos) {
        const size_t idx = size_t(pos - m_data);
        for (size_t i = idx; i + 1 < m_size; ++i) m_data[i] = m_data[i + 1];
        if (m_size) --m_size;
        return m_data + idx;
    }

    bool operator==(const DimArray& o) const {
        return m_size == o.m_size && std::equal(begin(), end(), o.begin());
    }
};

template <typename T>
struct compute_type_helper { using type = T; };
template <>
struct compute_type_helper<float16>  { using type = float; };
template <>
struct compute_type_helper<bfloat16> { using type = float; };
template <typename T>
using compute_type_t = typename compute_type_helper<T>::type;

enum class OpType {
    Fill, Random,
    Slice,
    Matmul,
    Add, Sub, Mul, Div,
    AddVec, SubVec, MulVec, DivVec,
    Abs, Repack,
    Sin, Cos, Tan, Exp, Relu, Silu, Tanh, Sqrt, Sigmoid, Pow, Log, Softmax,
    Clamp, Sum, SumVec, Min, MinVec, Max, MaxVec, Mean, Prod, ProdVec, Concat,
    Transpose2d, TransposeNd, Equal, NotEqual, Greater, Less, GreaterEqual, LessEqual,
    All, Any, Nonzero, Argmax, Argmin,
    // 条件选择、索引与累积和。也只有 Vulkan 需要把它们写进流水线键：其余后端
    // 直接跑实现，不查表。
    Where, IndexSelect, Gather, Cumsum, Conv2d
};

enum class Device { CPU, CUDA, SYCL, VULKAN };

// 运行期的设备身份：种类 + 序号。种类仍然决定编译期派发（`Impl<Device::D>`），
// 序号只在运行期挑后端实例——第几张卡。没有从 `Device` 到 `DeviceId` 的隐式
// 转换：序号是有意义的信息，不该被一个隐式的 0 吞掉。
struct DeviceId {
    Device kind  = Device::CPU;
    int    index = 0;

    constexpr DeviceId() = default;
    constexpr DeviceId(Device k, int i) : kind(k), index(i) {}

    friend constexpr bool operator==(const DeviceId&, const DeviceId&) = default;
};

struct DeviceIdHash {
    size_t operator()(const DeviceId& id) const noexcept {
        return static_cast<size_t>(id.kind) * 31u + static_cast<size_t>(id.index);
    }
};

// BOOL 是一等 dtype：每元素 1 字节（uint8_t），值规范化为 0/1，绝不做 bit-packing。
// 它的枚举值排在有符号整数之前，让包含它的序号比较继续成立；但类型提升与算术
// 门禁一律用下面的类别判定函数，不看序号。
enum class DataType {
    BOOL = 0,
    INT8,
    INT16,
    INT32,
    INT64,
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    FLOAT64
};

constexpr bool is_bool_dtype(DataType t) { return t == DataType::BOOL; }

// 只包含有符号整型（BOOL 由契约单独处理）。
constexpr bool is_integer_dtype(DataType t) {
    return t == DataType::INT8 || t == DataType::INT16 ||
           t == DataType::INT32 || t == DataType::INT64;
}

constexpr bool is_floating_dtype(DataType t) {
    return t == DataType::BFLOAT16 || t == DataType::FLOAT16 ||
           t == DataType::FLOAT32 || t == DataType::FLOAT64;
}

// BOOL 加有符号整型：NumPy 眼中的 integral 类型。
constexpr bool is_integral_dtype(DataType t) {
    return is_bool_dtype(t) || is_integer_dtype(t);
}

// 库级默认浮点 dtype。与 current device 一样是线程私有的：一个线程修改它不能影响
// 其它线程的工厂。存储槽定义在 src/core/dtype.cpp（facade 生成器把公共头里开出的
// 每个命名空间都视为粘性的，所以这里不引入辅助命名空间）。
DataType default_dtype();
void set_default_dtype(DataType dtype);

// 调用方未指定设备时新张量落到的设备。与上面的默认值一样按线程私有；初值为 CPU:0。
// 只带 dtype 的工厂经它解析，`TensorOptions{}` 也从它起步。
DeviceId default_device();
void set_default_device(DeviceId id);

// 工厂描述“造什么”的载体：shape 单独传，其余全在这里（FR-7）。
// `device_index == -1` 表示 `device` 后端上调用线程的 current device——
// 裸 `Device` 本来就是这层意思。
struct TensorOptions {
    // 默认构造的 options 携带进程默认值（FR-7）：dtype 取 default_dtype()，
    // 实例取 default_device()。只给 `Device` 不给索引时沿用“线程 current device”规则。
    DataType dtype = default_dtype();
    Device   device = default_device().kind;
    int      device_index = default_device().index;

    TensorOptions() = default;
    explicit TensorOptions(DataType dt) : dtype(dt) {}
    explicit TensorOptions(Device dev, int index = -1) : device(dev), device_index(index) {}
    TensorOptions(DataType dt, Device dev, int index = -1)
        : dtype(dt), device(dev), device_index(index) {}

    TensorOptions& with_dtype(DataType dt) { dtype = dt; return *this; }
    TensorOptions& with_device(Device dev, int index = -1) { device = dev; device_index = index; return *this; }
    TensorOptions& with_device(DeviceId id) { device = id.kind; device_index = id.index; return *this; }

    DeviceId resolved_id() const;  // device_index < 0 时取 `device` 的 current device
};

inline const char* op_to_string(OpType op) {
    switch (op) {
        case OpType::AddVec:        return "addvec";
        case OpType::SubVec:        return "subvec";
        case OpType::MulVec:        return "mulvec";
        case OpType::DivVec:        return "divvec";
        case OpType::Fill:          return "fill";
        case OpType::Random:        return "random";
        case OpType::Slice:         return "slice";
        case OpType::Matmul:        return "matmul";
        case OpType::Add:           return "add";
        case OpType::Sub:           return "sub";
        case OpType::Mul:           return "mul";
        case OpType::Div:           return "div";
        case OpType::Abs:           return "abs";
        case OpType::Sin:           return "sin";
        case OpType::Cos:           return "cos";
        case OpType::Tan:           return "tan";
        case OpType::Exp:           return "exp";
        case OpType::Relu:          return "relu";
        case OpType::Silu:          return "silu";
        case OpType::Tanh:          return "tanh";
        case OpType::Sigmoid:       return "sigmoid";
        case OpType::Softmax:       return "softmax";
        case OpType::Pow:           return "pow";
        case OpType::Log:           return "log";
        case OpType::Sqrt:          return "sqrt";
        case OpType::Clamp:         return "clamp";
        case OpType::Sum:           return "sum";
        case OpType::SumVec:        return "sumvec";
        case OpType::Min:           return "min";
        case OpType::MinVec:        return "minvec";
        case OpType::Max:           return "max";
        case OpType::MaxVec:        return "maxvec";
        case OpType::Mean:          return "mean";
        case OpType::Prod:          return "prod";
        case OpType::ProdVec:       return "prodvec";
        case OpType::Concat:        return "concat";
        case OpType::Transpose2d:   return "transpose2d";
        case OpType::TransposeNd:   return "transposeNd";
        case OpType::Repack:        return "repack";
        case OpType::Equal:         return "equal";
        case OpType::NotEqual:      return "not_equal";
        case OpType::Greater:       return "greater";
        case OpType::Less:          return "less";
        case OpType::GreaterEqual:  return "greater_equal";
        case OpType::LessEqual:     return "less_equal";
        case OpType::All:           return "all";
        case OpType::Any:           return "any";
        case OpType::Nonzero:       return "nonzero";
        case OpType::Argmax:        return "argmax";
        case OpType::Argmin:        return "argmin";
        case OpType::Where:         return "where";
        case OpType::IndexSelect:   return "index_select";
        case OpType::Gather:        return "gather";
        case OpType::Cumsum:        return "cumsum";
        case OpType::Conv2d:        return "conv2d";
        default: throw std::invalid_argument("Unknown OpType");
    }
}

constexpr std::string_view device_to_string(Device d) {
    switch (d) {
        case Device::CPU:    return "CPU";
        case Device::CUDA:   return "CUDA";
        case Device::SYCL:   return "SYCL";
        case Device::VULKAN: return "VULKAN";
        default:             return "UNKNOWN";
    }
}

// "CUDA:0" —— 所有跨设备错误信息都用它，因此不匹配时能点名实例而不只是后端。
inline std::string device_id_to_string(const DeviceId& id) {
    return std::format("{}:{}", device_to_string(id.kind), id.index);
}

constexpr std::string_view dtype_to_string(DataType t) {
    switch (t) {
        case DataType::BOOL:     return "bool";
        case DataType::INT8:     return "int8";
        case DataType::INT16:    return "int16";
        case DataType::INT32:    return "int32";
        case DataType::INT64:    return "int64";
        case DataType::FLOAT16:  return "float16";
        case DataType::FLOAT32:  return "float32";
        case DataType::FLOAT64:  return "float64";
        case DataType::BFLOAT16: return "bfloat16";
        default:                 return "UNKNOWN";
    }
}

inline std::string make_pipeline_key(OpType op, DataType dt) {
    return std::format("{}_{}", op_to_string(op), dtype_to_string(dt));
}

// Sum/Prod 的累加/输出 dtype（PyTorch 的 reduction_type）：整型一律按 int64 累加；
// f16/bf16 提升到 float32；宽浮点保持自身 dtype。
constexpr DataType reduction_type(DataType dtype) {
    switch (dtype) {
        case DataType::BOOL:
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::INT64:    return DataType::INT64;
        case DataType::BFLOAT16:
        case DataType::FLOAT16:
        case DataType::FLOAT32:  return DataType::FLOAT32;
        case DataType::FLOAT64:  return DataType::FLOAT64;
        default: throw std::runtime_error("Unsupported DataType");
    }
}

// Mean 的输出 dtype：NumPy 语义——整型输入变成 FLOAT32，只有 FLOAT64 保持 FLOAT64。
// BOOL 没有 Mean（契约在到达这个函数之前就拒绝了）。
constexpr DataType float_reduction_type(DataType dtype) {
    switch (dtype) {
        case DataType::BOOL:
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::INT64:
        case DataType::BFLOAT16:
        case DataType::FLOAT16:
        case DataType::FLOAT32:  return DataType::FLOAT32;
        case DataType::FLOAT64:  return DataType::FLOAT64;
        default: throw std::runtime_error("Unsupported DataType");
    }
}

inline size_t calc_dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::BOOL:     return 1;
        case DataType::INT8:     return 1;
        case DataType::INT16:    return 2;
        case DataType::INT32:    return 4;
        case DataType::INT64:    return 8;
        case DataType::FLOAT16:  return 2;
        case DataType::FLOAT32:  return 4;
        case DataType::FLOAT64:  return 8;
        case DataType::BFLOAT16: return 2;
        default: throw std::runtime_error("Unsupported DataType");
    }
}

using CppDTypeVariant = std::variant<
    uint8_t*, int8_t*, int16_t*, int32_t*, int64_t*,
    float16*, bfloat16*, float*, double*
>;
using ConstCppDTypeVariant = std::variant<
    const uint8_t*,
    const int8_t*,
    const int16_t*,
    const int32_t*,
    const int64_t*,
    const float16*,
    const bfloat16*,
    const float32*,
    const float64*
>;

template <typename T>
ConstCppDTypeVariant make_variant(const void* src) {
    ConstCppDTypeVariant var;
    var.emplace<const T*>(static_cast<const T*>(src));
    return var;
}

inline ConstCppDTypeVariant data_as_const_variant(DataType type, const void* src) {
    switch (type) {
        case DataType::BOOL:     return make_variant<uint8_t>(src);
        case DataType::INT8:     return make_variant<int8_t>(src);
        case DataType::INT16:    return make_variant<int16_t>(src);
        case DataType::INT32:    return make_variant<int32_t>(src);
        case DataType::INT64:    return make_variant<int64_t>(src);
        case DataType::FLOAT16:  return make_variant<float16>(src);
        case DataType::BFLOAT16: return make_variant<bfloat16>(src);
        case DataType::FLOAT32:  return make_variant<float32>(src);
        case DataType::FLOAT64:  return make_variant<float64>(src);
        default: throw std::runtime_error("Unsupported dtype for variant");
    }
}

template <typename Func>
void dispatch_dtype(DataType type, Func&& f) {
    switch (type) {
        case DataType::BOOL:     f(std::type_identity<uint8_t>{}); break;
        case DataType::INT8:     f(std::type_identity<int8_t>{}); break;
        case DataType::INT16:    f(std::type_identity<int16_t>{}); break;
        case DataType::INT32:    f(std::type_identity<int32_t>{}); break;
        case DataType::INT64:    f(std::type_identity<int64_t>{}); break;
        case DataType::FLOAT16:  f(std::type_identity<float16>{}); break;
        case DataType::BFLOAT16: f(std::type_identity<bfloat16>{}); break;
        case DataType::FLOAT32:  f(std::type_identity<float>{}); break;
        case DataType::FLOAT64:  f(std::type_identity<double>{}); break;
        default: throw std::runtime_error("Unsupported dtype");
    }
}

inline const std::set<std::pair<DataType, DataType>>& valid_casts() {
    // PyTorch 的 `.to(dtype)` 接受任意 dtype 对，包括 BOOL <-> 数值。
    // 仍以这个集合为唯一事实来源，保持公开助手的接口形状，但改为穷举生成而不是
    // 手工罗列。
    static const std::set<std::pair<DataType, DataType>> casts = [] {
        constexpr DataType all[] = {
            DataType::BOOL,     DataType::INT8,     DataType::INT16,
            DataType::INT32,    DataType::INT64,    DataType::BFLOAT16,
            DataType::FLOAT16,  DataType::FLOAT32,  DataType::FLOAT64};
        std::set<std::pair<DataType, DataType>> out;
        for (DataType a : all)
            for (DataType b : all) out.emplace(a, b);
        return out;
    }();
    return casts;
}

// PyTorch 类型提升规则：
//   * BOOL 与任何 dtype 相遇都是对方（bool + bool 仍是 bool）；
//   * 整型 + 整型 -> 更宽的整型；
//   * 浮点 + 浮点 -> 更宽的浮点，但 bfloat16 + float16 -> float32；
//   * 整型 + 浮点 -> 浮点 dtype。
inline DataType compute_type(DataType type0, DataType type1) {
    if (type0 == type1) return type0;
    if (is_bool_dtype(type0)) return type1;
    if (is_bool_dtype(type1)) return type0;

    const bool f0 = is_floating_dtype(type0);
    const bool f1 = is_floating_dtype(type1);
    if (f0 && f1) {
        const bool low0 = type0 == DataType::BFLOAT16 || type0 == DataType::FLOAT16;
        const bool low1 = type1 == DataType::BFLOAT16 || type1 == DataType::FLOAT16;
        if (low0 && low1) return DataType::FLOAT32;
        return type0 > type1 ? type0 : type1;
    }
    if (f0) return type0;
    if (f1) return type1;
    return type0 > type1 ? type0 : type1;
}

// 张量与“类 Python float 标量”运算的结果 dtype。PyTorch 把这种标量视为弱类型数值：
// 浮点张量保持自身 dtype，整型或 BOOL 张量则采用默认浮点 dtype。
inline DataType scalar_promote_type(DataType tensor_dtype) {
    return is_floating_dtype(tensor_dtype) ? tensor_dtype : default_dtype();
}

inline bool is_cast_valid(DataType src, DataType dst) {
    return valid_casts().contains({src, dst});
}

// 长度为 0 的维合法（空张量）；负数不合法的。
inline size_t calc_numel(const std::vector<int64_t>& shape) {
    size_t numel = 1;
    for (int64_t dim : shape) {
        if (dim < 0) throw std::runtime_error("Invalid tensor dimension");
        numel *= size_t(dim);
    }
    return numel;
}

inline size_t calc_numel(const DimArray& shape) {
    size_t numel = 1;
    for (int64_t dim : shape) {
        if (dim < 0) throw std::runtime_error("Invalid tensor dimension");
        numel *= size_t(dim);
    }
    return numel;
}

inline size_t calc_numel(std::initializer_list<int64_t> shape) {
    size_t numel = 1;
    for (int64_t dim : shape) {
        if (dim < 0) throw std::runtime_error("Invalid tensor dimension");
        numel *= dim;
    }
    return numel;
}

#if !defined(RESTRICT)
    #if defined(_MSC_VER)
        #define RESTRICT __restrict
    #elif defined(__GNUC__) || defined(__clang__)
        #define RESTRICT __restrict__
    #else
        #define RESTRICT
    #endif
#endif
