#pragma once

#include <algorithm>
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

// Pick native std::float16_t / std::bfloat16_t when the toolchain ships <stdfloat>
// AND we're not on MSVC (which historically lacks software-emulated std::float16_t).
// Otherwise fall back to the bundled software-emulated ops::float16_t / ops::bfloat16_t.
// `<stdfloat>` only declares std::float16_t/std::bfloat16_t when the compiler
// predefines the corresponding feature macros (GCC does, clang does not), so
// checking __has_include alone is not enough.
//
// TENSORVIA_FORCE_SOFTWARE_FLOAT16 is baked into the generated types header of
// the header-first (CUDA) lane: nvcc's C++20 front end has no std::float16_t,
// while the host TUs of the same artifact may compile as C++23. Pinning the
// software type in the shared header keeps one ABI across the whole artifact.
#if defined(TENSORVIA_FORCE_SOFTWARE_FLOAT16) && TENSORVIA_FORCE_SOFTWARE_FLOAT16
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


{
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
// `inline` gives the constant external linkage, which the header-first facade
// needs in order to re-export it with `export using`.
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
    Clamp, Sum, SumVec, Min, MinVec, Max, MaxVec, Mean, Concat,
    Transpose2d, TransposeNd, Equal, NotEqual, Greater, Less, GreaterEqual, LessEqual,
    All, Any, Nonzero, Argmax, Argmin, Temp
};

enum class Device { CPU, CUDA, SYCL, VULKAN };

enum class DataType {
    INT8 = 0,
    INT16,
    INT32,
    INT64,
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    FLOAT64
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
        case OpType::Temp:          return "temp";
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

constexpr std::string_view dtype_to_string(DataType t) {
    switch (t) {
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

// Accumulator / output dtype for full reductions. The Vulkan shaders promote
// block-local sums like this; the CPU side promotes inline in the kernels.
//   int8/int16 -> int32, int32/int64 -> int64, f16/bf16 -> float32.
constexpr DataType reduction_type(DataType dtype) {
    switch (dtype) {
        case DataType::INT8:
        case DataType::INT16:    return DataType::INT32;
        case DataType::INT32:
        case DataType::INT64:    return DataType::INT64;
        case DataType::BFLOAT16:
        case DataType::FLOAT16:
        case DataType::FLOAT32:  return DataType::FLOAT32;
        case DataType::FLOAT64:  return DataType::FLOAT64;
        default: throw std::runtime_error("Unsupported DataType");
    }
}

// Output dtype for Mean: NumPy semantics, so integer inputs become FLOAT32 and
// only FLOAT64 stays FLOAT64.
constexpr DataType float_reduction_type(DataType dtype) {
    switch (dtype) {
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
    int8_t*, int16_t*, int32_t*, int64_t*,
    float16*, bfloat16*, float*, double*
>;
using ConstCppDTypeVariant = std::variant<
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
    static const std::set<std::pair<DataType, DataType>> casts = {
        {DataType::INT8,     DataType::FLOAT32},
        {DataType::INT16,    DataType::FLOAT32},
        {DataType::INT16,    DataType::FLOAT64},
        {DataType::INT32,    DataType::FLOAT32},
        {DataType::INT32,    DataType::FLOAT64},
        {DataType::INT64,    DataType::FLOAT64},
        {DataType::FLOAT32,  DataType::FLOAT16},
        {DataType::FLOAT32,  DataType::INT32},
        {DataType::FLOAT32,  DataType::INT64},
        {DataType::INT8,     DataType::INT16},
        {DataType::INT8,     DataType::INT32},
        {DataType::INT8,     DataType::INT64},
        {DataType::INT16,    DataType::INT32},
        {DataType::INT32,    DataType::INT64},
        {DataType::FLOAT16,  DataType::FLOAT32},
        {DataType::FLOAT16,  DataType::FLOAT64},
        {DataType::FLOAT16,  DataType::BFLOAT16},
        {DataType::BFLOAT16, DataType::FLOAT32},
        {DataType::BFLOAT16, DataType::FLOAT64},
        {DataType::BFLOAT16, DataType::FLOAT16},
        {DataType::FLOAT32,  DataType::FLOAT64},
        {DataType::FLOAT64,  DataType::FLOAT32},
    };
    return casts;
}

inline DataType compute_type(DataType type0, DataType type1) {
    if (type0 == type1) return type0;
    if (type0 <= DataType::INT64 && type1 <= DataType::INT64) {
        return std::max(type0, type1);
    } else if (type0 > DataType::INT64 && type1 > DataType::INT64) {
        return std::max(std::max(type0, DataType::FLOAT32), std::max(type1, DataType::FLOAT32));
    } else {
        if (type0 == DataType::INT64 || type1 == DataType::INT64) return DataType::FLOAT64;
        return std::max(std::max(type0, DataType::FLOAT32), std::max(type1, DataType::FLOAT32));
    }
}

inline bool is_cast_valid(DataType src, DataType dst) {
    return src == dst || valid_casts().contains({src, dst});
}

inline size_t calc_numel(const std::vector<int64_t>& shape) {
    size_t numel = 1;
    for (int dim : shape) {
        if (dim <= 0) throw std::runtime_error("Invalid tensor dimension");
        numel *= dim;
    }
    return numel;
}

inline size_t calc_numel(const DimArray& shape) {
    size_t numel = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) throw std::runtime_error("Invalid tensor dimension");
        numel *= size_t(dim);
    }
    return numel;
}

inline size_t calc_numel(std::initializer_list<int64_t> shape) {
    size_t numel = 1;
    for (int dim : shape) {
        if (dim <= 0) throw std::runtime_error("Invalid tensor dimension");
        numel *= dim;
    }
    return numel;
}
}  // export

#if !defined(RESTRICT)
    #if defined(_MSC_VER)
        #define RESTRICT __restrict
    #elif defined(__GNUC__) || defined(__clang__)
        #define RESTRICT __restrict__
    #else
        #define RESTRICT
    #endif
#endif
