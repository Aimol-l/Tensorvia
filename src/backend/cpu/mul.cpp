#include <omp.h>
#include "backend/cpu/ops/mul.h"
using namespace via;

namespace ops {

template <typename T, typename R, typename S>
void mul_kernel_basic(const T* a_ptr, const R* b_ptr, S* res_ptr, size_t batch, size_t rows, size_t common, size_t cols) {
    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());//+std::declval<compute_type_t<float>>());
                PromotedType sum = 0;
                for (size_t k = 0; k < common; ++k) {
                    size_t a_idx = b * rows * common + i * common + k;
                    size_t b_idx = b * common * cols + k * cols + j;
                    sum += static_cast<PromotedType>(a_ptr[a_idx]) * static_cast<PromotedType>(b_ptr[b_idx]);
                }
                size_t res_idx = b * rows * cols + i * cols + j;
                res_ptr[res_idx] = static_cast<S>(sum);
            }
        }
    }
}
// 分块优化的矩阵乘法（参考SYCL版本）
template <typename T, typename R, typename S, int TILE = 64>
void mul_kernel_tiled(const T* a_ptr, const R* b_ptr, S* result, size_t batch, size_t rows, size_t common, size_t cols) {
    const size_t A_batch_stride = rows * common;
    const size_t B_batch_stride = common * cols;
    const size_t C_batch_stride = rows * cols;
    #pragma omp parallel for collapse(3)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t by = 0; by < rows; by += TILE) {
            for (size_t bx = 0; bx < cols; bx += TILE) {
                // 当前tile的范围
                const size_t row_end = std::min(by + TILE, rows);
                const size_t col_end = std::min(bx + TILE, cols);
                // 为当前tile分配临时存储
                std::vector<T> Atile(TILE * TILE);
                std::vector<R> Btile(TILE * TILE);
                // 处理当前tile
                for (size_t k0 = 0; k0 < common; k0 += TILE) {
                    const size_t k_end = std::min(k0 + TILE, common);
                    // 加载A的tile [by:row_end, k0:k_end]
                    for (size_t i = by; i < row_end; ++i) {
                        for (size_t k = k0; k < k_end; ++k) {
                            size_t tile_idx = (i - by) * TILE + (k - k0);
                            size_t a_idx = b * A_batch_stride + i * common + k;
                            Atile[tile_idx] = a_ptr[a_idx];
                        }
                    }
                    // 加载B的tile [k0:k_end, bx:col_end]
                    for (size_t k = k0; k < k_end; ++k) {
                        for (size_t j = bx; j < col_end; ++j) {
                            size_t tile_idx = (k - k0) * TILE + (j - bx);
                            size_t b_idx = b * B_batch_stride + k * cols + j;
                            Btile[tile_idx] = b_ptr[b_idx];
                        }
                    }
                    // 计算当前tile的贡献
                    for (size_t i = by; i < row_end; ++i) {
                        for (size_t j = bx; j < col_end; ++j) {
                            using PromotedType = decltype(std::declval<compute_type_t<T>>());// + std::declval<compute_type_t<float>>());
                            PromotedType sum = 0;
                            
                            for (size_t k = 0; k < (k_end - k0); ++k) {
                                size_t a_tile_idx = (i - by) * TILE + k;
                                size_t b_tile_idx = k * TILE + (j - bx);
                                sum += static_cast<PromotedType>(Atile[a_tile_idx]) * 
                                       static_cast<PromotedType>(Btile[b_tile_idx]);
                            }
                            size_t res_idx = b * C_batch_stride + i * cols + j;
                            if (k0 == 0) {
                                result[res_idx] = static_cast<S>(sum);
                            } else {
                                result[res_idx] += static_cast<S>(sum);
                            }
                        }
                    }
                }
            }
        }
    }
}
// 自动选择优化策略的矩阵乘法
template <typename T, typename R, typename S>
void mul_kernel(const T* a, const R* b, S* result, size_t batch, size_t rows, size_t common, size_t cols) {
    // 对于小矩阵使用基础版本，大矩阵使用分块版本
    const size_t total_elements = batch * rows * cols;
    if (total_elements < 1024 * 1024) { // 1M元素以下用基础版本
        mul_kernel_basic(a, b, result, batch, rows, common, cols);
    } else {
        mul_kernel_tiled<T, R, S, 64>(a, b, result, batch, rows, common, cols);
    }
}
Tensor MulImpl<Device::CPU>::execute(const Tensor& a, const Tensor& b) {
    int batch =     a.shape().size() == 3?a.shape(0):1;
    int rows =      a.shape().size() == 3?a.shape(1):a.shape(0);
    int common =    a.shape().size() == 3?a.shape(2):a.shape(1);
    int cols =      a.shape().size() == 3?b.shape(2):b.shape(1);
    
    std::vector<int64_t> newshape;
    if(a.shape().size() == 3){
        newshape = {batch,rows,cols};
    }else{
        newshape = {rows,cols};
    }

    const Tensor& a_ = (a.dtype() == DataType::FLOAT16 || a.dtype() == DataType::BFLOAT16)?ops::Typecast(a,DataType::FLOAT32) : a;
    const Tensor& b_ = (b.dtype() == DataType::FLOAT16 || b.dtype() == DataType::BFLOAT16)?ops::Typecast(b,DataType::FLOAT32) : b;
    DataType res_type = compute_type(a_.dtype(),b_.dtype());
    Tensor result(newshape,res_type,Device::CPU);
    omp_set_num_threads(std::min(omp_get_max_threads(), 16));
    auto c_visitor = [&]<typename T, typename R>(const T* a_ptr,const R* b_ptr) {
        switch (res_type) {
            case DataType::INT8:
                mul_kernel<T,R,int8_t>(a_ptr,b_ptr,static_cast<int8_t*>(result.data()),batch,rows, common, cols);break;
            case DataType::INT16:
                mul_kernel<T,R,int16_t>(a_ptr,b_ptr,static_cast<int16_t*>(result.data()),batch,rows, common, cols);break;
            case DataType::INT32:
                mul_kernel<T,R,int32_t>(a_ptr,b_ptr,static_cast<int32_t*>(result.data()),batch,rows, common, cols);break;
            case DataType::INT64:
                mul_kernel<T,R,int64_t>(a_ptr,b_ptr,static_cast<int64_t*>(result.data()),batch,rows, common, cols);break;
            case DataType::FLOAT16:
                mul_kernel<T,R,float16>(a_ptr,b_ptr,static_cast<float16*>(result.data()),batch,rows, common, cols);break;
            case DataType::BFLOAT16:
                mul_kernel<T,R,bfloat16>(a_ptr,b_ptr,static_cast<bfloat16*>(result.data()),batch,rows, common, cols);break;
            case DataType::FLOAT32:
                mul_kernel<T,R,float32>(a_ptr,b_ptr,static_cast<float32*>(result.data()),batch,rows, common, cols);break;
            case DataType::FLOAT64:
                mul_kernel<T,R,float64>(a_ptr,b_ptr,static_cast<float64*>(result.data()),batch,rows, common, cols);break;
            default: throw std::runtime_error("Unsupported destination dtype");
        }
    };
    auto A = data_as_const_variant(a_.dtype(),a_.data());
    auto B = data_as_const_variant(b_.dtype(),b_.data());
    
    std::visit([&](auto A_ptr, auto B_ptr){
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(A_ptr)>>;
        using R = std::remove_cv_t<std::remove_pointer_t<decltype(B_ptr)>>;
        c_visitor(static_cast<const T*>(a_.data()),static_cast<const R*>(b_.data()));
    },A,B);
    return result;
}

template struct MulImpl<Device::CPU>;
}  // namespace ops



#include <immintrin.h>
#include <algorithm>
#include <cstdint>

// float16_t 通常是 uint16_t 的别名 (存储half precision)
using float16_t = uint16_t;

// ========== F16C 辅助函数 ==========

// 加载8个float16 -> 转换为8个float32 (__m256)
static inline __m256 load_f16x8_as_f32(const float16_t* ptr) {
    __m128i f16x8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    return _mm256_cvtph_ps(f16x8);  // F16C: half -> float
}

// 存储8个float32 (__m256) -> 转换为8个float16
static inline void store_f32x8_as_f16(float16_t* ptr, __m256 v) {
    __m128i f16x8 = _mm256_cvtps_ph(v, 
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), f16x8);
}

// 广播单个float16 -> 8个相同的float32 (__m256)
static inline __m256 broadcast_f16_as_f32(float16_t val) {
    __m128i f16x8 = _mm_set1_epi16(val);  // 16个lane都填val
    return _mm256_cvtph_ps(f16x8);         // 取低8个转换
}

// ========== 主算子 ==========
void linear(
    float16_t* __restrict__ out,
    const float16_t* __restrict__ x,      // [rows, common], row-major
    const float16_t* __restrict__ w,      // [common, cols], row-major
    int64_t rows,
    int64_t common,
    int64_t cols,
    const float16_t* __restrict__ bias    // [cols]
) {
    constexpr int64_t MR = 8;    // Micro-kernel行分块 (寄存器级)
    constexpr int64_t NR = 8;    // Micro-kernel列分块 (8×float32=256bit)
    constexpr int64_t KR = 256;  // K维度缓存分块
    // ===== 输出矩阵分块 =====
    for (int64_t i0 = 0; i0 < rows; i0 += MR) {
        const int64_t i1 = std::min(i0 + MR, rows);
        const int64_t mr = i1 - i0;  // 当前块实际行数
        for (int64_t j0 = 0; j0 < cols; j0 += NR) {
            const int64_t j1 = std::min(j0 + NR, cols);
            const int64_t nr = j1 - j0;  // 当前块实际列数
            const int64_t nr_vec = (nr + 7) / 8;  // 需要的__m256向量数
            // ===== 寄存器累加器 (关键优化!) =====
            // acc[r][v] 存储 out[i0+r][j0+v*8 : j0+(v+1)*8] 的float32累加值
            __m256 acc[MR][1];  // NR=8 => 每行只需1个__m256
            for (int64_t r = 0; r < mr; ++r) {
                acc[r][0] = _mm256_setzero_ps();
            }
            // ===== K维度分块 (reduction维度) =====
            for (int64_t k0 = 0; k0 < common; k0 += KR) {
                const int64_t k1 = std::min(k0 + KR, common);
                // 预取: 提前加载w的下一块到缓存
                if (k1 < common) {
                    _mm_prefetch((const char*)(w + k1 * cols + j0), _MM_HINT_T0);
                }
                for (int64_t k = k0; k < k1; ++k) {
                    // 1. 加载 w[k][j0:j0+nr] -> __m256
                    float16_t w_tile[8] = {0};  // 边界填充0
                    for (int64_t c = 0; c < nr; ++c) {
                        w_tile[c] = w[k * cols + j0 + c];
                    }
                    __m256 w_vec = load_f16x8_as_f32(w_tile);
                    // 2. 对每个输出行做FMA (关键循环!)
                    #pragma unroll
                    for (int64_t r = 0; r < mr; ++r) {
                        const int64_t i = i0 + r;
                        // 加载 x[i][k] 并广播到8个lane
                        __m256 x_vec = broadcast_f16_as_f32(x[i * common + k]);
                        // FMA: acc = x * w + acc (单指令完成乘加!)
                        acc[r][0] = _mm256_fmadd_ps(x_vec, w_vec, acc[r][0]);
                    }
                }
            }
            // ===== 加bias并写回 =====
            float16_t bias_tile[8] = {0};
            for (int64_t c = 0; c < nr; ++c) {
                bias_tile[c] = bias[j0 + c];
            }
            __m256 bias_vec = load_f16x8_as_f32(bias_tile);
            
            for (int64_t r = 0; r < mr; ++r) {
                const int64_t i = i0 + r;
                // acc + bias
                __m256 result = _mm256_add_ps(acc[r][0], bias_vec);
                // float32 -> float16 并存储
                float16_t out_tile[8];
                store_f32x8_as_f16(out_tile, result);
                // 处理边界列
                for (int64_t c = 0; c < nr; ++c) {
                    out[i * cols + j0 + c] = out_tile[c];
                }
            }
        }
    }
}