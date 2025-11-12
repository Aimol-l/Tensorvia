#include <omp.h>
#include "backend/cpu/ops/mul.h"



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