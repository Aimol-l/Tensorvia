
#include "backend/cpu/ops/mul.h"



namespace ops {

// 3维度矩阵乘法 [b,w,h] @ [b,h,w] --> [b,w,w]
template <typename T>
void mul_kernel(const T* pa, const T* pb, T* pr, size_t batch, size_t rows, size_t cols, size_t common_dim) {
#pragma omp parallel for
    for (size_t i = 0; i < batch * rows * cols; ++i) {
        pr[i] = 0;
    }
// 优化后的并行策略
#pragma omp parallel for schedule(dynamic, 32)  // 这里使用动态调度和块大小16，这对大型矩阵效果较好
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < rows; ++j) {
            for (size_t k = 0; k < common_dim; ++k) {
                T a_val = pa[i * rows * common_dim + j * common_dim + k];
                // 内层循环连续访问
                for (size_t l = 0; l < cols; ++l) {
                    pr[i * rows * cols + j * cols + l] += a_val * pb[i * common_dim * cols + k * cols + l];
                }
            }
        }
    }
}

Tensor MulImpl<Device::CPU>::execute(const Tensor& a, const Tensor& b) {
    // 精度提升
    const Tensor& A = a.dtype() > b.dtype() ? a : ops::Typecast(a, b.dtype());
    const Tensor& B = a.dtype() <= b.dtype() ? b : ops::Typecast(b, a.dtype());
    // 创建结果张量
    Tensor res;
    int batch, cols, rows, common_dim;
    if (a.shape().size() == 3) {  // [batch,rows,cols]
        batch = a.shape(0);
        rows = a.shape(1);
        cols = b.shape(2);
        common_dim = a.shape(2);
        res = Tensor({batch, rows, cols}, A.dtype(), Device::CPU);
    } else {  // [w,h]
        batch = 1;
        rows = a.shape(0);
        cols = b.shape(1);
        common_dim = a.shape(1);
        res = Tensor({rows, cols}, A.dtype(), Device::CPU);
    }
    switch (A.dtype()) {
        case DataType::INT8:
            mul_kernel<int8_t>(static_cast<const int8_t*>(A.data()), static_cast<const int8_t*>(B.data()), static_cast<int8_t*>(res.data()), batch, rows, cols, common_dim);
            break;
        case DataType::INT16:
            mul_kernel<int16_t>(static_cast<const int16_t*>(A.data()), static_cast<const int16_t*>(B.data()), static_cast<int16_t*>(res.data()), batch, rows, cols, common_dim);
            break;
        case DataType::INT32:
            mul_kernel<int32_t>(static_cast<const int32_t*>(A.data()), static_cast<const int32_t*>(B.data()), static_cast<int32_t*>(res.data()), batch, rows, cols, common_dim);
            break;
        case DataType::INT64:
            mul_kernel<int64_t>(static_cast<const int64_t*>(A.data()), static_cast<const int64_t*>(B.data()), static_cast<int64_t*>(res.data()), batch, rows, cols, common_dim);
            break;
        case DataType::FLOAT16:
            mul_kernel<float16>(static_cast<const float16*>(A.data()), static_cast<const float16*>(B.data()), static_cast<float16*>(res.data()), batch, rows, cols, common_dim);
            break;
        case DataType::BFLOAT16:
            mul_kernel<bfloat16>(static_cast<const bfloat16*>(A.data()), static_cast<const bfloat16*>(B.data()), static_cast<bfloat16*>(res.data()), batch, rows, cols, common_dim);
            break;
        case DataType::FLOAT32:
            mul_kernel<float32>(static_cast<const float32*>(A.data()), static_cast<const float32*>(B.data()), static_cast<float32*>(res.data()), batch, rows, cols, common_dim);
            break;
        case DataType::FLOAT64:
            mul_kernel<float64>(static_cast<const float64*>(A.data()), static_cast<const float64*>(B.data()), static_cast<float64*>(res.data()), batch, rows, cols, common_dim);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for dot");
    }
    return res;
}

template struct MulImpl<Device::CPU>;
}  // namespace ops