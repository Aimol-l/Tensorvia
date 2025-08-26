#include "backend/sycl/ops/mul.h"


namespace ops {
    template <typename T, typename R, typename S, int TILE = 16>
    void mul_sycl_3d(const T* a_ptr,const R* b_ptr,S* res_ptr,int batch,int cols,int rows,int common,sycl::queue& q){
        // 维度别名
        const int B = batch;
        const int M = rows;
        const int K = common;
        const int N = cols;
        // 每个 batch 的基址跨度（按行主序）
        const size_t A_batch_stride = static_cast<size_t>(M) * static_cast<size_t>(K);
        const size_t B_batch_stride = static_cast<size_t>(K) * static_cast<size_t>(N);
        const size_t C_batch_stride = static_cast<size_t>(M) * static_cast<size_t>(N);
        // 网格大小：按 TILE 对齐到上界
        const size_t gridM = (M + TILE - 1) / TILE;
        const size_t gridN = (N + TILE - 1) / TILE;
        // 设定 3D NDRange：
        //   global:  [B, gridM*TILE, gridN*TILE]
        //   local:   [1, TILE, TILE]
        //   group:   [b, by, bx]
        sycl::range<3> global_range(static_cast<size_t>(B),gridM * TILE,gridN * TILE);
        sycl::range<3> local_range(1, TILE, TILE);
        q.submit([&](sycl::handler& cgh) {
            // 本地内存 tile（分别缓存 A 的 [TILE x TILE] 和 B 的 [TILE x TILE]）
            sycl::local_accessor<T, 2> Atile(sycl::range<2>(TILE, TILE), cgh);
            sycl::local_accessor<R, 2> Btile(sycl::range<2>(TILE, TILE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_range, local_range),
                [=](sycl::nd_item<3> it) {
                    // 组/局部索引
                    const int b  = static_cast<int>(it.get_group(0));  // batch
                    const int by = static_cast<int>(it.get_group(1));  // 行块 id
                    const int bx = static_cast<int>(it.get_group(2));  // 列块 id
                    const int ly = static_cast<int>(it.get_local_id(1)); // [0, TILE)
                    const int lx = static_cast<int>(it.get_local_id(2)); // [0, TILE)
                    // 该线程要计算的输出 C 的坐标 (row, col)
                    const int row = by * TILE + ly;
                    const int col = bx * TILE + lx;
                    // 计算当前 batch 的基址
                    const size_t A_base = static_cast<size_t>(b) * A_batch_stride;
                    const size_t B_base = static_cast<size_t>(b) * B_batch_stride;
                    const size_t C_base = static_cast<size_t>(b) * C_batch_stride;
                    // 中间累加用 float（对半精度/整数都更稳）
                    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<float>>());
                    PromotedType acc = 0;
                    // 沿 K 维度分块
                    for (int k0 = 0; k0 < K; k0 += TILE) {
                        // 载入 A 的 tile： [row, k0 + lx]
                        if (row < M && (k0 + lx) < K) {
                            Atile[ly][lx] =
                                a_ptr[A_base + static_cast<size_t>(row) * K + (k0 + lx)];
                        } else {
                            // 越界填 0
                            Atile[ly][lx] = T(0);
                        }
                        // 载入 B 的 tile： [k0 + ly, col]
                        if ((k0 + ly) < K && col < N) {
                            Btile[ly][lx] = b_ptr[B_base + static_cast<size_t>(k0 + ly) * N + col];
                        } else {
                            Btile[ly][lx] = R(0);
                        }
                        // 等待所有线程完成载入
                        it.barrier(sycl::access::fence_space::local_space);
                        // 进行 TILE 内乘加
                        for (int kk = 0; kk < TILE; ++kk) {
                            // 强制提升到 float 做乘加，避免半精度/整数精度问题
                            acc += static_cast<PromotedType>(Atile[ly][kk]) * static_cast<PromotedType>(Btile[kk][lx]);
                        }
                        // 下一个 tile 之前同步
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    // 写回结果
                    if (row < M && col < N) {
                        res_ptr[C_base + static_cast<size_t>(row) * N + col] = static_cast<S>(acc);
                    }
                }
            );
        });

        // 这里不做 q.wait()，由上层（你的调用者）按需同步
}
    // [w,h] @ [h,w] --> [w,w]
    // [b,w,h] @ [b,h,w] --> [b,w,w]
    Tensor MulImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b){
        int batch =     a.shape().size() == 3?a.shape(0):1;
        int rows =      a.shape().size() == 3?a.shape(1):a.shape(0);
        int common =    a.shape().size() == 3?a.shape(2):a.shape(1);
        int cols =      a.shape().size() == 3?b.shape(2):b.shape(1);
        std::vector<int> newshape = {batch, rows, cols};
        DataType res_type = compute_type(a.dtype(),b.dtype());
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue(); 
        // LOG_INFO("Data type: " << dtype_to_string(res_type));
        Tensor result(newshape,res_type,Device::SYCL);
        auto c_visitor = [&]<typename T, typename R>(const T* a_ptr,const R* b_ptr) {
            switch (res_type) {
                case DataType::INT8:
                    mul_sycl_3d<T,R,int8_t>(a_ptr,b_ptr,static_cast<int8_t*>(result.data()),batch,cols,rows,common,q);break;
                case DataType::INT16:
                    mul_sycl_3d<T,R,int16_t>(a_ptr,b_ptr,static_cast<int16_t*>(result.data()),batch,cols,rows,common,q);break;
                case DataType::INT32:
                    mul_sycl_3d<T,R,int32_t>(a_ptr,b_ptr,static_cast<int32_t*>(result.data()),batch,cols,rows,common,q);break;
                case DataType::INT64:
                    mul_sycl_3d<T,R,int64_t>(a_ptr,b_ptr,static_cast<int64_t*>(result.data()),batch,cols,rows,common,q);break;
                case DataType::FLOAT16:
                    mul_sycl_3d<T,R,float16>(a_ptr,b_ptr,static_cast<float16*>(result.data()),batch,cols,rows,common,q);break;
                case DataType::BFLOAT16:
                    mul_sycl_3d<T,R,bfloat16>(a_ptr,b_ptr,static_cast<bfloat16*>(result.data()),batch,cols,rows,common,q);break;
                case DataType::FLOAT32:
                    mul_sycl_3d<T,R,float32>(a_ptr,b_ptr,static_cast<float32*>(result.data()),batch,cols,rows,common,q);break;
                case DataType::FLOAT64:
                    mul_sycl_3d<T,R,float64>(a_ptr,b_ptr,static_cast<float64*>(result.data()),batch,cols,rows,common,q);break;
                default: throw std::runtime_error("Unsupported destination dtype");
            }
        };
        auto A = data_as_const_variant(a.dtype(),a.data());
        auto B = data_as_const_variant(b.dtype(),b.data());
        std::visit([&](auto A_ptr, auto B_ptr){
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(A_ptr)>>;
            using R = std::remove_cv_t<std::remove_pointer_t<decltype(B_ptr)>>;
            c_visitor(static_cast<const T*>(a.data()),static_cast<const R*>(b.data()));
        },A,B);
        q.wait();
        return result;
    }

    template struct MulImpl<Device::SYCL>;
}