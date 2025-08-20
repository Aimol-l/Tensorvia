#include "backend/sycl/ops/mul.h"


namespace ops {

    // [w,h] @ [h,w] --> [w,w]
    // [b,w,h] @ [b,h,w] --> [b,w,w]
    // template<typename T>
    // void mul_sycl_3d(void* a,void* b,void* res,
    //                     size_t batch,size_t cols,size_t rows,size_t common,
    //                     sycl::queue& q){
    //     T* a_ptr = static_cast<T*>(a);
    //     T* b_ptr = static_cast<T*>(b);
    //     T* res_ptr = static_cast<T*>(res);
    //     q.submit([&](sycl::handler& h) { 
    //         h.parallel_for(sycl::range<3>{batch,cols,rows},[=](sycl::id<3> idx){
    //             int b = idx[0];
    //             int i = idx[1];
    //             int j = idx[2];
    //             T sum = 0;
    //             for (int k = 0; k < common; ++k) {
    //                 // 计算全局偏移量：
    //                 // a[b, i, k] 的偏移 = b * (rows * common) + i * common + k
    //                 // b[b, k, j] 的偏移 = b * (common * cols) + k * cols + j
    //                 sum += a_ptr[b * rows * common + i * common + k] * b_ptr[b * common * cols + k * cols + j];
    //             }
    //             res_ptr[b * rows * cols + i * cols + j] = sum;
    //         });
    //     }).wait();
    // }

    // 优化版本

    template <typename T, int TILE = 16>
    void mul_sycl_3d(const void* a, const void* b, void* res,
                    size_t batch, size_t cols, size_t rows, size_t common,
                    sycl::queue& q) {
        const T* a_ptr = static_cast<const T*>(a);
        const T* b_ptr = static_cast<const T*>(b);
        T* res_ptr = static_cast<T*>(res);
        sycl::range<3> global_range(batch, (rows + TILE - 1) / TILE * TILE, (cols + TILE - 1) / TILE * TILE);
        sycl::range<3> local_range(1, TILE, TILE);
        auto e = q.submit([&](sycl::handler& h) {
            // Local tile buffer for A and B
            sycl::local_accessor<T, 2> tileA({TILE, TILE}, h);
            sycl::local_accessor<T, 2> tileB({TILE, TILE}, h);
            h.parallel_for(sycl::nd_range<3>(global_range, local_range), [=](sycl::nd_item<3> item) {
                size_t b = item.get_global_id(0); // batch index
                size_t row = item.get_global_id(1); // output row index
                size_t col = item.get_global_id(2); // output col index
                T sum = 0;
                // Tile loop over common dimension
                for (int t = 0; t < (int)((common + TILE - 1) / TILE); ++t) {
                    // Load A tile to local memory
                    if (row < rows && t * TILE + item.get_local_id(2) < common)
                        tileA[item.get_local_id(1)][item.get_local_id(2)] =
                            a_ptr[b * rows * common + row * common + t * TILE + item.get_local_id(2)];
                    else
                        tileA[item.get_local_id(1)][item.get_local_id(2)] = 0;
                    // Load B tile to local memory
                    if (t * TILE + item.get_local_id(1) < common && col < cols)
                        tileB[item.get_local_id(1)][item.get_local_id(2)] =
                            b_ptr[b * common * cols + (t * TILE + item.get_local_id(1)) * cols + col];
                    else
                        tileB[item.get_local_id(1)][item.get_local_id(2)] = 0;
                    item.barrier(sycl::access::fence_space::local_space);
                    // Compute partial sum
                    for (int k = 0; k < TILE; ++k)
                        sum += tileA[item.get_local_id(1)][k] * tileB[k][item.get_local_id(2)];

                    item.barrier(sycl::access::fence_space::local_space);
                }
                // Write result
                if (row < rows && col < cols)
                    res_ptr[b * rows * cols + row * cols + col] = sum;
            });
        });
        e.wait();
    }



    // [w,h] @ [h,w] --> [w,w]
    // [b,w,h] @ [b,h,w] --> [b,w,w]
    Tensor MulImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b){
        // 精度提升
        Tensor c;
        const void* a_prt = nullptr;
        const void* b_ptr = nullptr;
        if(a.dtype()>b.dtype()){
            c = typecast(b, a.dtype());
            a_prt = a.data();
            b_ptr = c.data();
        }else{
            c = typecast(a, b.dtype());
            a_prt = c.data();
            b_ptr = b.data();
        }
        int batch,rows,cols,common;
        std::vector<int> newshape(c.shape().size());
        if(c.shape().size() == 3){
            batch = c.shape(0);
            rows = a.shape()[1];
            cols = b.shape()[2];
            common = a.shape()[2];
            newshape = {batch,rows,cols};
        }else{
            batch = 1;
            rows = a.shape()[0];
            cols = b.shape()[1];
            common = a.shape()[1];
            newshape = {rows,cols};
        }
        Tensor result(newshape,c.dtype(),Device::SYCL);
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue(); 
        // 分发到模板 kernel（根据 dtype 决定类型）
        void* res_ptr = result.data();
        switch (c.dtype()) {
            case DataType::INT8:            mul_sycl_3d<int8_t>(a_prt,b_ptr,res_ptr,batch,cols,rows,common ,q);break;
            case DataType::INT16:           mul_sycl_3d<int16_t>(a_prt,b_ptr,res_ptr,batch,cols,rows,common ,q);break;
            case DataType::INT32:           mul_sycl_3d<int32_t>(a_prt,b_ptr,res_ptr,batch,cols,rows,common ,q);break;
            case DataType::INT64:           mul_sycl_3d<int64_t>(a_prt,b_ptr,res_ptr,batch,cols,rows,common ,q);break;
            case DataType::FLOAT16:         mul_sycl_3d<float16>(a_prt,b_ptr,res_ptr,batch,cols,rows,common ,q);break;
            case DataType::BFLOAT16:        mul_sycl_3d<bfloat16>(a_prt,b_ptr,res_ptr,batch,cols,rows,common ,q);break;
            case DataType::FLOAT32:         mul_sycl_3d<float32>(a_prt,b_ptr,res_ptr,batch,cols,rows,common ,q);break;
            case DataType::FLOAT64:         mul_sycl_3d<float64>(a_prt,b_ptr,res_ptr,batch,cols,rows,common ,q);break;
            default:throw std::runtime_error("Unsupported dtype for mul");
        }
        return result;
    }

    template struct MulImpl<Device::SYCL>;
}