#include "backend/sycl/ops/repack.h"
using namespace via;

void RepackImpl<Device::SYCL>::execute(const Metadata& meta,void* input,void* output,std::shared_ptr<SYCLContext> ctx) {
    // 获取SYCL队列
    auto& queue = ctx->get_queue();
    
    // 计算数据类型大小
    size_t dtype_size = calc_dtype_size(meta.dtype);
    size_t ndim = meta.shape.size();
    
    // 分配设备内存并复制shape和strides数据
    int64_t* d_shape = sycl::malloc_device<int64_t>(ndim, queue);
    int64_t* d_strides = sycl::malloc_device<int64_t>(ndim, queue);
    
    queue.memcpy(d_shape, meta.shape.data(), ndim * sizeof(int64_t)).wait();
    queue.memcpy(d_strides, meta.strides.data(), ndim * sizeof(int64_t)).wait();

    // 获取输入输出指针
    const char* src_base = static_cast<const char*>(input) + meta.offset * dtype_size;
    char* dst = static_cast<char*>(output);
    size_t numel = meta.numel;

    // 定义块大小和网格大小
    size_t block_size = 256;
    size_t grid_size = (numel + block_size - 1) / block_size;
    size_t total_threads = grid_size * block_size;

    // 提交SYCL kernel任务
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total_threads), [=](sycl::id<1> idx) {
            size_t linear_idx = idx[0];
            if (linear_idx >= numel) return;
            
            // 将线性索引转换为多维坐标并计算源偏移
            size_t temp = linear_idx;
            size_t src_offset = 0;
            
            for (int i = ndim - 1; i >= 0; --i) {
                size_t coord = temp % static_cast<size_t>(d_shape[i]);
                temp /= static_cast<size_t>(d_shape[i]);
                src_offset += coord * static_cast<size_t>(d_strides[i]);
            }
            
            // 拷贝 dtype_size 字节
            for (size_t b = 0; b < dtype_size; ++b) {
                dst[linear_idx * dtype_size + b] = src_base[src_offset * dtype_size + b];
            }
        });
    }).wait();

    // 释放分配的设备内存
    sycl::free(d_shape, queue);
    sycl::free(d_strides, queue);
}

template struct RepackImpl<Device::SYCL>;