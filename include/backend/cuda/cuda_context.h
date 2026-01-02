#pragma once
#include <memory>
#include <cuda_runtime.h>
#include "core/context.h"
#include <iostream>
#include <format>

class CUDAContext : public ContextImpl {
private:
    int m_device_id;
    cudaStream_t m_stream{};
public:
    // Constructor
    CUDAContext(){
        // 先自动查找当前的cuda设备
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            throw std::runtime_error("No CUDA devices found");
        }
        // 选择性能最好的设备（基于显存大小）
        int best_device = 0;
        size_t max_memory = 0;
        for(int i = 0; i < device_count; i++){
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            // 选择显存最大的设备
            if (prop.totalGlobalMem > max_memory) {
                max_memory = prop.totalGlobalMem;
                best_device = i;
            }
        }
        // 设置实际执行的设备,选择最高性能的设备
        m_device_id = best_device;
        cudaSetDevice(m_device_id);
        // 创建CUDA流
        cudaStreamCreate(&m_stream);

        // 打印设备信息
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, m_device_id);
        std::cout << "*************************CUDA Device Info******************" << std::endl;
        std::cout << "Device Name: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Global memory size: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << std::format("Max block dimensions:[{},{},{}]",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2])<< std::endl;
        std::cout << std::format("Max grid size:[{},{},{}]",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2])<< std::endl;
        std::cout << "***********************************************************" << std::endl;
    }
    ~CUDAContext() override {
        if (m_stream != nullptr) {
            cudaStreamDestroy(m_stream);
        }
    }
    void* ctx_raw_ptr() override{
        return &m_stream;
    }
    cudaStream_t stream() const {
        return m_stream;
    }
    int device_id() const {
        return m_device_id;
    }
    void wait() const {
        cudaStreamSynchronize(m_stream);
    }
};