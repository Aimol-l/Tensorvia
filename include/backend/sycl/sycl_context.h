#pragma once
#include "core/context.h"
#include <sycl/sycl.hpp>
class SYCLContext: public ContextImpl {
private:
    sycl::queue m_queue;
public:
    ~SYCLContext()override{}
    SYCLContext(){
        // sycl::property_list props{sycl::property::queue::enable_profiling()};
        // sycl::queue q(sycl::gpu_selector{}, props);
        m_queue = sycl::queue(sycl::default_selector_v);
        std::cout<<("*************************Backend Device Info******************")<<std::endl;
        sycl::info::device_type info = m_queue.get_device().get_info<sycl::info::device::device_type>();
        if(sycl::info::device_type::cpu == info) 
            std::cout<<("Device Type: CPU (GPU not found!)")<<std::endl;
        if(sycl::info::device_type::gpu == info) 
            std::cout<<("Device Type: GPU")<<std::endl;
        if(sycl::info::device_type::accelerator == info) 
            std::cout<<("Device Type: FPGA")<<std::endl;
        std::cout<<("Device Name: ")<<m_queue.get_device().get_info<sycl::info::device::name>()<<std::endl;
        std::cout<<("Max work item size: ")<<m_queue.get_device().get_info<sycl::info::device::max_work_group_size>()<<std::endl;
        std::cout<<("Global memory size: ")<<m_queue.get_device().get_info<sycl::info::device::global_mem_size>()/float(1073741824)<<std::endl;
        std::cout<<("Max compute units: ")<<m_queue.get_device().get_info<sycl::info::device::max_compute_units>()<<std::endl;
        std::cout<<("*************************************************************")<<std::endl;
    }
    void* ctx_raw_ptr() override{
        return &m_queue;
    }
    sycl::queue& get_queue(){
        m_queue.wait();
        return m_queue;
    }
};