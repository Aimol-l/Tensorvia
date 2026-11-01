#include <cstddef>
#include <memory>

#include "tensorvia/types.hpp"
#include "tensorvia/factory.hpp"
#include "tensorvia/backend/sycl/sycl_context.hpp"
#include "tensorvia/backend/sycl/sycl_tensor.hpp"
#include "sycl_island.h"

namespace {
// sycl::context 与设备只有在请求第一个 Device::SYCL 张量时才会被触及；纯 CPU 程序
// 从不加载 libsycl。
struct SyclLazyRegistrar {
    SyclLazyRegistrar() {
        register_device_probe(Device::SYCL, [] {
            return DeviceProbe{sycl_c_device_count(), sycl_c_default_device_index()};
        });
        register_backend_initializer(Device::SYCL, [](int index) {
            auto ctx = std::make_shared<SYCLContext>(index);
            register_tensor_impl(DeviceId{Device::SYCL, index},
                                 [ctx](void* ptr, size_t numel, DataType dtype) {
                                     return std::make_shared<SYCLTensor>(ptr, numel, dtype, ctx);
                                 });
            // 每条（线程, 上下文）有自己的 in-order 队列，wait() 会等它们全部清空，
            // 因此这就是设备级同步，与 cudaDeviceSynchronize 一致。USM 由驱动直接
            // 管理，没有缓存可清。
            register_device_control(DeviceId{Device::SYCL, index}, DeviceControl{
                [ctx] { ctx->wait(); },
                [] {},
            });
        });
    }
} sycl_lazy_registrar;
}  // namespace
