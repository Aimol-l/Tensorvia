#include <cstddef>
#include <memory>

#include "tensorvia/types.hpp"
#include "tensorvia/factory.hpp"
#include "tensorvia/backend/sycl/sycl_context.hpp"
#include "tensorvia/backend/sycl/sycl_tensor.hpp"
#include "sycl_island.h"

namespace {
// 队列与 USM 分配器只有在请求第一个 Device::SYCL 张量时才会被触及；纯 CPU 程序
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
            // 队列是 in-order 的：等它清空就是设备级同步。USM 由驱动直接管理，没有
            // 缓存可清。SYCL 没有 per-thread 的流，所以这一条覆盖所有宿主线程。
            register_device_control(DeviceId{Device::SYCL, index}, DeviceControl{
                [ctx] { ctx->wait(); },
                [] {},
            });
        });
    }
} sycl_lazy_registrar;
}  // namespace
