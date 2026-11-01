#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/factory.hpp"

// 宿主 <-> 设备传输实现在 TensorImpl 上，所以每个后端都通过同样的两个原语参与，
// 而这些符号无论编入了多少个后端都只定义一次。
void copy_device_to_host(std::shared_ptr<TensorImpl> src,
                         std::shared_ptr<TensorImpl> dst,
                         DataType dtype) {
    if (!src || !dst) throw std::invalid_argument("copy_device_to_host: null tensor implementation");
    src->copy_to_host(dst->data(), dtype);
}

void copy_host_to_device(std::shared_ptr<TensorImpl> src,
                         std::shared_ptr<TensorImpl> dst,
                         DataType dtype) {
    if (!src || !dst) throw std::invalid_argument("copy_host_to_device: null tensor implementation");
    dst->copy_from_host(src->data(), dtype);
}
