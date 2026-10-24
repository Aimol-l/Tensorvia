#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/factory.hpp"

// Host <-> device transfer lives on TensorImpl, so every backend participates
// through the same two primitives and these symbols are defined exactly once
// regardless of how many backends are compiled in.
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
