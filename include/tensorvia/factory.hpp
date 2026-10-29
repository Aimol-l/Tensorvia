#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"

using TensorImplFactory =
    std::function<std::shared_ptr<TensorImpl>(void* ptr, size_t numel, DataType)>;
// Takes the instance index being brought up, so one backend can own several
// contexts (one per GPU).
using BackendInitializer = std::function<void(int device_index)>;
// How many instances a backend has, and which one new work should default to.
// Probed lazily and cached: the answer depends on the hardware, and probing it
// loads the driver.
struct DeviceProbe {
    int count = 0;
    int default_index = 0;
};
using BackendProbe = std::function<DeviceProbe()>;

void register_tensor_impl(DeviceId device, TensorImplFactory factory);

// Registers a callback that is invoked lazily, the first time a device instance
// is requested without an implementation having been registered for it. This
// keeps optional backends (Vulkan) completely dormant in CPU-only runs.
void register_backend_initializer(Device device, BackendInitializer init);

// Registers how to discover the backend's instances. Called at most once per
// backend, on the first question about its device count or current device.
void register_device_probe(Device device, BackendProbe probe);
// The backend's probe result; runs the probe on first use. Returns {0, 0} when
// the backend has no probe (absent or not compiled in).
DeviceProbe device_probe(Device device);

// The calling thread's current instance index for `kind`, resolved on first use
// to the probe's default. Throws when the backend has no usable device: asking
// for the current device of an absent backend is a caller error, not "0".
int  current_device_index(Device kind);
// Sets the calling thread's current index, rejecting anything outside
// [0, device_probe(kind).count).
void set_current_device_index(Device kind, int index);

// Device-scoped operations that are not tensor computations, so they do not
// belong on Tensor or on the backend tensor implementations. Both are
// optional: a backend that leaves one empty is not asked for it.
struct DeviceControl {
    // Waits for everything already submitted to the device — every stream, not
    // just the calling thread's current one. The counterpart of
    // torch.cuda.synchronize() / torch.accelerator.synchronize().
    std::function<void()> synchronize;
    // Hands reclaimable cached memory back to the driver. Live buffers are not
    // touched. The counterpart of torch.cuda.empty_cache().
    std::function<void()> empty_cache;
};

// Registered like a tensor implementation: either eagerly (CPU) or from the
// device's lazy initializer (accelerators).
void register_device_control(DeviceId device, DeviceControl control);

// The controls of every device instance that has been brought up. An instance
// whose backend never initialised is absent, so asking to wait or to release
// memory cannot drag a dormant backend (and its driver) into existence.
std::vector<std::pair<DeviceId, DeviceControl>> live_device_controls();

std::shared_ptr<TensorImpl> create_tensor_impl(size_t numel,
                                                      DataType dtype,
                                                      DeviceId device);
std::shared_ptr<TensorImpl> create_tensor_impl(void* ptr,
                                                     size_t numel,
                                                     DataType dtype,
                                                     DeviceId device);

void copy_device_to_host(std::shared_ptr<TensorImpl> src,
                               std::shared_ptr<TensorImpl> dst,
                               DataType dtype);
void copy_host_to_device(std::shared_ptr<TensorImpl> src,
                               std::shared_ptr<TensorImpl> dst,
                               DataType dtype);
