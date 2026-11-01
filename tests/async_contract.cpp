// 隐式同步契约（FR-4.4）：除显式点（to_host / item<T> / synchronize / 跨设备
// 拷贝）外，算子调用不得让宿主等待设备。
//
// 判据不是计时——快机器上"提交比同步快"会偶发成立，慢机器上又会误报——而是
// 可观测计数器：每次宿主真正等待设备，tensorvia::sync::note_wait 就记一次
// （按等待类别分槽，见 include/_sync_counters.h）。这些用例因此是确定性的：
// 数次数，不比快慢。
//
// 各 lane 的行为差异（CUDA 同步当前流、SYCL 等待调用线程的队列、Vulkan 等
// 队首 fence）只影响显式点的计数，提交阶段在所有后端上都必须是 0。
#include "test_support.hpp"

#include "_sync_counters.h"

namespace {

using tensorvia::sync::Wait;

// 在目标设备上建一对可乘的方阵（一次性成本，计入计数器之前）。
std::pair<Tensor, Tensor> square_pair(Device dev, int64_t side) {
    std::vector<float> v(static_cast<size_t>(side * side), 1.0f);
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i % 7) * 0.25f;
    return {moved_to(Tensor(v, std::vector<int64_t>{side, side}), dev),
            moved_to(Tensor(v, std::vector<int64_t>{side, side}), dev)};
}

}  // namespace

// ---------- 提交阶段不等待 ----------
TEST_F(DeviceOps, SubmittingOpsDoesNotWaitForDevice) {
    if (dev == Device::CPU) GTEST_SKIP() << "CPU 后端是同步的，没有提交/等待之分";
    if (!backend_available(dev)) GTEST_SKIP() << device_to_string(dev) << " is unavailable";

    auto [a, b] = square_pair(dev, 16);
    // 连续视图（带 offset 的领先维切片）：不触发物化，是"视图输入进内核"的路径。
    const Tensor view = a.narrow(0, 4, 8);

    tensorvia::sync::reset();
    Tensor acc = ops::MatMul(a, b);
    for (int i = 0; i < 32; ++i) {
        acc = ops::MatMul(acc, b);          // 归约族
        (void)ops::Relu(acc);               // 逐元素族（原地形式）
        (void)ops::Exp(view);               // 视图输入
        (void)ops::Sum(acc, 1);             // 轴归约
        (void)ops::Softmax(acc, 1);
    }
    EXPECT_EQ(tensorvia::sync::count(), 0u)
        << "提交阶段发生了隐式同步（copy=" << tensorvia::sync::count(Wait::Copy)
        << ", synchronize=" << tensorvia::sync::count(Wait::Synchronize)
        << ", backlog=" << tensorvia::sync::count(Wait::Backlog) << "）";
    ops::synchronize();
}

// ---------- 显式点各计一次，提交不再计数 ----------
TEST_F(DeviceOps, ExplicitSyncPointsAreTheOnlyWaits) {
    if (dev == Device::CPU) GTEST_SKIP() << "CPU 后端是同步的，没有提交/等待之分";
    if (!backend_available(dev)) GTEST_SKIP() << device_to_string(dev) << " is unavailable";

    auto [a, b] = square_pair(dev, 16);
    tensorvia::sync::reset();

    // 1) 连续提交：0 次等待。
    Tensor acc = ops::MatMul(a, b);
    for (int i = 0; i < 16; ++i) acc = ops::MatMul(acc, b);
    EXPECT_EQ(tensorvia::sync::count(), 0u) << "提交阶段隐式同步";

    // 2) item()：一次设备到宿主搬运。
    const uint64_t after_submit_copy = tensorvia::sync::count(Wait::Copy);
    const float probe = acc.narrow(0, 0, 1).narrow(1, 0, 1).item<float>();
    EXPECT_GE(tensorvia::sync::count() - after_submit_copy, 1u) << "item() 必须同步";
    EXPECT_EQ(tensorvia::sync::count(Wait::Copy) - after_submit_copy, 1u)
        << "item() 是一次拷贝等待";
    EXPECT_TRUE(std::isfinite(probe));

    // 3) 之后再提交：计数不变（提交仍然不等待）。
    const uint64_t after_item = tensorvia::sync::count();
    for (int i = 0; i < 16; ++i) acc = ops::MatMul(acc, b);
    EXPECT_EQ(tensorvia::sync::count(), after_item) << "同步点之后的提交又引入了等待";

    // 4) synchronize()：显式设备级同步，至少一次等待；类别是 Synchronize。
    const uint64_t before_sync = tensorvia::sync::count();
    ops::synchronize();
    EXPECT_GE(tensorvia::sync::count() - before_sync, 1u) << "synchronize() 必须等待";
    EXPECT_GE(tensorvia::sync::count(Wait::Synchronize), 1u);

    // 5) to_host()：再一次拷贝等待。
    const uint64_t after_sync_copy = tensorvia::sync::count(Wait::Copy);
    Tensor host_side = acc;
    host_side.to_host();
    EXPECT_EQ(tensorvia::sync::count(Wait::Copy) - after_sync_copy, 1u)
        << "to_host() 是一次拷贝等待";

    ops::synchronize();
}
