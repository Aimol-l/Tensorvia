// 宿主等待设备的可观测计数器。
//
// 为什么存在：FR-4.4 要求"除显式点外，算子调用不得隐式等待设备"。这条性质
// 不能用计时断言——快机器上"提交比同步快"会偶发成立，慢机器上又会误报——只能
// 数次数。每个让宿主真正等待设备的调用点调一次 note_wait，tests/async_contract.cpp
// 断言提交阶段的计数为 0、显式点各计一次。
//
// 为什么常开而不是"仅测试构建"：计数是一次 relaxed 原子加，而这些调用点本身就是
// 微秒到毫秒级的设备往返，开销可以忽略；条件编译会让四条 lane 各自少一份覆盖。
// 计数只由测试读取（tensorvia::sync::count()），生产路径不读它，也没有环境变量
// 开关——保持确定性。
//
// 这是内部头（与 _platform.h / _simd_math.h 同级的 include/ 位置），不进公开
// 安装面。
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace tensorvia::sync {

// 等待类别——"位置"的粗粒度记录：每类一个槽位，测试按类断言。
enum class Wait {
    Copy,        // 搬运：to_host / item<T> / to_device / 跨设备 / clone 的拷贝同步
    Synchronize, // ops::synchronize() 的显式设备级同步
    Backlog,     // Vulkan 在飞通道积压到上限（kMaxInFlight）时等最老的一次提交
    OomFree,     // 分配器 OOM 补救路径上的 cudaFree（驱动隐含一次设备同步）
    kCount,
};

inline std::atomic<uint64_t> g_waits[static_cast<size_t>(Wait::kCount)] = {};

inline void note_wait(Wait where) noexcept {
    g_waits[static_cast<size_t>(where)].fetch_add(1, std::memory_order_relaxed);
}

// 全部等待的总次数。
inline uint64_t count() noexcept {
    uint64_t total = 0;
    for (const auto& slot : g_waits) total += slot.load(std::memory_order_relaxed);
    return total;
}

inline uint64_t count(Wait where) noexcept {
    return g_waits[static_cast<size_t>(where)].load(std::memory_order_relaxed);
}

inline void reset() noexcept {
    for (auto& slot : g_waits) slot.store(0, std::memory_order_relaxed);
}

}  // namespace tensorvia::sync
