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
