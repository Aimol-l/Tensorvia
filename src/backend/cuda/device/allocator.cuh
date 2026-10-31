
#ifndef TVV_CUDA_ALLOCATOR_CUH
#define TVV_CUDA_ALLOCATOR_CUH

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_launch.hpp"

namespace tvv::cuda {

class CachingAllocator {
public:
    // 每次 allocate / deallocate 之前，调用方必须已在调用线程上把 `device` 设为
    // current（CUDAContext::activate 在每个入口都这么做；分配器不会背着调用方切换
    // 设备——c10::cuda::CUDACachingAllocator 也是靠外部守卫做到这一点）。
    explicit CachingAllocator(int device) noexcept : m_device(device) {}
    ~CachingAllocator();

    CachingAllocator(const CachingAllocator&) = delete;
    CachingAllocator& operator=(const CachingAllocator&) = delete;
    CachingAllocator(CachingAllocator&&) = delete;
    CachingAllocator& operator=(CachingAllocator&&) = delete;

    // 永不返回空。只有在 OOM 触发 empty_cache() 仍腾不出空间后才抛 std::bad_alloc。
    void* allocate(size_t bytes, cudaStream_t stream);

    // noexcept：由 CUDATensor 的析构函数调用。未知指针直接忽略而不报错——无论
    // 哪种处理都没有安全可言。
    void deallocate(void* ptr, cudaStream_t stream) noexcept;

    // 返回已无活动张量引用的 segment。只要还有块在用，该 segment 就保留，因为它的
    // 地址空间是共享的。
    void empty_cache() noexcept;

    struct Stats {
        size_t malloc_calls = 0;   // cudaMalloc 往返次数
        size_t cache_hits   = 0;   // 由空闲链表满足的分配次数
        size_t cached_bytes = 0;   // 当前空闲块持有的字节数
    };
    Stats stats() const;

private:
    // 小请求从 2 MiB 的池 segment 中切出；大请求向上取整到 2 MiB 的倍数。这两个
    // 尺寸都取自 torch。
    static constexpr size_t kMinBlock    = 512;
    static constexpr size_t kSmallSize   = size_t{1} << 20;   // 1 MiB（小请求阈值）
    static constexpr size_t kSmallBuffer = size_t{2} << 20;   // 2 MiB（小请求池大小）
    static constexpr size_t kLargeRound  = size_t{2} << 20;   // 2 MiB（大请求取整步长）

    struct Block {
        void*        ptr     = nullptr;
        size_t       size    = 0;
        size_t       segment = 0;   // m_segments 中的下标
        cudaStream_t stream  = nullptr;   // 块只在这条流上复用
    };

    struct Segment {
        void*  ptr        = nullptr;   // 归还驱动后为 nullptr
        size_t total      = 0;
        size_t free_bytes = 0;
    };

    static size_t round_request(size_t bytes) noexcept;
    static size_t pool_size(size_t need) noexcept;

    // 需持有 m_mtx。
    std::pair<void*, Block*> take_free(size_t need, cudaStream_t stream);
    void                     release(Block* block, cudaStream_t stream) noexcept;
    void                     split(Block* block, size_t need);
    void                     insert_free(Block* block) noexcept;
    void                     release_free_segments() noexcept;
    size_t                   cached_bytes() const noexcept;

    int                  m_device;
    mutable std::mutex   m_mtx;
    // 按流分池：块只在释放它的那条流上复用，同流按序执行即保证安全，不需要事件。
    // 每条流内部是最佳适配——lower_bound(need) 是仍能容纳请求的最小块。
    std::unordered_map<cudaStream_t, std::multimap<size_t, Block*>> m_free;
    std::unordered_map<void*, Block*> m_live;   // 仅含在用块
    std::vector<Segment>              m_segments;

    Stats m_stats;
};

// ---------------------------------------------------------------------------
// 实现
// ---------------------------------------------------------------------------

inline size_t CachingAllocator::round_request(size_t bytes) noexcept {
    if (bytes < kMinBlock) return kMinBlock;
    return kMinBlock * ((bytes + kMinBlock - 1) / kMinBlock);
}

inline size_t CachingAllocator::pool_size(size_t need) noexcept {
    if (need <= kSmallSize) return kSmallBuffer;
    return kLargeRound * ((need + kLargeRound - 1) / kLargeRound);
}

inline void* CachingAllocator::allocate(size_t bytes, cudaStream_t stream) {
    const size_t need = round_request(bytes);

    std::lock_guard<std::mutex> lock(m_mtx);

    if (auto [ptr, block] = take_free(need, stream); block != nullptr) {
        m_stats.cache_hits++;
        // 块只会来自调用方自己的流，之前的写入按序排在本请求之前。
        m_live[ptr] = block;
        return ptr;
    }

    const size_t pool = pool_size(need);
    void*        ptr  = nullptr;
    cudaError_t  err  = cudaMalloc(&ptr, pool);
    if (err != cudaSuccess) {
        // 唯一的补救：清空缓存再试一次。
        release_free_segments();
        err = cudaMalloc(&ptr, pool);
        if (err != cudaSuccess) throw std::bad_alloc();
    }
    m_stats.malloc_calls++;

    m_segments.push_back(Segment{ptr, pool, 0});
    auto* block = new Block{ptr, pool, m_segments.size() - 1, stream};
    m_live[ptr] = block;
    split(block, need);   // 未用的尾部交还空闲链表
    return ptr;
}

inline void CachingAllocator::deallocate(void* ptr, cudaStream_t stream) noexcept {
    if (ptr == nullptr) return;
    std::lock_guard<std::mutex> lock(m_mtx);
    const auto it = m_live.find(ptr);
    if (it == m_live.end()) return;
    Block* block = it->second;
    m_live.erase(it);
    release(block, stream);
}

// 需持有 m_mtx。块回到它所在的流：只有那条流上的后续请求能拿到它。
inline void CachingAllocator::release(Block* block, cudaStream_t stream) noexcept {
    block->stream = stream;
    insert_free(block);
}

inline std::pair<void*, CachingAllocator::Block*>
CachingAllocator::take_free(size_t need, cudaStream_t stream) {
    const auto bucket = m_free.find(stream);
    if (bucket == m_free.end()) return {nullptr, nullptr};

    const auto it = bucket->second.lower_bound(need);
    if (it == bucket->second.end()) return {nullptr, nullptr};

    Block* block = it->second;
    bucket->second.erase(it);
    m_segments[block->segment].free_bytes -= block->size;
    // split() 会把尾部重新插回（并把其字节数加回计数器）。
    split(block, need);
    return {block->ptr, block};
}

inline void CachingAllocator::split(Block* block, size_t need) {
    // 小于最小块大小的尾部不值得单列一条空闲项；这次分配直接留着它。
    if (block->size < need + kMinBlock) return;

    auto* tail  = new Block{static_cast<char*>(block->ptr) + need,
                            block->size - need, block->segment, block->stream};
    block->size = need;
    insert_free(tail);
}

inline void CachingAllocator::insert_free(Block* block) noexcept {
    m_free[block->stream].emplace(block->size, block);
    m_segments[block->segment].free_bytes += block->size;
}

inline size_t CachingAllocator::cached_bytes() const noexcept {
    size_t sum = 0;
    for (const auto& seg : m_segments) sum += seg.free_bytes;
    return sum;
}

inline void CachingAllocator::release_free_segments() noexcept {
    for (size_t i = 0; i < m_segments.size(); ++i) {
        Segment& seg = m_segments[i];
        // 还有块没释放的 segment 与活动张量共享地址空间，无法归还。
        if (seg.ptr == nullptr || seg.free_bytes != seg.total) continue;

        for (auto& entry : m_free) {
            auto& bucket = entry.second;
            for (auto it = bucket.begin(); it != bucket.end();) {
                if (it->second->segment != i) {
                    ++it;
                    continue;
                }
                delete it->second;
                it = bucket.erase(it);
            }
        }
        // cudaFree 隐含一次设备同步，因此回收"空闲但流上可能还有活"的块是安全的；
        // 这也是 empty_cache 与 OOM 补救路径会阻塞的原因。
        (void)cudaFree(seg.ptr);
        seg = Segment{};   // ptr == nullptr 标记它已归还
    }

    // 没有活动块时整张表都可清掉；否则跳过已归还的条目，让它们留到分配器本身被
    // 销毁为止。
    bool live = false;
    for (const auto& seg : m_segments) {
        if (seg.ptr != nullptr) {
            live = true;
            break;
        }
    }
    if (!live) m_segments.clear();
}

inline void CachingAllocator::empty_cache() noexcept {
    std::lock_guard<std::mutex> lock(m_mtx);
    release_free_segments();
}

inline CachingAllocator::Stats CachingAllocator::stats() const {
    std::lock_guard<std::mutex> lock(m_mtx);
    Stats out  = m_stats;
    out.cached_bytes = cached_bytes();
    return out;
}

inline CachingAllocator::~CachingAllocator() {
    const Stats final_stats = stats();
    // 计数器只在 segment 归还驱动之前有意义。
    if (const char* env = std::getenv("TENSORVIA_CUDA_ALLOC_STATS");
        env != nullptr && env[0] != '\0' && env[0] != '0') {
        const size_t total = final_stats.malloc_calls + final_stats.cache_hits;
        std::fprintf(stderr,
                     "[tensorvia] cuda allocator (device %d): cudaMalloc=%zu "
                     "cache_hits=%zu hit_rate=%.1f%% cached_bytes=%zu\n",
                     m_device, final_stats.malloc_calls, final_stats.cache_hits,
                     total == 0 ? 0.0
                                : 100.0 * static_cast<double>(final_stats.cache_hits) /
                                      static_cast<double>(total),
                     final_stats.cached_bytes);
    }
    // 在用的块意味着有张量活过了其上下文，而上下文的 shared_ptr 所有权已禁止这种
    // 情况；这里的拆卸工作只需释放缓存。
    release_free_segments();
}

}  // namespace tvv::cuda

#endif  // TVV_CUDA_ALLOCATOR_CUH
