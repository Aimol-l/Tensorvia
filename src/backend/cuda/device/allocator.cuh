
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
    // The caller must have made `device` current on the calling thread before
    // every allocate / deallocate (CUDAContext::activate does this at each
    // entry point; the allocator does not switch devices behind the caller's
    // back, matching what c10::cuda::CUDACachingAllocator relies on a guard for).
    explicit CachingAllocator(int device) noexcept : m_device(device) {}
    ~CachingAllocator();

    CachingAllocator(const CachingAllocator&) = delete;
    CachingAllocator& operator=(const CachingAllocator&) = delete;
    CachingAllocator(CachingAllocator&&) = delete;
    CachingAllocator& operator=(CachingAllocator&&) = delete;

    // Never returns null. Throws std::bad_alloc only after an OOM-triggered
    // empty_cache() failed to make room.
    void* allocate(size_t bytes, cudaStream_t stream);

    // noexcept: called from CUDATensor's destructor. An unknown pointer is
    // ignored rather than reported — there is nothing safe to do either way.
    void deallocate(void* ptr, cudaStream_t stream) noexcept;

    // Returns the segments that no live tensor references. A segment with any
    // block still in use stays, because its address space is shared.
    void empty_cache() noexcept;

    struct Stats {
        size_t malloc_calls = 0;   // cudaMalloc round-trips
        size_t cache_hits   = 0;   // allocations served from the free list
        size_t cached_bytes = 0;   // bytes currently held in free blocks
    };
    Stats stats() const;

private:
    // Small requests are carved out of a 2 MiB pool segment; large ones are
    // rounded up to a 2 MiB multiple. Both sizes come from torch.
    static constexpr size_t kMinBlock    = 512;
    static constexpr size_t kSmallSize   = size_t{1} << 20;   // 1 MiB
    static constexpr size_t kSmallBuffer = size_t{2} << 20;   // 2 MiB
    static constexpr size_t kLargeRound  = size_t{2} << 20;   // 2 MiB

    struct Block {
        void*        ptr     = nullptr;
        size_t       size    = 0;
        size_t       segment = 0;   // index into m_segments
        cudaStream_t stream  = nullptr;
        cudaEvent_t  event   = nullptr;   // recorded on release, lazily created
    };

    struct Segment {
        void*  ptr        = nullptr;   // nullptr once returned to the driver
        size_t total      = 0;
        size_t free_bytes = 0;
    };

    static size_t round_request(size_t bytes) noexcept;
    static size_t pool_size(size_t need) noexcept;

    // Require m_mtx.
    std::pair<void*, Block*> take_free(size_t need);
    void                     release(Block* block, cudaStream_t stream) noexcept;
    void                     split(Block* block, size_t need);
    void                     insert_free(Block* block) noexcept;
    void                     release_free_segments() noexcept;
    size_t                   cached_bytes() const noexcept;

    int                  m_device;
    mutable std::mutex   m_mtx;
    // Best-fit: lower_bound(need) is the smallest block that still fits.
    std::multimap<size_t, Block*>     m_free;
    std::unordered_map<void*, Block*> m_live;   // in-use blocks only
    std::vector<Segment>              m_segments;

    Stats m_stats;
};

// ---------------------------------------------------------------------------
// Implementation
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

    if (auto [ptr, block] = take_free(need); block != nullptr) {
        m_stats.cache_hits++;
        // The block's previous owner may still be running on another stream.
        // Waiting on its release event is a host-side enqueue, not a sync.
        if (block->event != nullptr && block->stream != stream) {
            const cudaError_t err = cudaStreamWaitEvent(stream, block->event);
            if (err != cudaSuccess) {
                // Put it back and fail loudly rather than hand out memory that
                // another stream may still be writing.
                insert_free(block);
                throw std::runtime_error(std::format(
                    "CUDA error in CachingAllocator::allocate: cudaStreamWaitEvent ({})",
                    cudaGetErrorName(err)));
            }
        }
        block->stream = stream;
        m_live[ptr] = block;
        return ptr;
    }

    const size_t pool = pool_size(need);
    void*        ptr  = nullptr;
    cudaError_t  err  = cudaMalloc(&ptr, pool);
    if (err != cudaSuccess) {
        // The only recovery: drop the cache and try once more.
        release_free_segments();
        err = cudaMalloc(&ptr, pool);
        if (err != cudaSuccess) throw std::bad_alloc();
    }
    m_stats.malloc_calls++;

    m_segments.push_back(Segment{ptr, pool, 0});
    auto* block = new Block{ptr, pool, m_segments.size() - 1, stream, nullptr};
    m_live[ptr] = block;
    split(block, need);   // hands the unused tail back to the free list
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

// Requires m_mtx: the block is already off the free list.
inline void CachingAllocator::release(Block* block, cudaStream_t stream) noexcept {
    block->stream = stream;
    if (block->event == nullptr &&
        cudaEventCreateWithFlags(&block->event, cudaEventDisableTiming) != cudaSuccess) {
        // Without an event there is nothing for a future stream to wait on, so
        // make the memory idle now. Rare (driver OOM on event creation) and the
        // one place where releasing may block.
        block->event = nullptr;
        (void)cudaStreamSynchronize(stream);
    }
    if (block->event != nullptr) (void)cudaEventRecord(block->event, stream);
    insert_free(block);
}

inline std::pair<void*, CachingAllocator::Block*> CachingAllocator::take_free(size_t need) {
    const auto it = m_free.lower_bound(need);
    if (it == m_free.end()) return {nullptr, nullptr};

    Block* block = it->second;
    m_free.erase(it);
    m_segments[block->segment].free_bytes -= block->size;
    // split() re-inserts the tail (and re-adds its bytes to the counter).
    split(block, need);
    return {block->ptr, block};
}

inline void CachingAllocator::split(Block* block, size_t need) {
    // A tail smaller than the minimum block size is not worth a free-list
    // entry; the allocation simply keeps it.
    if (block->size < need + kMinBlock) return;

    auto* tail  = new Block{static_cast<char*>(block->ptr) + need,
                            block->size - need, block->segment, block->stream, nullptr};
    block->size = need;
    insert_free(tail);
}

inline void CachingAllocator::insert_free(Block* block) noexcept {
    m_free.emplace(block->size, block);
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
        // A segment whose blocks are not all free shares its address space with
        // live tensors and cannot be handed back.
        if (seg.ptr == nullptr || seg.free_bytes != seg.total) continue;

        for (auto it = m_free.begin(); it != m_free.end();) {
            if (it->second->segment != i) {
                ++it;
                continue;
            }
            Block* block = it->second;
            if (block->event != nullptr) (void)cudaEventDestroy(block->event);
            delete block;
            it = m_free.erase(it);
        }
        (void)cudaFree(seg.ptr);
        seg = Segment{};   // ptr == nullptr marks it released
    }

    // With nothing live the whole table can go; otherwise released entries are
    // skipped and simply stay until the allocator itself is destroyed.
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
    // The counters are only meaningful before the segments go back to the driver.
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
    // In-use blocks belong to tensors that outlived their context, which the
    // context's shared_ptr ownership already forbids; freeing the cache is all
    // the teardown there is.
    release_free_segments();
}

}  // namespace tvv::cuda

#endif  // TVV_CUDA_ALLOCATOR_CUH
