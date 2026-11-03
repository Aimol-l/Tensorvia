#include "test_support.hpp"

#include <algorithm>
#include <chrono>

// P2 异步执行：算子入队后不在宿主等待；同步点只有文档列出的那几个。
// 异步是性能属性，只能用"入队耗时远小于到同步点为止的总耗时"来证明。这条断言
// 只有在 GPU 上每个算子的耗时明显压过宿主提交开销时才有意义——否则 GPU 在排队
// 期间就跑完了，总时长本来就约等于入队时长。各后端的提交开销差两个数量级
// （CUDA 十几微秒，Vulkan 在这台机器上约半毫秒），所以用例先量提交开销，再把
// 矩阵放大到"单次往返明显大于提交开销"为止。

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// 只入队 count 次矩阵乘，返回入队耗时；收尾调到同步点，让队列回到干净状态。
double time_enqueue(const Tensor& a, const Tensor& b, int count) {
    const auto start = Clock::now();
    Tensor last;
    for (int i = 0; i < count; ++i) last = ops::MatMul(a, b);
    const double elapsed = ms_since(start);
    ops::synchronize();
    return elapsed;
}

}  // namespace

// ---------- 入队立即返回 ----------
TEST_F(DeviceOps, EnqueueDoesNotWaitForCompletion) {
    if (dev == Device::CPU) GTEST_SKIP() << "CPU 后端是同步的，不存在入队/同步之分";
    if (!backend_available(dev)) GTEST_SKIP() << device_to_string(dev) << " is unavailable";

    // 1) 提交开销：8x8 的矩阵乘 GPU 工作可忽略，剩下的就是每次提交的宿主成本。
    Tensor tiny = Tensor::Ones({8, 8}, DataType::FLOAT32, dev);
    Tensor tiny_warm = ops::MatMul(tiny, tiny);
    host_values<float>(tiny_warm);
    constexpr int kProbeOps = 16;
    const double overhead_ms = std::max(time_enqueue(tiny, tiny, kProbeOps) / kProbeOps, 1e-4);

    // 2) 放大矩阵，直到单次"入队 + 同步"往返达到提交开销的 6 倍，即 GPU 工作
    //    至少是提交开销的 5 倍。
    //
    //    上限是 4096 而不是 2048：mcpp test 会并行跑同一设备上的其他测试二进制，
    //    别的进程占着 GPU 时每次提交能被驱动拖到几毫秒（空闲时的十倍以上），而
    //    内核越快、单次往返里的 GPU 工作越少，2048 就不够 6 倍了。这跟实现是否
    //    异步无关——是分母被外部进程抬高了。到 4096 仍不满足就说明这台机器此刻
    //    量不出这条性质，跳过而不是把一个被噪声支配的比值判成失败。
    int64_t side = 256;
    double round_ms = 0.0;
    Tensor a, b;
    while (true) {
        a = Tensor::Ones({side, side}, DataType::FLOAT32, dev);
        b = Tensor::Ones({side, side}, DataType::FLOAT32, dev);
        // 预热：首次启动要建流水线/上下文，不该算进任何一侧的耗时。
        Tensor warm = ops::MatMul(a, b);
        host_values<float>(warm);

        const auto round_start = Clock::now();
        Tensor one = ops::MatMul(a, b);
        ops::synchronize();
        round_ms = ms_since(round_start);
        if (round_ms >= 6.0 * overhead_ms || side >= 4096) break;
        side *= 2;
    }
    if (round_ms < 6.0 * overhead_ms) {
        GTEST_SKIP() << "machine too loaded to measure asynchrony: a " << side << "x"
                     << side << " round trip is " << round_ms
                     << " ms, the measured submit overhead is " << overhead_ms << " ms";
    }

    // 3) 按实测往返耗时决定做多少次，让总工作量落在几十毫秒。
    ASSERT_GT(round_ms, 0.0);
    const int ops = std::clamp(static_cast<int>(40.0 / round_ms), 4, 64);

    // 只入队、不读结果。
    const auto enqueue_start = Clock::now();
    Tensor last;
    for (int i = 0; i < ops; ++i) last = ops::MatMul(a, b);
    const double enqueue_ms = ms_since(enqueue_start);

    // 同步点：到这里为止的所有工作才必须完成。
    ops::synchronize();
    const double total_ms = ms_since(enqueue_start);

    std::println("[tensorvia] {} async: side {}, {} ops, enqueue {:.2f} ms, until sync {:.2f} ms",
                 device_to_string(dev), side, ops, enqueue_ms, total_ms);
    // 老行为（每次启动后 wait）下 enqueue_ms ≈ total_ms；异步后 enqueue 只含提交
    // 开销。留足 2 倍余量，避免机器抖动导致误报。
    EXPECT_GT(total_ms, 2.0 * enqueue_ms)
        << "enqueue took " << enqueue_ms << " ms of " << total_ms << " ms until synchronize";
}

// ---------- 同步点之后结果可见 ----------
TEST_F(DeviceOps, SynchronizeMakesPriorWorkVisible) {
    if (!backend_available(dev)) GTEST_SKIP() << device_to_string(dev) << " is unavailable";

    constexpr int64_t kSide = 64;
    Tensor a = Tensor::Ones({kSide, kSide}, DataType::FLOAT32, dev);

    // 只入队，中间的每一步都不碰宿主。
    Tensor acc = ops::MatMul(a, a);        // 64 * 1 = 64
    acc       = ops::MatMul(a, acc);       // 64 * 64 = 4096
    acc       = ops::MatMul(a, acc);       // 64 * 4096
    ops::synchronize();

    const auto values = host_values<float>(acc);
    ASSERT_EQ(values.size(), static_cast<size_t>(kSide * kSide));
    for (float v : values) EXPECT_FLOAT_EQ(v, 64.0f * 4096.0f);
}

// ---------- 同流内存复用 ----------
// 每轮都申请/释放同样大小的块：直接命中缓存分配器（CUDA）或等价路径，验证复用
// 不会把上一轮还在用的数据交出去。
TEST_F(DeviceOps, RepeatedAllocFreeKeepsValuesCorrect) {
    if (!backend_available(dev)) GTEST_SKIP() << device_to_string(dev) << " is unavailable";

    Tensor seed = Tensor::Ones({32, 32}, DataType::FLOAT32, dev);
    for (int round = 0; round < 200; ++round) {
        Tensor scratch = seed + seed;          // 新分配
        Tensor scaled  = scratch * 3.0f;       // 又一次新分配，同时释放 scratch
        const auto values = host_values<float>(scaled);
        ASSERT_EQ(values.size(), 1024u) << "round " << round;
        EXPECT_FLOAT_EQ(values.front(), 6.0f) << "round " << round;
        EXPECT_FLOAT_EQ(values.back(), 6.0f) << "round " << round;
    }
    ops::synchronize();
}

// ---------- 每线程各自一条队列：并发提交互不干扰 ----------
// 每条线程只碰自己的张量，并在自己的同步点上读回结果。每（线程, 设备实例）一条
// 队列时，这些提交真正并发；共享一条队列时它们只是被串行化，结果同样必须正确。
TEST_F(DeviceOps, ConcurrentThreadsKeepTheirOwnResults) {
    if (dev == Device::CPU) GTEST_SKIP() << "CPU 后端没有队列，并发提交不经过异步路径";
    if (!backend_available(dev)) GTEST_SKIP() << device_to_string(dev) << " is unavailable";

    constexpr int kThreads = 4;
    constexpr int kRounds = 8;
    std::array<std::atomic<int>, kThreads> mismatches{};

    std::vector<std::thread> workers;
    for (int i = 0; i < kThreads; ++i) {
        workers.emplace_back([&, i] {
            const float base = static_cast<float>(i + 1);
            for (int round = 0; round < kRounds; ++round) {
                // 全部是线程私有张量：跨线程没有任何共享写入。
                Tensor a = Tensor::Fill({32, 32}, base, DataType::FLOAT32, dev);
                Tensor b = ops::MatMul(a, a);           // 32 * base * base
                Tensor c = b + a;                       // 32 * base^2 + base
                // 只在本线程的同步点上读：这条队列之前的提交必须先完成。
                const auto values = host_values<float>(c);
                const float expected = 32.0f * base * base + base;
                for (float v : values) {
                    if (std::abs(v - expected) > kEps) {
                        mismatches[i].fetch_add(1);
                        break;
                    }
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();

    for (int i = 0; i < kThreads; ++i) {
        EXPECT_EQ(mismatches[i].load(), 0) << "thread " << i;
    }
    EXPECT_NO_THROW(ops::synchronize());
}

// ---------- 设备级控制覆盖所有线程 ----------
// 每条线程各自入队，最后由主线程 synchronize：所有线程的流都要被覆盖。
TEST_F(DeviceOps, SynchronizeCoversOtherThreadsWork) {
    if (dev == Device::CPU) GTEST_SKIP() << "CPU 后端没有流，等价于 no-op";
    if (!backend_available(dev)) GTEST_SKIP() << device_to_string(dev) << " is unavailable";

    std::vector<Tensor> results(4);
    std::vector<std::thread> workers;
    for (size_t i = 0; i < results.size(); ++i) {
        workers.emplace_back([&, i] {
            Tensor local = Tensor::Fill({16, 16}, static_cast<float>(i + 1),
                                        DataType::FLOAT32, dev);
            Tensor doubled = local * 2.0f;
            results[i] = doubled;   // 只入队，不在工作线程上读
        });
    }
    for (auto& worker : workers) worker.join();

    ops::synchronize();
    for (size_t i = 0; i < results.size(); ++i) {
        const auto values = host_values<float>(results[i]);
        ASSERT_EQ(values.size(), 256u) << "thread " << i;
        EXPECT_FLOAT_EQ(values.front(), static_cast<float>(2 * (i + 1))) << "thread " << i;
    }
}
