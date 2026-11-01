// 基准：每个用例输出三段耗时——submit（调用返回）/ sync（synchronize() 耗时）/
// total（两者之和）——以及按总耗时计算的吞吐。冷启动（首次调用，含 SYCL JIT、
// Vulkan 流水线创建、CUDA 模块加载）单独报告，不并入热态中位数。
//
// 为什么必须三段：异步后端的调用返回不等于算完。把三段混成一个数字，就无法区分
// 内核、传输与同步各自占多少（FR-4.1）；CPU 后端的 sync 近似为 0，仍照常输出。
// 吞吐按 total 计算——那才是设备侧真正算完的时间。
//
// 用法: bench [cpu|cuda|vulkan|sycl]（与 examples/ 的参数约定一致；不传参数时测
// 本构建编译进来的后端，请求的后端没编进本构建或机器上没有时明确退回，不静默降级）。
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <print>
#include <string_view>
#include <vector>

#include "tensorvia.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// 三段计时 + 冷启动。f() 是被测调用；冷启动那次也排空队列，让"首次"包含全部
// 一次性成本（JIT / 流水线 / 模块加载）。
struct Phases {
    double submit_ms = 0;  // 调用返回
    double sync_ms   = 0;  // synchronize() 耗时
    double total_ms  = 0;  // submit + sync
    double cold_ms   = 0;  // 首次调用（含一次性成本）
};

template <class F>
Phases measure(F&& f, int reps = 9) {
    Phases p;
    const auto cold0 = Clock::now();
    f();
    ops::synchronize();
    p.cold_ms = ms_since(cold0);

    std::vector<double> submit, sync, total;
    submit.reserve(reps);
    sync.reserve(reps);
    total.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        const auto t0 = Clock::now();
        f();
        const auto t1 = Clock::now();
        ops::synchronize();
        const auto t2 = Clock::now();
        submit.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        sync.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        total.push_back(std::chrono::duration<double, std::milli>(t2 - t0).count());
    }
    p.submit_ms = median(submit);
    p.sync_ms   = median(sync);
    p.total_ms  = median(total);
    return p;
}

void report(std::string_view name, const Phases& p, double bytes, bool flops_mode = false) {
    const double v = bytes / (p.total_ms / 1e3);
    std::println("{:<24} submit {:>8.3f}  sync {:>8.3f}  total {:>9.3f} ms  {:>8.2f} {}"
                 "  cold {:>9.3f} ms",
                 name, p.submit_ms, p.sync_ms, p.total_ms, v / 1e9,
                 flops_mode ? "GFLOPS" : "GB/s", p.cold_ms);
}

// 与 examples/ 一致的设备解析：请求的后端不可用时明确退回默认后端，不静默降级。
constexpr std::string_view kDeviceNames[] = {"cpu", "cuda", "vulkan", "sycl"};
constexpr Device kDeviceKinds[] = {Device::CPU, Device::CUDA, Device::VULKAN, Device::SYCL};

// 默认测本构建编译进来的后端：mcpp test 的各 lane 因此各测各的（CUDA lane 测
// CUDA，而不是莫名其妙地测 CPU）。与 tests/test_support.hpp 的 kTestDevice 同一
// 套优先级。
constexpr Device kDefaultDevice =
#if defined(BACKEND_VULKAN)
    Device::VULKAN;
#elif defined(BACKEND_CUDA)
    Device::CUDA;
#elif defined(BACKEND_SYCL)
    Device::SYCL;
#else
    Device::CPU;
#endif

Device resolve_device(std::string_view name) {
    for (std::size_t i = 0; i < std::size(kDeviceNames); ++i) {
        if (name != kDeviceNames[i]) continue;
        if (kDeviceKinds[i] == kDefaultDevice) return kDefaultDevice;
        if (kDeviceKinds[i] == Device::CPU) return Device::CPU;  // CPU 总在
        if (ops::device_count(kDeviceKinds[i]) > 0) return kDeviceKinds[i];
        std::println("{} is not available in this build; falling back to {}",
                     device_to_string(kDeviceKinds[i]), device_to_string(kDefaultDevice));
        return kDefaultDevice;
    }
    std::println("unknown device '{}'; falling back to {}", name,
                 device_to_string(kDefaultDevice));
    return kDefaultDevice;
}

}  // namespace

int main(int argc, char* argv[]) {
    using DT = DataType;
    const Device dev = argc > 1 ? resolve_device(argv[1]) : kDefaultDevice;
    std::println("device {} | reps 9 (heavy 5) | throughput on total time", device_to_string(dev));

    {   // 带宽上限参照：同 dtype 拷贝（读 + 写）
        constexpr std::int64_t N = std::int64_t{1} << 24;  // 16M f32
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32, dev);
        const Phases p = measure([&] { (void)a.clone(); });
        report("memcpy f32 16M", p, 2.0 * double(N) * 4);
    }
    {   // element-wise add：读 a+b 写 dst，带宽型
        constexpr std::int64_t N = std::int64_t{1} << 24;  // 16M f32
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor b = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor d({N}, DT::FLOAT32, dev);
        const Phases p = measure([&] { ops::Add(a, b, d); });
        report("add f32 16M", p, 3.0 * double(N) * 4);
    }
    {   // 逐元素乘法
        constexpr std::int64_t N = std::int64_t{1} << 24;
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor b = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor d({N}, DT::FLOAT32, dev);
        const Phases p = measure([&] { ops::Mul(a, b, d); });
        report("mul f32 16M", p, 3.0 * double(N) * 4);
    }
    {   // 标量 in-place：读写各一次
        constexpr std::int64_t N = std::int64_t{1} << 24;
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32, dev);
        const Phases p = measure([&] { ops::Add(a, 1.5f); });
        report("add-scalar f32 16M", p, 2.0 * double(N) * 4);
    }
    {   // 超越函数（CPU 侧标量 libm，无法自动向量化）
        constexpr std::int64_t N = std::int64_t{1} << 22;  // 4M
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32, dev);
        const Phases p = measure([&] { (void)ops::Exp(a); }, 5);
        report("exp f32 4M", p, 2.0 * double(N) * 4);
    }
    {   // 全量 reduce
        constexpr std::int64_t N = std::int64_t{1} << 24;
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32, dev);
        const Phases p = measure([&] { (void)ops::Sum(a); });
        report("sum f32 16M", p, double(N) * 4);
    }
    {   // 带轴 reduce：axis=1（内存布局连续方向）
        Tensor a = Tensor::Random({4096, 4096}, 0.0f, 1.0f, DT::FLOAT32, dev);
        const Phases p = measure([&] { (void)ops::Sum(a, 1); }, 5);
        report("sum-axis1 4096x4096", p, double(4096) * 4096 * 4);
    }
    {   // 2D 转置（tiled kernel）
        Tensor a = Tensor::Random({1024, 1024}, 0.0f, 1.0f, DT::FLOAT32, dev);
        const Phases p = measure([&] { ops::Transpose(a); }, 5);
        report("transpose2d 1024x1024", p, 2.0 * 1024 * 1024 * 4);
    }
    {   // N 维转置（逐元素 div/mod + 堆分配）
        Tensor a = Tensor::Random({64, 64, 64}, 0.0f, 1.0f, DT::FLOAT32, dev);
        const Phases p = measure([&] { (void)ops::Transpose(a, {2, 0, 1}); }, 5);
        report("transpose3d 64^3", p, 2.0 * 64 * 64 * 64 * 4);
    }
    {   // 非连续视图物化（clone_as_contiguous gather）
        Tensor a = Tensor::Random({128, 64, 64}, 0.0f, 1.0f, DT::FLOAT32, dev);
        auto p = a.permute({2, 0, 1});
        const Phases ph = measure([&] { (void)p.contiguous(); }, 5);
        report("gather permute 128x64x64", ph, 2.0 * 128 * 64 * 64 * 4);
    }
    {   // matmul：256 / 512 / 1024 三档
        constexpr std::int64_t M = 256;
        Tensor a = Tensor::Random({M, M}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor b = Tensor::Random({M, M}, 0.0f, 1.0f, DT::FLOAT32, dev);
        const Phases p = measure([&] { (void)ops::MatMul(a, b); }, 5);
        report("matmul f32 256", p, 2.0 * M * M * M, true);
    }
    {
        constexpr std::int64_t M = 512;
        Tensor a = Tensor::Random({M, M}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor b = Tensor::Random({M, M}, 0.0f, 1.0f, DT::FLOAT32, dev);
        const Phases p = measure([&] { (void)ops::MatMul(a, b); }, 5);
        report("matmul f32 512", p, 2.0 * M * M * M, true);
    }
    {
        constexpr std::int64_t M = 1024;
        Tensor a = Tensor::Random({M, M}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor b = Tensor::Random({M, M}, 0.0f, 1.0f, DT::FLOAT32, dev);
        const Phases p = measure([&] { (void)ops::MatMul(a, b); }, 5);
        report("matmul f32 1024", p, 2.0 * M * M * M, true);
    }
    {   // arange：单写（设备生成器形式：宿主算值一次性搬到目标设备）
        constexpr std::int64_t N = std::int64_t{1} << 24;  // 16M
        TensorOptions opts(DT::FLOAT32);
        opts.with_device(dev, 0);
        const Phases p = measure([&] { (void)ops::Arange(0.0f, float(N), 1.0f, opts); });
        report("arange f32 16M", p, double(N) * 4);
    }
    {   // where：cond + a + b 读 + out 写
        constexpr std::int64_t N = std::int64_t{1} << 24;
        Tensor cond = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32, dev) > 0.5f;
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor b = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32, dev);
        const Phases p = measure([&] { (void)ops::Where(cond, a, b); });
        report("where f32 16M", p, 4.0 * double(N) * 4);
    }
    {   // where 标量：cond + a 读 + out 写
        constexpr std::int64_t N = std::int64_t{1} << 24;
        Tensor cond = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32, dev) > 0.5f;
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32, dev);
        const Phases p = measure([&] { (void)ops::Where(cond, a, 0.5f); });
        report("where-scalar f32 16M", p, 3.0 * double(N) * 4);
    }
    {   // index_select：embedding 查表（V=4096, D=64, N=4096）
        constexpr std::int64_t V = 4096, D = 64, N = 4096;
        Tensor emb = Tensor::Random({V, D}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor idx = Tensor::Random({N}, 0.0f, float(V - 1), DT::INT32, dev);
        const Phases p = measure([&] { (void)ops::IndexSelect(emb, 0, idx); });
        report("index_select 4096x64", p, 2.0 * double(N) * D * 4);
    }
    {   // gather：out + in + idx 各一次访问
        constexpr std::int64_t M = 1024;
        Tensor g = Tensor::Random({M, M}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor gi = Tensor::Random({M, M}, 0.0f, float(M - 1), DT::INT32, dev);
        const Phases p = measure([&] { (void)ops::Gather(g, 1, gi); });
        report("gather 1024x1024", p, 3.0 * double(M) * M * 4);
    }
    {   // layer_norm：融合 weight/bias，AVX2 归约 + FMA 仿射
        constexpr std::int64_t R = 8192, H = 4096;
        Tensor a = Tensor::Random({R, H}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor w = Tensor::Random({H}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor b = Tensor::Random({H}, 0.0f, 1.0f, DT::FLOAT32, dev);
        const Phases p = measure([&] { (void)ops::LayerNorm(a, w, b, 1, 1e-5f); }, 5);
        report("layer_norm f32 8192x4096", p, 4.0 * double(R) * H * 4);
    }
    {   // rms_norm：AVX2 归约 mean(x²) + 乘 weight
        constexpr std::int64_t R = 8192, H = 4096;
        Tensor a = Tensor::Random({R, H}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor w = Tensor::Random({H}, 0.0f, 1.0f, DT::FLOAT32, dev);
        const Phases p = measure([&] { (void)ops::RmsNorm(a, w, 1, 1e-5f); }, 5);
        report("rms_norm f32 8192x4096", p, 3.0 * double(R) * H * 4);
    }
    {   // randn：写 N 个高斯样本
        constexpr std::int64_t N = std::int64_t{1} << 24;
        const Phases p = measure([&] { (void)ops::Randn({N}, DT::FLOAT32, 0.0f, 1.0f, dev); });
        report("randn f32 16M", p, double(N) * 4);
    }
    {   // mse_loss：读 a+b，归约平方差（仅 CPU 实现，设备 lane 跳过）
        constexpr std::int64_t N = std::int64_t{1} << 24;
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor b = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32, dev);
        if (dev == Device::CPU) {
            const Phases p = measure([&] { (void)ops::MseLoss(a, b); });
            report("mse_loss f32 16M", p, 2.0 * double(N) * 4);
        }
    }
    {   // nll_loss：log_prob 查表 + 归约（仅 CPU 实现，设备 lane 跳过）
        constexpr std::int64_t N = 65536, C = 1024;
        Tensor logp = Tensor::Random({N, C}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor tgt = Tensor::Random({N}, 0.0f, float(C - 1), DT::INT64, dev);
        if (dev == Device::CPU) {
            const Phases p = measure([&] { (void)ops::NllLoss(logp, tgt); });
            report("nll_loss f32 65536x1024", p, double(N) * 4);
        }
    }
    {   // cross_entropy：log_softmax + nll（仅 CPU 实现，设备 lane 跳过）
        constexpr std::int64_t N = 8192, C = 256;
        Tensor logits = Tensor::Random({N, C}, 0.0f, 1.0f, DT::FLOAT32, dev);
        Tensor tgt = Tensor::Random({N}, 0.0f, float(C - 1), DT::INT64, dev);
        if (dev == Device::CPU) {
            const Phases p = measure([&] { (void)ops::CrossEntropyLoss(logits, tgt); }, 5);
            report("cross_entropy f32 8192x256", p, 4.0 * double(N) * C * 4);
        }
    }
    return 0;
}
