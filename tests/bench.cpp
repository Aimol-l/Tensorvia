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

template <class F>
double median_ms(F&& f, int reps = 5) {
    f();  // warmup（填充缓存 / JIT 线程池）
    std::vector<double> samples;
    samples.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void report(std::string_view name, double ms, double bytes, bool flops_mode = false) {
    const double v = bytes / (ms / 1e3);
    if (flops_mode)
        std::println("{:<26} {:>10.3f} ms   {:>8.2f} GFLOPS", name, ms, v / 1e9);
    else
        std::println("{:<26} {:>10.3f} ms   {:>8.2f} GB/s", name, ms, v / 1e9);
}

}  // namespace

int main(int argc, char* argv[]) {
    
    using DT = DataType;

    {   // element-wise add：读 a+b 写 dst，带宽型
        constexpr std::int64_t N = std::int64_t{1} << 24;  // 16M f32
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32);
        Tensor b = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32);
        Tensor d({N}, DT::FLOAT32);
        double ms = median_ms([&] { ops::Add(a, b, d); });
        report("add f32 16M", ms, 3.0 * double(N) * 4);
    }
    {   // element-wise mul
        constexpr std::int64_t N = std::int64_t{1} << 24;
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32);
        Tensor b = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32);
        Tensor d({N}, DT::FLOAT32);
        double ms = median_ms([&] { ops::Mul(a, b, d); });
        report("mul f32 16M", ms, 3.0 * double(N) * 4);
    }
    {   // 标量 in-place：读写各一次
        constexpr std::int64_t N = std::int64_t{1} << 24;
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32);
        double ms = median_ms([&] { ops::Add(a, 1.5f); });
        report("add-scalar f32 16M", ms, 2.0 * double(N) * 4);
    }
    {   // 超越函数（当前标量 libm，无法自动向量化）
        constexpr std::int64_t N = std::int64_t{1} << 22;  // 4M
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32);
        double ms = median_ms([&] { ops::Exp(a); }, 3);
        report("exp f32 4M", ms, 2.0 * double(N) * 4);
    }
    {   // 全量 reduce
        constexpr std::int64_t N = std::int64_t{1} << 24;
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32);
        double ms = median_ms([&] { (void)ops::Sum(a); });
        report("sum f32 16M", ms, double(N) * 4);
    }
    {   // 带轴 reduce：axis=1（内存布局连续方向）
        Tensor a = Tensor::Random({4096, 4096}, 0.0f, 1.0f, DT::FLOAT32);
        double ms = median_ms([&] { (void)ops::Sum(a, 1); }, 3);
        report("sum-axis1 4096x4096", ms, double(4096) * 4096 * 4);
    }
    {   // 2D 转置（tiled kernel）
        Tensor a = Tensor::Random({1024, 1024}, 0.0f, 1.0f, DT::FLOAT32);
        double ms = median_ms([&] { ops::Transpose(a); }, 3);
        report("transpose2d 1024x1024", ms, 2.0 * 1024 * 1024 * 4);
    }
    {   // N 维转置（当前逐元素 div/mod + 堆分配）
        Tensor a = Tensor::Random({64, 64, 64}, 0.0f, 1.0f, DT::FLOAT32);
        double ms = median_ms([&] { (void)ops::Transpose(a, {2, 0, 1}); }, 3);
        report("transpose3d 64^3", ms, 2.0 * 64 * 64 * 64 * 4);
    }
    {   // 非连续视图物化（clone_as_contiguous gather）
        Tensor a = Tensor::Random({128, 64, 64}, 0.0f, 1.0f, DT::FLOAT32);
        auto p = a.permute({2, 0, 1});
        double ms = median_ms([&] { (void)p.contiguous(); }, 3);
        report("gather permute 128x64x64", ms, 2.0 * 128 * 64 * 64 * 4);
    }
    {   // matmul（当前 naive / tiled）
        constexpr std::int64_t M = 256;
        Tensor a = Tensor::Random({M, M}, 0.0f, 1.0f, DT::FLOAT32);
        Tensor b = Tensor::Random({M, M}, 0.0f, 1.0f, DT::FLOAT32);
        double ms = median_ms([&] { (void)ops::MatMul(a, b); }, 3);
        report("matmul f32 256", ms, 2.0 * M * M * M, true);
    }
    {   // matmul 大矩阵
        constexpr std::int64_t M = 512;
        Tensor a = Tensor::Random({M, M}, 0.0f, 1.0f, DT::FLOAT32);
        Tensor b = Tensor::Random({M, M}, 0.0f, 1.0f, DT::FLOAT32);
        double ms = median_ms([&] { (void)ops::MatMul(a, b); }, 3);
        report("matmul f32 512", ms, 2.0 * M * M * M, true);
    }
    {   // arange：单写
        constexpr std::int64_t N = std::int64_t{1} << 24;  // 16M
        double ms = median_ms([&] { (void)ops::Arange(0.0f, float(N), 1.0f); });
        report("arange f32 16M", ms, double(N) * 4);
    }
    {   // where：cond + a + b 读 + out 写
        constexpr std::int64_t N = std::int64_t{1} << 24;
        Tensor cond = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32) > 0.5f;
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32);
        Tensor b = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32);
        double ms = median_ms([&] { (void)ops::Where(cond, a, b); });
        report("where f32 16M", ms, 4.0 * double(N) * 4);
    }
    {   // where 标量：cond + a 读 + out 写
        constexpr std::int64_t N = std::int64_t{1} << 24;
        Tensor cond = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32) > 0.5f;
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32);
        double ms = median_ms([&] { (void)ops::Where(cond, a, 0.5f); });
        report("where-scalar f32 16M", ms, 3.0 * double(N) * 4);
    }
    {   // index_select：embedding 查表（V=4096, D=64, N=4096）
        constexpr std::int64_t V = 4096, D = 64, N = 4096;
        Tensor emb = Tensor::Random({V, D}, 0.0f, 1.0f, DT::FLOAT32);
        Tensor idx = Tensor::Random({N}, 0.0f, float(V - 1), DT::INT32);
        double ms = median_ms([&] { (void)ops::IndexSelect(emb, 0, idx); });
        report("index_select 4096x64", ms, 2.0 * double(N) * D * 4);
    }
    {   // gather：out + in + idx 各一次访问
        constexpr std::int64_t M = 1024;
        Tensor g = Tensor::Random({M, M}, 0.0f, 1.0f, DT::FLOAT32);
        Tensor gi = Tensor::Random({M, M}, 0.0f, float(M - 1), DT::INT32);
        double ms = median_ms([&] { (void)ops::Gather(g, 1, gi); });
        report("gather 1024x1024", ms, 3.0 * double(M) * M * 4);
    }
    {   // layer_norm：融合 weight/bias，AVX2 归约 + FMA 仿射
        constexpr std::int64_t R = 8192, H = 4096;
        Tensor a = Tensor::Random({R, H}, 0.0f, 1.0f, DT::FLOAT32);
        Tensor w = Tensor::Random({H}, 0.0f, 1.0f, DT::FLOAT32);
        Tensor b = Tensor::Random({H}, 0.0f, 1.0f, DT::FLOAT32);
        double ms = median_ms([&] { (void)ops::LayerNorm(a, w, b, 1, 1e-5f); }, 3);
        report("layer_norm f32 8192x4096", ms, 4.0 * double(R) * H * 4);
    }
    {   // rms_norm：AVX2 归约 mean(x²) + 乘 weight
        constexpr std::int64_t R = 8192, H = 4096;
        Tensor a = Tensor::Random({R, H}, 0.0f, 1.0f, DT::FLOAT32);
        Tensor w = Tensor::Random({H}, 0.0f, 1.0f, DT::FLOAT32);
        double ms = median_ms([&] { (void)ops::RmsNorm(a, w, 1, 1e-5f); }, 3);
        report("rms_norm f32 8192x4096", ms, 3.0 * double(R) * H * 4);
    }
    {   // randn：写 N 个高斯样本
        constexpr std::int64_t N = std::int64_t{1} << 24;
        double ms = median_ms([&] { (void)ops::Randn({N}, DT::FLOAT32, 0.0f, 1.0f); });
        report("randn f32 16M", ms, double(N) * 4);
    }
    {   // mse_loss：读 a+b，归约平方差
        constexpr std::int64_t N = std::int64_t{1} << 24;
        Tensor a = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32);
        Tensor b = Tensor::Random({N}, 0.0f, 1.0f, DT::FLOAT32);
        double ms = median_ms([&] { (void)ops::MseLoss(a, b); });
        report("mse_loss f32 16M", ms, 2.0 * double(N) * 4);
    }
    {   // nll_loss：log_prob 查表 + 归约
        constexpr std::int64_t N = 65536, C = 1024;
        Tensor logp = Tensor::Random({N, C}, 0.0f, 1.0f, DT::FLOAT32);
        Tensor tgt = Tensor::Random({N}, 0.0f, float(C - 1), DT::INT64);
        double ms = median_ms([&] { (void)ops::NllLoss(logp, tgt); });
        report("nll_loss f32 65536x1024", ms, double(N) * 4);
    }
    {   // cross_entropy：log_softmax（标量 exp）+ nll
        constexpr std::int64_t N = 8192, C = 256;
        Tensor logits = Tensor::Random({N, C}, 0.0f, 1.0f, DT::FLOAT32);
        Tensor tgt = Tensor::Random({N}, 0.0f, float(C - 1), DT::INT64);
        double ms = median_ms([&] { (void)ops::CrossEntropyLoss(logits, tgt); }, 3);
        report("cross_entropy f32 8192x256", ms, 4.0 * double(N) * C * 4);
    }
    return 0;
}
