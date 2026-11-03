#include "test_support.hpp"

#include <cstring>
#include <format>

// 设备生命周期、多设备寻址与 CUDA context/stream 用例。
// 这些用例看的是后端自身，不只是张量 API。
#if defined(BACKEND_CUDA)
// 流相关用例要看 CUDAContext 本身，无法只靠公开的张量 API 表达。
#include "tensorvia/backend/cuda/cuda_context.hpp"
#include "tensorvia/backend/cuda/cuda_tensor.hpp"
#endif
#if defined(BACKEND_VULKAN)
// 通道相关用例要看 VulkanContext 本身的队列池。
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"
#endif

// ---------- 0. 被测后端 ----------
// 先报告这次构建选中的后端在这台机器上能不能跑，避免把 CPU 的结果误当成
// 加速器路径的结果。
TEST(Backend, AvailableDevices) {
    std::size_t available = 0;
    std::println("[tensorvia] compiled backend under test: {}", device_to_string(kTestDevice));
    for (Device device : kAllDevices) {
        const bool ok = backend_available(device);
        available += ok ? 1 : 0;
        std::println("[tensorvia]   {:<7}{}", device_to_string(device),ok ? "✅" : "❌");
    }
    EXPECT_GE(available, 1u) << "no tensor backend is usable in this build";
}


// ---------- 0b. 设备相关用例：跑在本次编译启用的后端上 ----------
// 夹具 DeviceOps 见 test_support.hpp（p0.cpp 也要用）。
TEST_F(DeviceOps, ElementwiseAndMatMul) {
    std::vector<float> av{1, 2, 3, 4, 5, 6};  // 2x3
    std::vector<float> bv{6, 5, 4, 3, 2, 1};  // 2x3
    std::vector<float> cv{1, 0, 2, 1, 0, 1};  // 3x2，matmul 的右操作数

    const Tensor a_cpu(av, {2, 3}), b_cpu(bv, {2, 3}), c_cpu(cv, {3, 2});
    const auto add_ref = host_values<float>(a_cpu + b_cpu);
    const auto sub_ref = host_values<float>(a_cpu - b_cpu);
    const auto mul_ref = host_values<float>(a_cpu * b_cpu);
    const auto div_ref = host_values<float>(a_cpu / b_cpu);
    const auto sca_ref = host_values<float>(a_cpu * 2.0f);
    const auto mm_ref  = host_values<float>(ops::MatMul(a_cpu, c_cpu));

    const Tensor a = moved_to(a_cpu, dev);
    const Tensor b = moved_to(b_cpu, dev);
    const Tensor c = moved_to(c_cpu, dev);

    const auto sum = a + b;
    EXPECT_TRUE(sum.device() == dev) << "result must stay on " << device_to_string(dev);
    expect_close(host_values<float>(sum), add_ref, "add");
    expect_close(host_values<float>(a - b), sub_ref, "sub");
    expect_close(host_values<float>(a * b), mul_ref, "mul");
    expect_close(host_values<float>(a / b), div_ref, "div");
    expect_close(host_values<float>(a * 2.0f), sca_ref, "mul-scalar");
    expect_close(host_values<float>(ops::MatMul(a, c)), mm_ref, "matmul");
}

TEST_F(DeviceOps, MatMulHalfPrecision) {
    // 半精度 matmul：Vulkan 上启用 cooperative matrix 后这条路径会换成 tensor
    // core 内核（口径见 VulkanCapabilities.CooperativeMatrixReflectsDeviceState），
    // 其余后端是标量内核，两者必须给出相同结果。
    //
    // 取值都是 0.25 的倍数、P 和不超过 256，f16/bf16 与 f32 都能精确表示，所以
    // 参考值只需按半精度的舍入误差放宽，不必给累加漂移留余量。形状覆盖三种
    // 边界：M/N/K 都不整除 tile、整块对齐、以及带 batch 的 3-D。
    auto check = [&](DataType dtype, std::int64_t batch, std::int64_t M, std::int64_t K,
                     std::int64_t N) {
        const std::int64_t elems_a = batch * M * K;
        const std::int64_t elems_b = batch * K * N;
        std::vector<float> av(elems_a), bv(elems_b);
        for (std::int64_t i = 0; i < elems_a; ++i) av[i] = float((i * 7) % 5) * 0.5f;
        for (std::int64_t i = 0; i < elems_b; ++i) bv[i] = float((i * 11) % 7) * 0.25f;

        const Tensor a32(av, {batch, M, K});
        const Tensor b32(bv, {batch, K, N});
        // 参考值：f32 精确相乘，再按目标 dtype 舍入一次。
        const Tensor ref32 = ops::MatMul(a32, b32);
        const Tensor ref   = ops::Typecast(ref32, dtype);
        const auto   want  = host_values<float>(ops::Typecast(ref, DataType::FLOAT32));

        const Tensor a = moved_to(ops::Typecast(a32, dtype), dev);
        const Tensor b = moved_to(ops::Typecast(b32, dtype), dev);
        const Tensor c = ops::MatMul(a, b);
        ASSERT_TRUE(c.device() == dev) << "result must stay on " << device_to_string(dev);
        const auto got = host_values<float>(ops::Typecast(c, DataType::FLOAT32));

        ASSERT_EQ(got.size(), want.size()) << "matmul " << dtype_to_string(dtype);
        // 两三个 ulp 的余量：bfloat16 只有 8 位有效位，相对误差约 0.8%。
        const double tol = dtype == DataType::FLOAT16 ? 4e-3 : 3e-2;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const double scale = std::max(1.0, std::abs(double(want[i])));
            EXPECT_NEAR(got[i], want[i], tol * scale)
                << "matmul " << dtype_to_string(dtype) << " batch=" << batch << " M=" << M
                << " K=" << K << " N=" << N << " at index " << i;
        }
    };

    for (const DataType dtype : {DataType::FLOAT16, DataType::BFLOAT16}) {
        check(dtype, 1, 70, 70, 70);   // 三个维度都留下半块 tile
        check(dtype, 1, 128, 128, 128);  // 整块对齐
        check(dtype, 2, 80, 48, 96);   // 批处理 + K 不整除 64
    }
}

TEST_F(DeviceOps, UnaryMath) {
    std::vector<float> av{0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f};  // log/sqrt 需要正数
    const Tensor a_cpu(av, {2, 3});

    const auto exp_ref  = host_values<float>(ops::Exp(a_cpu));
    const auto sqrt_ref = host_values<float>(ops::Sqrt(a_cpu));
    const auto log_ref  = host_values<float>(ops::Log(a_cpu, 2.0f));
    const auto abs_ref  = host_values<float>(ops::Abs(a_cpu));
    const auto relu_ref = host_values<float>(ops::Relu(a_cpu));

    const Tensor a = moved_to(a_cpu, dev);
    expect_close(host_values<float>(ops::Exp(a)), exp_ref, "exp");
    expect_close(host_values<float>(ops::Sqrt(a)), sqrt_ref, "sqrt");
    expect_close(host_values<float>(ops::Log(a, 2.0f)), log_ref, "log");
    expect_close(host_values<float>(ops::Abs(a)), abs_ref, "abs");
    expect_close(host_values<float>(ops::Relu(a)), relu_ref, "relu");
}

TEST_F(DeviceOps, ReductionAndSoftmax) {
    std::vector<float> av{1, 2, 3, 4, 5, 6};
    const Tensor a_cpu(av, {2, 3});
    const float sum_ref  = ops::Sum(a_cpu).item<float>();
    const float mean_ref = ops::Mean(a_cpu).item<float>();
    const float min_ref  = ops::Min(a_cpu).item<float>();
    const float max_ref  = ops::Max(a_cpu).item<float>();
    const auto axis_ref  = host_values<float>(ops::Sum(a_cpu, 1));
    const auto sm_ref    = host_values<float>(ops::Softmax(a_cpu, 1));

    const Tensor a = moved_to(a_cpu, dev);
    EXPECT_NEAR(ops::Sum(a).item<float>(), sum_ref, kEps) << "sum on " << device_to_string(dev);
    EXPECT_NEAR(ops::Mean(a).item<float>(), mean_ref, kEps) << "mean on " << device_to_string(dev);
    EXPECT_NEAR(ops::Min(a).item<float>(), min_ref, kEps) << "min on " << device_to_string(dev);
    EXPECT_NEAR(ops::Max(a).item<float>(), max_ref, kEps) << "max on " << device_to_string(dev);
    expect_close(host_values<float>(ops::Sum(a, 1)), axis_ref, "sum-axis");
    expect_close(host_values<float>(ops::Softmax(a, 1)), sm_ref, "softmax");

    // FR-3 在设备上同样成立：全归约是 0-dim，keepdim 与负轴不需要落回 CPU。
    Tensor s = ops::Sum(a);
    EXPECT_EQ(s.dims(), 0u) << "sum on " << device_to_string(dev);
    EXPECT_EQ(s.device_id(), a.device_id());
    EXPECT_NEAR(s.item<float>(), sum_ref, kEps) << "sum-item on " << device_to_string(dev);

    auto keep = ops::Sum(a, 1, true);
    EXPECT_EQ(keep.dims(), 2u) << "keepdim on " << device_to_string(dev);
    EXPECT_EQ(keep.shape(1), 1);
    expect_close(host_values<float>(keep), axis_ref, "sum-axis-keepdim");

    auto neg = ops::Sum(a, -1);
    EXPECT_EQ(neg.dims(), 1u) << "negative axis on " << device_to_string(dev);
    expect_close(host_values<float>(neg), axis_ref, "sum-axis-negative");
}

// 全归约的两趟路径。触发阈值是 64K 元素（Vulkan 256K），这里取 1M 与一个素数
// 尺寸：素数尺寸是唯一能测出末段截断的形状——2 的幂尺寸下 B*chunk == N、末段长度
// 为 0，截断写错了也发现不了。
//
// 容差一律用相对值：kEps 是绝对值，而 1M 个元素的和量级约 5e5，float32 输出的
// 最终舍入就有 ~0.03，绝对容差下必然失败。Prod 放到 5e-3——CUDA/SYCL 的浮点归约
// 累加器是 float32，1M 个元素连乘的偏差本就有一档，两趟比单趟略差；这个容差仍能
// 抓住结构性错误（范围算错、漏末段、分母错都会给出 O(1) 偏差）。
TEST_F(DeviceOps, FullReduceOnLargeTensors) {
    const std::string_view dev_name = device_to_string(dev);
    // 两组数据。A 组含负数与零，覆盖 Sum/Min/Max/Mean；B 组量级贴近 1，专供
    // Prod——A 组 1M 个量级 ~8 的数连乘必然溢出（部分积变 ±Inf，再乘 0 得 NaN），
    // 那种输入在任何累加顺序下都没有有意义的结果，不该用来测归约路径。
    auto make_a = [](int64_t n) {
        std::vector<float> v(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i)
            v[static_cast<size_t>(i)] = static_cast<float>(i % 97) * 0.25f - 8.0f;
        return v;
    };
    auto make_b = [](int64_t n) {
        std::vector<float> v(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i)
            v[static_cast<size_t>(i)] = 1.0f + static_cast<float>(i % 5 - 2) * 1e-6f;
        return v;
    };

    for (const int64_t n : {int64_t{1} << 20, int64_t{1'000'003}}) {
        std::vector<float> av = make_a(n);
        const Tensor a_cpu(av, {n});
        const Tensor a     = moved_to(a_cpu, dev);
        const float sum_ref  = ops::Sum(a_cpu).item<float>();
        const float mean_ref = ops::Mean(a_cpu).item<float>();
        const float min_ref  = ops::Min(a_cpu).item<float>();
        const float max_ref  = ops::Max(a_cpu).item<float>();

        EXPECT_NEAR(ops::Sum(a).item<float>(), sum_ref, 1e-5 * std::abs(sum_ref))
            << "sum full n=" << n << " on " << dev_name;
        EXPECT_NEAR(ops::Mean(a).item<float>(), mean_ref, 1e-5 * std::abs(mean_ref))
            << "mean full n=" << n << " on " << dev_name;
        // min/max 与累加顺序无关，必须逐位一致。
        EXPECT_EQ(ops::Min(a).item<float>(), min_ref) << "min full n=" << n << " on " << dev_name;
        EXPECT_EQ(ops::Max(a).item<float>(), max_ref) << "max full n=" << n << " on " << dev_name;

        std::vector<float> bv = make_b(n);
        const Tensor b_cpu(bv, {n});
        const Tensor b       = moved_to(b_cpu, dev);
        const float prod_ref = ops::Prod(b_cpu).item<float>();
        // 容差放到 5e-3：SYCL 的浮点归约累加器是 float32（reduction_acc_t
        // 只把窄整型与 16 位浮点加宽），1M 个元素连乘的 float32 偏差本就有一档，
        // 两趟（1024 次/段）比单趟（1M 次）略差。CUDA 与 VULKAN 已把浮点 Prod
        // 的累加器对齐到 CPU 参考的 double（见 reduce.cu 的 prod_uses_wide_acc
        // 与 shader/prod/prodvec_float32.comp），这条宽松容差现在只为 SYCL
        // 保留。它仍能抓住结构性错误——范围算错、漏掉末段、分母错都会给出
        // O(1) 的偏差。
        EXPECT_NEAR(ops::Prod(b).item<float>(), prod_ref, 5e-3 * std::abs(prod_ref))
            << "prod full n=" << n << " on " << dev_name;

        // 全归约仍是 0-dim、留在原设备：两趟路径不改变这些契约。
        const Tensor s = ops::Sum(a);
        EXPECT_EQ(s.dims(), 0u) << "sum full dims n=" << n << " on " << dev_name;
        EXPECT_EQ(s.device_id(), a.device_id());
    }

    // dtype 提升路径：int8 的全归约按 INT64 累加（趟 2 的输入 dtype 因此是
    // int64，是一条独立的 pipeline 组合）。
    {
        const int64_t n = int64_t{1} << 18;
        std::vector<int8_t> v(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i)
            v[static_cast<size_t>(i)] = static_cast<int8_t>((i % 5) - 2);
        const Tensor cpu(v, {n});
        const Tensor a = moved_to(cpu, dev);
        const int64_t sum_ref = ops::Sum(cpu).item<int64_t>();
        EXPECT_EQ(ops::Sum(a).item<int64_t>(), sum_ref)
            << "sum int8 full n=" << n << " on " << dev_name;
        EXPECT_EQ(ops::Sum(a).dtype(), DataType::INT64);
    }

    // float64：累加器本就是 double，两趟与单趟的相对偏差应在 1e-13 以内。
    {
        const int64_t n = int64_t{1} << 18;
        std::vector<double> v(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i)
            v[static_cast<size_t>(i)] = static_cast<double>(i % 251) * 1e-3;
        const Tensor cpu(v, {n});
        const Tensor a = moved_to(cpu, dev);
        const double sum_ref = ops::Sum(cpu).item<double>();
        EXPECT_NEAR(ops::Sum(a).item<double>(), sum_ref, 1e-9 * std::abs(sum_ref))
            << "sum f64 full n=" << n << " on " << dev_name;
    }

    // 阈值以下仍走单趟：小张量的结果必须与参考一致（截断恒等的另一面）。
    {
        std::vector<float> small{1, 2, 3, 4, 5};
        const Tensor cpu(small, std::vector<int64_t>{5});
        const Tensor a   = moved_to(cpu, dev);
        EXPECT_NEAR(ops::Sum(a).item<float>(), 15.0f, kEps) << "sum small on " << dev_name;
        EXPECT_NEAR(ops::Mean(a).item<float>(), 3.0f, kEps) << "mean small on " << dev_name;
    }
}

TEST_F(DeviceOps, ComparisonTypecastConcat) {
    std::vector<float> av{1, 2, 3, 4, 5, 6};
    std::vector<float> bv{6, 5, 4, 3, 2, 1};
    const Tensor a_cpu(av, {2, 3}), b_cpu(bv, {2, 3});

    const auto gt_ref   = host_values<std::int8_t>(ops::Greater(a_cpu, b_cpu));
    const auto gt0_ref  = host_values<std::int8_t>(ops::Greater(a_cpu, 3.0f));
    const auto cast_ref = host_values<double>(ops::Typecast(a_cpu, DataType::FLOAT64));
    const auto cat_ref  = host_values<float>(ops::Concat({a_cpu, b_cpu}, 0));

    const Tensor a = moved_to(a_cpu, dev);
    const Tensor b = moved_to(b_cpu, dev);
    expect_close(host_values<std::int8_t>(ops::Greater(a, b)), gt_ref, "greater");
    expect_close(host_values<std::int8_t>(ops::Greater(a, 3.0f)), gt0_ref, "greater-scalar");
    expect_close(host_values<double>(ops::Typecast(a, DataType::FLOAT64)), cast_ref, "typecast");
    expect_close(host_values<float>(ops::Concat({a, b}, 0)), cat_ref, "concat");
}

// 9 种 dtype 两两转换（含同型）。参考值由 CPU 后端给出，比较用原始字节：取值
// 都落在所有 dtype 都能精确表示的范围内，所以"逐位相同"是合理要求，也省掉了
// 为 f16/bf16 写浮点比较。BOOLe 源单独给 0/1 序列，免得整列都是 1。
TEST_F(DeviceOps, TypecastAllDtypePairs) {
    constexpr DataType kTypes[] = {
        DataType::BOOL,     DataType::INT8,     DataType::INT16,    DataType::INT32,
        DataType::INT64,    DataType::FLOAT16,  DataType::BFLOAT16, DataType::FLOAT32,
        DataType::FLOAT64};
    // -1.75 用来看浮点 -> 整型的截断方向（向零），-3 覆盖负数。
    const std::vector<double> numeric{-3.0, -1.75, 0.0, 0.5, 1.0, 2.0, 7.0, -1.0};
    const std::vector<double> flags{0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};

    auto expect_bits = [](Tensor got, const Tensor& want, std::string_view what) {
        ASSERT_EQ(got.dtype(), want.dtype()) << what;
        ASSERT_EQ(got.numel(), want.numel()) << what;
        got.to_host();
        const size_t bytes = got.numel() * calc_dtype_size(got.dtype());
        EXPECT_EQ(std::memcmp(got.data(), want.data(), bytes), 0)
            << what << " differs from the CPU reference";
    };

    for (DataType src_type : kTypes) {
        // 构造函数收的是非 const 左值引用，所以先落成一个具名对象。
        std::vector<double> base_values = src_type == DataType::BOOL ? flags : numeric;
        const Tensor base_cpu(base_values, std::vector<int64_t>{2, 4});
        const Tensor src_cpu = ops::Typecast(base_cpu, src_type);
        for (DataType dst_type : kTypes) {
            const std::string what =
                std::format("cast {}->{} on {}", dtype_to_string(src_type),
                            dtype_to_string(dst_type), device_to_string(dev));
            const Tensor want = ops::Typecast(src_cpu, dst_type);
            Tensor got        = ops::Typecast(moved_to(src_cpu, dev), dst_type);
            EXPECT_TRUE(got.device() == dev) << what;
            expect_bits(got, want, what);

            // 原地入口（to_type_）与返回新张量的入口共用内核，但换 buffer 的
            // 路径不同：旧存储在提交在飞时就要退役。
            Tensor in_place = moved_to(src_cpu, dev);
            in_place.to_type_(dst_type);
            expect_bits(in_place, want, what + " (in-place)");
        }
    }
}

TEST_F(DeviceOps, SliceAndTranspose) {
    std::vector<float> av{1, 2, 3, 4, 5, 6};
    const Tensor a_cpu(av, {2, 3});

    const auto slice_ref  = host_values<float>(ops::Slice(a_cpu, {{0, 1}, {1, 3}}));
    const auto transp_ref = host_values<float>(ops::Transpose(a_cpu, {1, 0}));

    const Tensor a = moved_to(a_cpu, dev);
    expect_close(host_values<float>(ops::Slice(a, {{0, 1}, {1, 3}})), slice_ref, "slice");
    expect_close(host_values<float>(ops::Transpose(a, {1, 0})), transp_ref, "transpose");

    // 三维六种置换逐一对照。非方阵（3x5x7）保证步长推导的错误不会被对称性
    // 掩盖；其中 {0,1,2} 与 {1,0,2} 走 rank-3 的拷贝路径，其余走 tile 转置。
    std::vector<float> v3(3 * 5 * 7);
    for (std::size_t i = 0; i < v3.size(); ++i) v3[i] = static_cast<float>(i + 1);
    const Tensor t3_cpu(v3, {3, 5, 7});
    const Tensor t3 = moved_to(t3_cpu, dev);
    constexpr int64_t kPerms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                      {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    for (const auto& perm : kPerms) {
        const std::string what = "transpose3d " + std::to_string(perm[0]) +
                                 std::to_string(perm[1]) + std::to_string(perm[2]);
        expect_close(host_values<float>(ops::Transpose(t3, {perm[0], perm[1], perm[2]})),
                     host_values<float>(ops::Transpose(t3_cpu, {perm[0], perm[1], perm[2]})),
                     what);
    }

    // 四维走通用内核，确认 rank-3 快路径没有误接管。
    std::vector<float> v4(2 * 3 * 4 * 5);
    for (std::size_t i = 0; i < v4.size(); ++i) v4[i] = static_cast<float>(i) * 0.5f;
    const Tensor t4_cpu(v4, {2, 3, 4, 5});
    const Tensor t4 = moved_to(t4_cpu, dev);
    expect_close(host_values<float>(ops::Transpose(t4, {2, 0, 3, 1})),
                 host_values<float>(ops::Transpose(t4_cpu, {2, 0, 3, 1})), "transpose4d");
}

// 分配器的复用路径：多个线程各自反复建小张量、写值、读回。块常常由一个线程
// 释放、被另一个线程取走，所以只要复用时的同步或记账有偏差，读回的值就会错。
TEST_F(DeviceOps, ConcurrentSmallTensorChurn) {
    constexpr int kThreads = 8;
    constexpr int kRounds  = 64;
    std::atomic<int> failures{0};

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t] {
            std::vector<float> pattern(64, static_cast<float>(t) + 1.0f);
            try {
                for (int i = 0; i < kRounds; ++i) {
                    Tensor a(pattern, {8, 8});
                    a.to_device(dev);
                    const auto back = host_values<float>(a);
                    if (!std::equal(back.begin(), back.end(), pattern.begin())) ++failures;
                }
            } catch (const std::exception& e) {
                std::println("churn thread {} threw: {}", t, e.what());
                ++failures;
            }
        });
    }
    for (auto& w : workers) w.join();
    EXPECT_EQ(failures.load(), 0);
}


// ---------- 0c. ops::synchronize / empty_cache 覆盖所有已起后端 ----------
TEST(DeviceControl, EveryLiveBackendAnswersBothCalls) {
    // CPU 上两者都是合法的 no-op，泛型代码不必先判断设备。
    EXPECT_NO_THROW(ops::synchronize());
    EXPECT_NO_THROW(ops::empty_cache());

    // 先把各后端拉起来，再确认没人漏注册。
    for (const Device dev : {Device::CUDA, Device::VULKAN, Device::SYCL}) {
        if (!backend_available(dev)) continue;
        EXPECT_NO_THROW(Tensor::Ones({2, 2}, DataType::FLOAT32, dev)) << device_to_string(dev);
    }
    EXPECT_NO_THROW(ops::synchronize());
    EXPECT_NO_THROW(ops::empty_cache());
}


// ---------- 0d. 多设备寻址：current device / DeviceGuard ----------
// 单卡机器上这些用例验证的是"序号空间的边界与状态机"，不是第二张卡的实际行为。
TEST(DeviceId, CpuIsAlwaysIndexZero) {
    EXPECT_EQ(ops::device_count(Device::CPU), 1);
    EXPECT_EQ(ops::current_device(Device::CPU), 0);
    EXPECT_THROW(ops::set_device(Device::CPU, 1), std::exception);
    // 每张 CPU 张量都带序号 0，视图与克隆也不例外。
    std::vector<float> v{1.0f, 2.0f};
    Tensor t(v, {2});
    EXPECT_EQ(t.device_index(), 0);
    EXPECT_EQ(t.device_id(), (DeviceId{Device::CPU, 0}));
}

TEST(DeviceId, SetDeviceRejectsOutOfRange) {
    if (!backend_available(kTestDevice)) GTEST_SKIP() << "test backend unavailable";
    const int count = ops::device_count(kTestDevice);
    ASSERT_GE(count, 1);
    const int before = ops::current_device(kTestDevice);
    EXPECT_THROW(ops::set_device(kTestDevice, count), std::exception);
    EXPECT_THROW(ops::set_device(kTestDevice, count + 7), std::exception);
    // 失败不改变当前设备。
    EXPECT_EQ(ops::current_device(kTestDevice), before);
}

TEST(DeviceId, SetDeviceRejectsNegative) {
    if (!backend_available(kTestDevice)) GTEST_SKIP() << "test backend unavailable";
    EXPECT_THROW(ops::set_device(kTestDevice, -1), std::exception);
    EXPECT_THROW(ops::set_device(Device::CPU, -1), std::exception);
}

TEST(DeviceId, GuardRestoresPrevious) {
    if (!backend_available(kTestDevice)) GTEST_SKIP() << "test backend unavailable";
    const int before = ops::current_device(kTestDevice);
    {
        ops::DeviceGuard guard(kTestDevice, before);
        EXPECT_EQ(ops::current_device(kTestDevice), before);
        {
            ops::DeviceGuard nested(DeviceId{kTestDevice, before});
            EXPECT_EQ(ops::current_device(kTestDevice), before);
        }
        EXPECT_EQ(ops::current_device(kTestDevice), before);
    }
    EXPECT_EQ(ops::current_device(kTestDevice), before);
}

TEST(DeviceId, CurrentDeviceIsThreadLocal) {
    if (!backend_available(kTestDevice)) GTEST_SKIP() << "test backend unavailable";
    const int main_view = ops::current_device(kTestDevice);
    int child_view = -1;
    std::thread child([&] { child_view = ops::current_device(kTestDevice); });
    child.join();
    // 子线程没见过主线程的设置，拿到的是后端自己的默认值。单卡上两者数值
    // 相同，真正有区分力的是多卡机上的 [multi] 用例。
    EXPECT_EQ(child_view, ops::current_device(kTestDevice));
    EXPECT_EQ(main_view, ops::current_device(kTestDevice));
}

TEST(DeviceId, OpOutputsFollowInput) {
    if (!backend_available(kTestDevice)) GTEST_SKIP() << "test backend unavailable";
    const Tensor a = Tensor::Ones({2, 3}, DataType::FLOAT32, kTestDevice);
    const Tensor b = Tensor::Ones({2, 3}, DataType::FLOAT32, kTestDevice);
    const DeviceId want = a.device_id();

    EXPECT_EQ((a + b).device_id(), want);
    EXPECT_EQ((a - b).device_id(), want);
    EXPECT_EQ((a * b).device_id(), want);
    EXPECT_EQ((a / b).device_id(), want);
    EXPECT_EQ(ops::Relu(a).device_id(), want);
    EXPECT_EQ(ops::Sum(a, 1).device_id(), want);
    EXPECT_EQ(ops::Typecast(a, DataType::FLOAT64).device_id(), want);
    EXPECT_EQ(ops::Concat({a, b}, 0).device_id(), want);
}

TEST(DeviceId, ViewsCloneTypecastPreserveIndex) {
    if (!backend_available(kTestDevice)) GTEST_SKIP() << "test backend unavailable";
    Tensor a = Tensor::Ones({2, 3}, DataType::FLOAT32, kTestDevice);
    const int want_index = a.device_index();

    // 视图共享 impl，但元数据里的序号必须跟着走：丢掉它，后续 clone() 会分到
    // 另一张卡上。
    EXPECT_EQ(a.slice({{0, 1}, {0, 2}}).device_index(), want_index);
    EXPECT_EQ(a.t().device_index(), want_index);
    EXPECT_EQ(a.permute({1, 0}).device_index(), want_index);
    EXPECT_EQ(a.t().contiguous().device_index(), want_index);
    EXPECT_EQ(a.clone().device_index(), want_index);
    EXPECT_EQ(a.to_type(DataType::FLOAT64).device_index(), want_index);

    // 回到 host 就是 CPU:0，不再是加速器的序号。
    Tensor host = a.clone();
    host.to_host();
    EXPECT_EQ(host.device_id(), (DeviceId{Device::CPU, 0}));
}

TEST(DeviceId, SynchronizeCoversAllIndices) {
    for (const Device dev : kAllDevices) {
        const int count = ops::device_count(dev);
        for (int index = 0; index < count; ++index) {
            ops::DeviceGuard guard(dev, index);
            EXPECT_NO_THROW(Tensor::Ones({2, 2}, DataType::FLOAT32, dev))
                << device_to_string(dev) << ":" << index;
        }
    }
    EXPECT_NO_THROW(ops::synchronize());
    EXPECT_NO_THROW(ops::empty_cache());
}


// ---------- 0e. 双卡用例：单卡机器上整组 SKIP ----------
// 序号空间在第二张卡上的行为只有多卡机才验证得了；单卡上整组跳过，既不进失败
// 统计，也不会把"这台机器只有一张卡"误报成回归。
class MultiDevice : public ::testing::Test {
protected:
    void SetUp() override {
        if (!backend_available(kTestDevice)) {
            GTEST_SKIP() << device_to_string(kTestDevice) << " is unavailable in this build";
        }
        const int count = ops::device_count(kTestDevice);
        if (count < 2) {
            GTEST_SKIP() << "needs two " << device_to_string(kTestDevice)
                         << " devices, this machine has " << count;
        }
    }
    static constexpr DeviceId kSecond{kTestDevice, 1};
};

TEST_F(MultiDevice, CrossOperandRejected) {
    Tensor a = Tensor::Ones({4}, DataType::FLOAT32, kTestDevice);
    Tensor b = Tensor::Ones({4}, DataType::FLOAT32, kTestDevice);
    b.to_device(kSecond);
    ASSERT_EQ(b.device_id(), kSecond);
    // 同一后端的两张卡也是两个设备：拿 a 的 context 读 b 的指针是静默的数据错误。
    EXPECT_THROW(ops::Add(a, b), std::exception);
    EXPECT_THROW(ops::Concat({a, b}, 0), std::exception);
}

TEST_F(MultiDevice, ToDeviceCopiesValues) {
    std::vector<float> values{1.0f, 2.0f, 3.0f};
    Tensor t(values, {3});
    t.to_device(kSecond);
    ASSERT_EQ(t.device_id(), kSecond);
    EXPECT_EQ(host_values<float>(t), values);
    // 再回到 host 就是 {CPU, 0}，序号不残留。
    t.to_host();
    EXPECT_EQ(t.device_id(), (DeviceId{Device::CPU, 0}));
    EXPECT_EQ(host_values<float>(t), values);
}

TEST_F(MultiDevice, DevicePlusIndexMatchesDeviceId) {
    // 两个入口解析出同一个目标：`(Device, index)` 只是给 DeviceId 的语法糖。
    std::vector<float> values{4.0f, 5.0f};
    Tensor by_id(values, {2});
    Tensor by_pair(values, {2});
    by_id.to_device(kSecond);
    by_pair.to_device(kTestDevice, 1);
    EXPECT_EQ(by_id.device_id(), by_pair.device_id());
    EXPECT_EQ(host_values<float>(by_id), values);
    EXPECT_EQ(host_values<float>(by_pair), values);
}

TEST_F(MultiDevice, CurrentDeviceIsPerThreadPerIndex) {
    const int before = ops::current_device(kTestDevice);
    ops::set_device(kTestDevice, 0);
    int child_index = -1;
    std::thread child([&] {
        // 子线程没见过主线程的设置，先自己切到 1 号卡再建张量。
        ops::set_device(kTestDevice, 1);
        child_index = Tensor::Ones({2}, DataType::FLOAT32, kTestDevice).device_index();
    });
    child.join();
    // 子线程的切换不外溢，主线程还在 0 号卡。
    EXPECT_EQ(child_index, 1);
    EXPECT_EQ(ops::current_device(kTestDevice), 0);
    ops::set_device(kTestDevice, before);
}


// ---------- 0f. CUDA 流：current stream 按线程取值（P2） ----------
#if defined(BACKEND_CUDA)
namespace {

std::shared_ptr<CUDAContext> cuda_context_of(const Tensor& t) {
    auto impl = std::dynamic_pointer_cast<CUDATensor>(t.get_impl());
    return impl == nullptr ? nullptr
                           : std::dynamic_pointer_cast<CUDAContext>(impl->context());
}

}  // namespace

TEST(CudaStream, OneLazyStreamPerThreadAndOpsStayCorrect) {
    if (!backend_available(Device::CUDA)) GTEST_SKIP() << "CUDA is unavailable in this build";

    Tensor probe = Tensor::Ones({4, 4}, DataType::FLOAT32, Device::CUDA);
    auto   ctx   = cuda_context_of(probe);
    ASSERT_NE(ctx, nullptr);

    // 首次调用时创建，之后同一线程每次拿到的都是同一条。
    const auto mine = ctx->current_stream();
    ASSERT_NE(mine.handle, nullptr);
    EXPECT_EQ(mine.device, ctx->device_index());
    EXPECT_EQ(ctx->current_stream().handle, mine.handle);

    // 算子入队到这条流；读取结果（宿主同步点）后必须正确。
    Tensor a = Tensor::Ones({2, 3}, DataType::FLOAT32, Device::CUDA);
    a = a + a;
    for (float v : host_values<float>(a)) EXPECT_FLOAT_EQ(v, 2.0f);

    // 流是线程私有的：另一个线程惰性创建自己的那一条。
    void* other_stream = nullptr;
    std::thread other([&] { other_stream = ctx->current_stream().handle; });
    other.join();
    EXPECT_NE(other_stream, nullptr);
    EXPECT_NE(other_stream, mine.handle);

    ctx->synchronize_all();
}
#endif  // BACKEND_CUDA


// ---------- 0f'. Vulkan 提交通道：每线程一条（P2） ----------
#if defined(BACKEND_VULKAN)
namespace {

std::shared_ptr<VulkanContext> vulkan_context_of(const Tensor& t) {
    auto impl = std::dynamic_pointer_cast<VKTensor>(t.get_impl());
    return impl == nullptr ? nullptr
                           : std::dynamic_pointer_cast<VulkanContext>(impl->context());
}

}  // namespace

TEST(VulkanQueue, OneChannelPerThreadAndOpsStayCorrect) {
    if (!backend_available(Device::VULKAN)) GTEST_SKIP() << "Vulkan is unavailable in this build";

    Tensor probe = Tensor::Ones({4, 4}, DataType::FLOAT32, Device::VULKAN);
    auto   ctx   = vulkan_context_of(probe);
    ASSERT_NE(ctx, nullptr);
    ASSERT_GE(ctx->queue_channel_count(), 1u);
    std::println("[tensorvia] Vulkan queue channels: {}", ctx->queue_channel_count());

    // 同一线程每次拿到的是同一条通道。
    VulkanContext::QueueChannel* mine = &ctx->currentQueue();
    EXPECT_EQ(&ctx->currentQueue(), mine);

    // 算子走这条通道，入队后回到宿主的同步点仍必须正确。
    Tensor a = probe + probe;
    for (float v : host_values<float>(a)) EXPECT_FLOAT_EQ(v, 2.0f);

    // 另一条线程拿到的通道可能相同（线程数超过通道数时按设计共享），但它的算子
    // 与宿主读取必须自成一体。
    void* other_channel = nullptr;
    float other_value   = 0.0f;
    std::thread thread([&] {
        other_channel = &ctx->currentQueue();
        Tensor local  = probe * 3.0f;
        other_value   = host_values<float>(local).front();
    });
    thread.join();
    ASSERT_NE(other_channel, nullptr);
    EXPECT_FLOAT_EQ(other_value, 3.0f);

    // 设备级同步覆盖所有通道。
    EXPECT_NO_THROW(ops::synchronize());
}
#endif  // BACKEND_VULKAN


// ---------- 0f''. Vulkan 可选能力：cooperative matrix 探测 ----------
#if defined(BACKEND_VULKAN)
TEST(VulkanCapabilities, CooperativeMatrixReflectsDeviceState) {
    if (!backend_available(Device::VULKAN)) GTEST_SKIP() << "Vulkan is unavailable in this build";

    Tensor probe = Tensor::Ones({4, 4}, DataType::FLOAT32, Device::VULKAN);
    auto   ctx   = vulkan_context_of(probe);
    ASSERT_NE(ctx, nullptr);

    const auto device        = ctx->getPhysicalDevice();
    bool       has_extension = false;
    for (const auto& ext : device.enumerateDeviceExtensionProperties())
        if (std::strcmp(ext.extensionName, "VK_KHR_cooperative_matrix") == 0) has_extension = true;

    // 报告值只能比"设备有扩展"更严格：没有扩展却报支持，将来的内核会在创建
    // 管线时失败；有扩展却漏报，则白白丢掉一条可选路径。
    if (!has_extension) {
        EXPECT_FALSE(ctx->supports_cooperative_matrix());
        EXPECT_TRUE(ctx->cooperative_matrix_configs().empty());
        return;
    }

    vk::PhysicalDeviceCooperativeMatrixFeaturesKHR features{};
    vk::PhysicalDeviceFeatures2                  features2{};
    features2.pNext = &features;
    device.getFeatures2(&features2);
    EXPECT_EQ(ctx->supports_cooperative_matrix(), static_cast<bool>(features.cooperativeMatrix));

    if (!ctx->supports_cooperative_matrix()) return;
    const auto& configs = ctx->cooperative_matrix_configs();
    ASSERT_FALSE(configs.empty());
    for (const auto& c : configs) {
        EXPECT_GT(c.MSize, 0u) << "a reported configuration must have a positive M";
        EXPECT_GT(c.NSize, 0u) << "a reported configuration must have a positive N";
        EXPECT_GT(c.KSize, 0u) << "a reported configuration must have a positive K";
        std::println("[tensorvia] coopmat M={} N={} K={} A={} B={} C={} R={}", c.MSize, c.NSize,
                     c.KSize, static_cast<uint32_t>(c.AType), static_cast<uint32_t>(c.BType),
                     static_cast<uint32_t>(c.CType), static_cast<uint32_t>(c.ResultType));
    }

    // MatMul 的半精度内核要求的正是 subgroup 32 + 16x16x16 + f32 累加。这些条件
    // 都在时，管线应当真的换成了 coopmat 版本——回落是建管线失败时的安全网，
    // 不是这里的预期结果。
    vk::PhysicalDeviceSubgroupProperties subgroup{};
    vk::PhysicalDeviceProperties2        props2{};
    props2.pNext = &subgroup;
    device.getProperties2(&props2);
    if (subgroup.subgroupSize != 32) return;

    bool f16 = false, bf16 = false;
    for (const auto& c : configs) {
        if (c.scope != VK_SCOPE_SUBGROUP_KHR || c.MSize != 16 || c.NSize != 16 || c.KSize != 16)
            continue;
        if (c.CType != VK_COMPONENT_TYPE_FLOAT32_KHR ||
            c.ResultType != VK_COMPONENT_TYPE_FLOAT32_KHR)
            continue;
        f16 |= c.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && c.BType == VK_COMPONENT_TYPE_FLOAT16_KHR;
        bf16 |=
            c.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR && c.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR;
    }
    if (f16) EXPECT_TRUE(ctx->matmul_uses_cooperative_matrix(DataType::FLOAT16));
    if (bf16) EXPECT_TRUE(ctx->matmul_uses_cooperative_matrix(DataType::BFLOAT16));
    std::println("[tensorvia] matmul coopmat in use: f16={} bf16={}",
                 ctx->matmul_uses_cooperative_matrix(DataType::FLOAT16),
                 ctx->matmul_uses_cooperative_matrix(DataType::BFLOAT16));
}
#endif  // BACKEND_VULKAN


// ---------- 0g. empty_cache 不碰活着的张量（CUDA） ----------
#if defined(BACKEND_CUDA)
TEST(DeviceControl, CudaEmptyCacheKeepsLiveTensors) {
    if (!backend_available(Device::CUDA)) GTEST_SKIP() << "CUDA is unavailable in this build";

    Tensor probe = Tensor::Ones({4, 4}, DataType::FLOAT32, Device::CUDA);
    auto   ctx   = cuda_context_of(probe);
    ASSERT_NE(ctx, nullptr);

    {
        // 算子只入队；读取结果时才回到宿主。
        Tensor a = probe + probe;
        for (float v : host_values<float>(a)) EXPECT_FLOAT_EQ(v, 2.0f);
    }
    EXPECT_NO_THROW(ops::synchronize());

    // 释放之后积压的块进缓存，empty_cache 把它们还给驱动；仍然活着的张量
    // （probe）不受影响。
    probe = Tensor();
    EXPECT_NO_THROW(ops::empty_cache());

    // 交还之后还能正常再要内存。
    Tensor again = Tensor::Zeros({2, 2}, DataType::FLOAT32, Device::CUDA);
    for (float v : host_values<float>(again)) EXPECT_FLOAT_EQ(v, 0.0f);
    EXPECT_NO_THROW(ops::synchronize());
}
#endif  // BACKEND_CUDA


// ---------- 0h. 每张卡一个 context（CUDA） ----------
#if defined(BACKEND_CUDA)
// 单卡机器上也能验证"每个序号一个独立 context"：直接构造同序号的两个对象。
TEST(CudaIndex, TwoContextsOfSameIndexAreDistinct) {
    if (!backend_available(Device::CUDA)) GTEST_SKIP() << "CUDA is unavailable in this build";
    auto first  = std::make_shared<CUDAContext>(0);
    auto second = std::make_shared<CUDAContext>(0);
    EXPECT_NE(first.get(), second.get());
    EXPECT_EQ(first->device_index(), 0);
    EXPECT_EQ(second->device_index(), 0);
    // 每（线程, context）一条流：两个 context 上的流互不相同。
    EXPECT_NE(first->current_stream().handle, second->current_stream().handle);

    auto a = std::make_shared<CUDATensor>(4, DataType::FLOAT32, first);
    auto b = std::make_shared<CUDATensor>(4, DataType::FLOAT32, second);
    // copy_to 只在同一块存储内成立；跨 context 必须被拒，而不是拷一串错地址。
    EXPECT_THROW(a->copy_to(b), std::invalid_argument);
}

TEST(CudaIndex, FactoryContextMatchesIndex) {
    if (!backend_available(Device::CUDA)) GTEST_SKIP() << "CUDA is unavailable in this build";
    const int index = ops::current_device(Device::CUDA);
    Tensor t = Tensor::Ones({4}, DataType::FLOAT32, Device::CUDA);
    ASSERT_EQ(t.device_index(), index);
    auto ctx = cuda_context_of(t);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->device_index(), t.device_index());
}
#endif  // BACKEND_CUDA

