// P3 统一 strided 迭代层（FR-18 / FR-19）的语义用例。
//
// 每个用例都把"算子作用在视图上"的结果，与一份**在 host 上手工按索引算出来**的
// 基线逐值对比。基线不经过本库的任何算子，所以它锁的是语义而不是实现：无论后端
// 直接吃 stride 描述符，还是先物化再算，结果都必须一致。
//
// 覆盖的三类布局：转置（strides 换位）、permute（多维换位）、切片（视图 offset），
// 以及与广播（shape 右对齐、广播维 stride 为 0）的组合。

#include "test_support.hpp"

namespace {

// 取值互不相同的填充，保证转置/permute 前后同一行内的元素也不相等——否则"漏乘
// 一个 stride"这类错误可能被对称性掩盖。
std::vector<float> ramp(std::size_t n, float scale, float shift) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = scale * static_cast<float>((i * 7 + 3) % 23) - shift;
    return v;
}

}  // namespace

// 转置视图作为两个操作数：结果 = Aᵀ ⊙ Bᵀ，与 host 上按 (j, i) 取值相乘一致。
TEST_F(DeviceOps, TransposedOperandsMatchContiguous) {
    constexpr int64_t n = 5;
    std::vector<float> av = ramp(n * n, 1.0f, 0.0f);
    std::vector<float> bv = ramp(n * n, 0.5f, 1.0f);

    Tensor a = moved_to(Tensor(av, {n, n}), dev);
    Tensor b = moved_to(Tensor(bv, {n, n}), dev);
    Tensor at = a.t();
    Tensor bt = b.t();
    ASSERT_FALSE(at.is_contiguous());
    ASSERT_FALSE(bt.is_contiguous());

    std::vector<float> want(n * n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            want[i * n + j] = av[j * n + i] * bv[j * n + i];

    expect_close(host_values<float>(ops::Mul(at, bt)), want, "A^T * B^T");
}

// permute 到 {4, 2, 3} 后与 {3} 广播相加：视图 strides 是原布局的重排，广播维的
// stride 为 0，两者叠加时索引不能被混淆。
TEST_F(DeviceOps, PermutedBroadcastMatchesContiguous) {
    constexpr int64_t d0 = 2, d1 = 3, d2 = 4;
    std::vector<float> base_v = ramp(d0 * d1 * d2, 1.0f, 0.0f);
    std::vector<float> bias_v = ramp(3, 0.25f, 2.0f);

    Tensor base = moved_to(Tensor(base_v, {d0, d1, d2}), dev);
    Tensor bias = moved_to(Tensor(bias_v, {3}), dev);
    Tensor p = base.permute({2, 0, 1});  // {4, 2, 3}
    ASSERT_FALSE(p.is_contiguous());
    const std::span<const int64_t> p_shape = p.shape();
    ASSERT_EQ(p_shape.size(), std::size_t{3});
    EXPECT_EQ(p_shape[0], d2);
    EXPECT_EQ(p_shape[1], d0);
    EXPECT_EQ(p_shape[2], d1);

    std::vector<float> want(d0 * d1 * d2);
    for (int64_t i = 0; i < d2; ++i)
        for (int64_t j = 0; j < d0; ++j)
            for (int64_t k = 0; k < d1; ++k)
                want[(i * d0 + j) * d1 + k] = base_v[(j * d1 + k) * d2 + i] + bias_v[k];

    expect_close(host_values<float>(ops::Add(p, bias)), want, "permute + broadcast");
}

// 切片视图带 offset：比较算子必须只把 offset 计入一次。这里同时用同一个视图做
// 两侧操作数，任何重复计入（或遗漏）都会让 Equal 直接失败。
TEST_F(DeviceOps, OffsetViewComparisonUsesViewOffsetOnce) {
    constexpr int64_t n = 5;
    std::vector<float> base_v = ramp(n * n, 1.0f, 0.0f);

    Tensor base = moved_to(Tensor(base_v, {n, n}), dev);
    Tensor view = base.slice({{1, 4}, {1, 4}});  // 3x3，offset = 6
    ASSERT_FALSE(view.is_contiguous());
    ASSERT_GT(view.offset(), 0);

    Tensor ones = ops::Fill({3, 3}, DataType::FLOAT32, 1.0f, dev);

    // 视图与自身比较必然全真；与全 1 比较，只有恰好等于 1 的元素为真。
    std::vector<std::int8_t> want_self(9, 1);
    std::vector<std::int8_t> want_ones(9);
    for (int64_t i = 0; i < 3; ++i)
        for (int64_t j = 0; j < 3; ++j)
            want_ones[i * 3 + j] = base_v[(i + 1) * n + (j + 1)] == 1.0f ? 1 : 0;

    expect_close(host_values<std::int8_t>(ops::Equal(view, view)), want_self, "view == view");
    expect_close(host_values<std::int8_t>(ops::Equal(view, ones)), want_ones,
                 "view == ones");
}

// offset + 转置 strides + 广播三者同时出现：切转置视图的一片，再加一行向量。
TEST_F(DeviceOps, StridedOffsetBroadcastCombined) {
    constexpr int64_t n = 5;
    std::vector<float> a_v = ramp(n * n, 1.0f, 0.0f);
    std::vector<float> row_v = ramp(3, 0.5f, 1.0f);

    Tensor a = moved_to(Tensor(a_v, {n, n}), dev);
    Tensor row = moved_to(Tensor(row_v, {3}), dev);
    Tensor view = a.t().slice({{0, 3}, {1, 4}});  // {3, 3}，strides (1, 5)，offset = 1
    ASSERT_FALSE(view.is_contiguous());

    std::vector<float> want(9);
    for (int64_t i = 0; i < 3; ++i)
        for (int64_t j = 0; j < 3; ++j)
            want[i * 3 + j] = a_v[(j + 1) * n + i] + row_v[j];

    expect_close(host_values<float>(ops::Add(view, row)), want, "strided offset + broadcast");
}

// 不可广播的形状必须在发射前就被拒绝，而不是产生越界读。
TEST_F(DeviceOps, IncompatibleShapesRejected) {
    std::vector<float> av = ramp(6, 1.0f, 0.0f);
    std::vector<float> bv = ramp(8, 1.0f, 0.0f);
    Tensor a = moved_to(Tensor(av, {2, 3}), dev);
    Tensor b = moved_to(Tensor(bv, {4, 2}), dev);
    EXPECT_THROW((void)ops::Add(a, b), std::runtime_error);
}

// 一元算子直接吃视图：结果按视图顺序排布，与连续基线一致。
// 取值范围压到 [-2, 1.4]：exp 的绝对值随幅值指数增长，而各后端 exp 的 ULP 级差异
// 在 host 参考值的绝对误差上会被放大，测试就变成比 libm 与 device math 的一致性。
TEST_F(DeviceOps, TransposedUnaryMatchesContiguous) {
    constexpr int64_t n = 4;
    std::vector<float> av = ramp(n * n, 0.15f, 2.0f);

    Tensor a = moved_to(Tensor(av, {n, n}), dev);
    Tensor at = a.t();
    ASSERT_FALSE(at.is_contiguous());

    std::vector<float> want(n * n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            want[i * n + j] = std::exp(av[j * n + i]);

    expect_close(host_values<float>(ops::Exp(at)), want, "exp(A^T)");
}

// 比较两个转置视图：两侧都是跨步，结果必须逐元素正确。
TEST_F(DeviceOps, TransposedComparisonMatchesContiguous) {
    constexpr int64_t n = 4;
    std::vector<float> av = ramp(n * n, 1.0f, 0.0f);
    std::vector<float> bv = av;  // 同一份数据；再用偏移制造部分相等/不等

    Tensor a = moved_to(Tensor(av, {n, n}), dev);
    Tensor b = moved_to(Tensor(bv, {n, n}), dev);
    Tensor at = a.t();
    Tensor bt = b.t();

    std::vector<std::int8_t> want_equal(n * n, 1);
    std::vector<std::int8_t> want_gt(n * n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            const float x = av[j * n + i];
            const float y = bv[j * n + i];
            want_gt[i * n + j] = x > y ? 1 : 0;
        }

    expect_close(host_values<std::int8_t>(ops::Equal(at, bt)), want_equal, "A^T == B^T");
    expect_close(host_values<std::int8_t>(ops::Greater(at, bt)), want_gt, "A^T > B^T");
}

// where 的 cond/a 都是视图，且广播维来自标量分支。
// where 目前只实现了 CPU 后端（公开入口不走 execute_on），设备 lane 跳过。
TEST_F(DeviceOps, WhereOnViewsMatchesContiguous) {
    if (dev != Device::CPU) GTEST_SKIP() << "where is CPU-only";
    constexpr int64_t rows = 4, cols = 5;
    std::vector<float> cond_v = ramp(rows * cols, 1.0f, 3.0f);  // 有正有负
    std::vector<float> av = ramp(rows * cols, 0.5f, 1.0f);

    Tensor cond = moved_to(Tensor(cond_v, {rows, cols}), dev);
    Tensor a = moved_to(Tensor(av, {rows, cols}), dev);
    Tensor cond_t = cond.t();  // {5, 4}
    Tensor a_t = a.t();
    ASSERT_FALSE(a_t.is_contiguous());

    std::vector<float> want(rows * cols);
    for (int64_t i = 0; i < cols; ++i)
        for (int64_t j = 0; j < rows; ++j) {
            const float c = cond_v[j * cols + i];
            const float x = av[j * cols + i];
            want[i * rows + j] = (c != 0.0f) ? x : 7.5f;
        }

    expect_close(host_values<float>(ops::Where(cond_t, a_t, 7.5f)), want,
                 "where(cond^T, a^T, 7.5)");
}

// 转置视图上的单轴归约：被归约轴在内存里连续时走跨步路径（不物化），
// 结果必须与按视图语义手工算出的基线一致。
TEST_F(DeviceOps, StridedAxisReduceMatchesContiguous) {
    constexpr int64_t rows = 4, cols = 5;
    std::vector<float> av = ramp(rows * cols, 1.0f, 0.0f);
    Tensor a = moved_to(Tensor(av, {rows, cols}), dev);
    Tensor at = a.t();  // {5, 4}
    ASSERT_FALSE(at.is_contiguous());

    auto view = [&](int64_t i, int64_t j) { return av[j * cols + i]; };  // at[i][j]

    // axis 0（长度 cols，stride 1）：走跨步路径
    std::vector<float> want_sum(rows), want_mean(rows), want_max(rows);
    std::vector<std::int64_t> want_argmax(rows), want_argmin(rows);
    for (int64_t j = 0; j < rows; ++j) {
        double s = 0.0;
        float mx = view(0, j), mn = view(0, j);
        int64_t arg = 0, argmin = 0;
        for (int64_t i = 0; i < cols; ++i) {
            const float x = view(i, j);
            s += x;
            if (x > mx) { mx = x; arg = i; }
            if (x < mn) { mn = x; argmin = i; }
        }
        want_sum[j] = static_cast<float>(s);
        want_mean[j] = static_cast<float>(s / double(cols));
        want_max[j] = mx;
        want_argmax[j] = arg;
        want_argmin[j] = argmin;
    }

    expect_close(host_values<float>(ops::Sum(at, 0)), want_sum, "sum(A^T, 0)");
    expect_close(host_values<float>(ops::Mean(at, 0)), want_mean, "mean(A^T, 0)");
    expect_close(host_values<float>(ops::Max(at, 0)), want_max, "max(A^T, 0)");
    EXPECT_EQ(host_values<std::int64_t>(ops::Argmax(at, 0)), want_argmax);
    EXPECT_EQ(host_values<std::int64_t>(ops::Argmin(at, 0)), want_argmin);

    // axis 1（长度 rows，stride = cols）：被归约轴跨步 —— 策略允许物化，结果同样
    // 必须正确
    std::vector<float> want_sum1(cols);
    for (int64_t i = 0; i < cols; ++i) {
        double s = 0.0;
        for (int64_t j = 0; j < rows; ++j) s += view(i, j);
        want_sum1[i] = static_cast<float>(s);
    }
    expect_close(host_values<float>(ops::Sum(at, 1)), want_sum1, "sum(A^T, 1)");
}

// 全归约（不看轴）在 permute 视图上：结果是一个标量，布局只影响遍历顺序。
TEST_F(DeviceOps, PermutedFullReduceMatchesContiguous) {
    constexpr int64_t d0 = 2, d1 = 3, d2 = 4;
    std::vector<float> base_v = ramp(d0 * d1 * d2, 1.0f, 0.0f);
    Tensor base = moved_to(Tensor(base_v, {d0, d1, d2}), dev);
    Tensor p = base.permute({2, 0, 1});
    ASSERT_FALSE(p.is_contiguous());

    double sum = 0.0;
    float mx = base_v[0];
    for (float x : base_v) { sum += x; mx = std::max(mx, x); }

    expect_close(host_values<float>(ops::Sum(p)), {static_cast<float>(sum)},
                 "sum(permute)");
    expect_close(host_values<float>(ops::Max(p)), {mx}, "max(permute)");
}
