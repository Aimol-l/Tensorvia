#include "test_support.hpp"

// =====================================================================
// P0 语义契约（specs/2026-10-30-inference-tensor-contract.md FR-3..FR-7）
// =====================================================================

// ---------- FR-3：多轴归约 ----------
TEST_F(DeviceOps, MultiAxisReduction) {
    std::vector<float> v(24);
    for (int i = 0; i < 24; ++i) v[i] = static_cast<float>(i - 5);  // 含负数
    const Tensor cpu(v, {2, 3, 4});
    const Tensor a = moved_to(cpu, dev);

    // 参考值：对轴 {0, 2} 归约，保留轴 1（长度 3）。
    float w_sum[3] = {0, 0, 0}, w_min[3], w_max[3];
    for (int b = 0; b < 3; ++b) {
        w_min[b] = std::numeric_limits<float>::max();
        w_max[b] = std::numeric_limits<float>::lowest();
    }
    for (int i = 0; i < 24; ++i) {
        const int b = (i / 4) % 3;
        w_sum[b] += v[i];
        w_min[b] = std::min(w_min[b], v[i]);
        w_max[b] = std::max(w_max[b], v[i]);
    }

    const Tensor s = ops::Sum(a, {0, 2});
    EXPECT_EQ(s.dims(), 1u);
    EXPECT_EQ(s.shape(0), 3);
    expect_close(host_values<float>(s), std::vector<float>(w_sum, w_sum + 3), "sum {0,2}");
    expect_close(host_values<float>(ops::Min(a, {0, 2})), std::vector<float>(w_min, w_min + 3), "min {0,2}");
    expect_close(host_values<float>(ops::Max(a, {0, 2})), std::vector<float>(w_max, w_max + 3), "max {0,2}");
    // Mean 先按 float_reduction_type 累加，平均值为 8 个元素之和除以 8。
    float w_mean[3];
    for (int b = 0; b < 3; ++b) w_mean[b] = w_sum[b] / 8.0f;
    expect_close(host_values<float>(ops::Mean(a, {0, 2})), std::vector<float>(w_mean, w_mean + 3), "mean {0,2}");

    // keepdim：被归约的轴保留为长度 1。
    const Tensor k = ops::Sum(a, {0, 2}, true);
    EXPECT_EQ(k.dims(), 3u);
    EXPECT_EQ(k.shape(0), 1);
    EXPECT_EQ(k.shape(1), 3);
    EXPECT_EQ(k.shape(2), 1);
    expect_close(host_values<float>(k), std::vector<float>(w_sum, w_sum + 3), "sum {0,2} keepdim");

    // 负轴从尾部数起，与正轴等价。
    expect_close(host_values<float>(ops::Sum(a, {-3, -1})), std::vector<float>(w_sum, w_sum + 3),
                 "negative axes");

    // 空轴列表 = 归约所有轴，得到 0 维张量。
    const Tensor full = ops::Sum(a, std::vector<int64_t>{});
    EXPECT_EQ(full.dims(), 0u);
    float total = 0;
    for (float x : v) total += x;
    EXPECT_FLOAT_EQ(full.item<float>(), total);
    const Tensor full_keep = ops::Sum(a, std::vector<int64_t>{}, true);
    EXPECT_EQ(full_keep.dims(), 3u);
    EXPECT_EQ(full_keep.shape(0), 1);
    EXPECT_EQ(full_keep.shape(1), 1);
    EXPECT_EQ(full_keep.shape(2), 1);

    // 重复轴、越界轴都要被拒绝。
    EXPECT_THROW(ops::Sum(a, {0, 0}), std::exception);
    EXPECT_THROW(ops::Sum(a, std::vector<int64_t>{3}), std::exception);
    // Var/Std/Norm 无法逐轴折叠，多轴形式明确拒绝。
    EXPECT_THROW((ops::Var(a, {0, 2})), std::exception);
}

// Prod 目前只有 CPU 实现，多轴形式单独在 host 上验证。
TEST(Reduce, MultiAxisProdCpu) {
    std::vector<float> v(24);
    for (int i = 0; i < 24; ++i) v[i] = static_cast<float>(i - 5);
    const Tensor a(v, {2, 3, 4});
    float want[3] = {1, 1, 1};
    for (int i = 0; i < 24; ++i) want[(i / 4) % 3] *= v[i];
    expect_close(host_values<float>(ops::Prod(a, {0, 2})), std::vector<float>(want, want + 3), "prod {0,2}");
}

// ---------- FR-3：带 dtype 的归约 ----------
TEST_F(DeviceOps, ReductionDtype) {
    std::vector<float> v{1, 2, 3, 4, 5, 6};
    const Tensor a = moved_to(Tensor(v, {2, 3}), dev);

    const Tensor s64 = ops::Sum(a, DataType::FLOAT64);
    EXPECT_EQ(s64.dtype(), DataType::FLOAT64);
    EXPECT_EQ(s64.dims(), 0u);
    EXPECT_DOUBLE_EQ(s64.item<double>(), 21.0);

    const Tensor i32 = ops::Sum(a, 0, false, DataType::INT32);
    EXPECT_EQ(i32.dtype(), DataType::INT32);
    EXPECT_EQ(i32.shape(0), 3);
    const auto i32v = host_values<int32_t>(i32);
    EXPECT_EQ(i32v[0], 5);  // 1 + 4
    EXPECT_EQ(i32v[2], 9);  // 3 + 6

    const Tensor many = ops::Mean(a, {0, 1}, false, DataType::FLOAT64);
    EXPECT_EQ(many.dtype(), DataType::FLOAT64);
    EXPECT_DOUBLE_EQ(many.item<double>(), 3.5);

    // BOOL 不是合法的归约输出 dtype。
    EXPECT_THROW(ops::Sum(a, DataType::BOOL), std::exception);
    EXPECT_THROW(ops::Mean(a, 0, false, DataType::BOOL), std::exception);
}

// ---------- FR-3：all/any 是 0 维 BOOL 张量 ----------
TEST_F(DeviceOps, AllAnyTruthiness) {
    const Tensor a = moved_to(Tensor::Fill({2, 2}, 3.0f), dev);
    const Tensor all = ops::All(a);
    EXPECT_EQ(all.dims(), 0u);
    EXPECT_EQ(all.dtype(), DataType::BOOL);
    EXPECT_TRUE(all.item<bool>());
    EXPECT_TRUE(ops::Any(a).item<bool>());

    Tensor cz = Tensor::Fill({2, 2}, 3.0f);
    cz.set<float>({0, 1}, 0.0f);  // set() 只在 CPU 上有定义
    const Tensor with_zero = moved_to(cz, dev);
    EXPECT_FALSE(ops::All(with_zero).item<bool>());
    EXPECT_TRUE(ops::Any(with_zero).item<bool>());

    // BOOL 输入走同一条真值路径（内部经 INT8 折叠）。
    const Tensor pred = with_zero > 0.5f;
    EXPECT_EQ(pred.dtype(), DataType::BOOL);
    EXPECT_FALSE(ops::All(pred).item<bool>());
    EXPECT_TRUE(ops::Any(pred).item<bool>());

    Tensor zeros = Tensor::Zeros({2, 2}, DataType::FLOAT32);
    EXPECT_FALSE(ops::Any(zeros).item<bool>());

    // 空张量：All 空洞为真，Any 为假。
    const Tensor e{moved_to(Tensor::Zeros({2, 0}, DataType::FLOAT32), dev)};
    EXPECT_TRUE(ops::All(e).item<bool>());
    EXPECT_FALSE(ops::Any(e).item<bool>());
    EXPECT_EQ(ops::All(e).dims(), 0u);
}

// ---------- FR-4：视图族 ----------
TEST(View, NarrowSelectExpand) {
    std::vector<float> v{1, 2, 3, 4, 5, 6, 7, 8, 9};
    Tensor a(v, {3, 3});

    Tensor rows = a.narrow(0, 1, 2);  // 行 1..2
    EXPECT_EQ(rows.dims(), 2u);
    EXPECT_EQ(rows.shape(0), 2);
    EXPECT_FLOAT_EQ(rows.at<float>({0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(rows.at<float>({1, 2}), 9.0f);

    Tensor col = a.select(1, 2);  // 第 2 列，秩 1
    EXPECT_EQ(col.dims(), 1u);
    EXPECT_EQ(col.shape(0), 3);
    EXPECT_FLOAT_EQ(col.at<float>({2}), 9.0f);
    EXPECT_FLOAT_EQ(col.at<float>({0}), 3.0f);

    // expand：长度 1 的维度可扩，其余维度必须已经匹配。
    Tensor row0 = a.select(0, 0).unsqueeze(0);  // (1, 3)
    Tensor big = row0.expand({4, 3});
    EXPECT_EQ(big.shape(0), 4);
    EXPECT_EQ(big.strides(0), 0);
    EXPECT_FLOAT_EQ(big.at<float>({3, 2}), 3.0f);
    EXPECT_THROW(a.expand({3, 4}), std::exception);

    // 写穿视图：set() 落在共享存储上，源张量同步可见。
    rows.set<float>({0, 0}, 100.0f);
    EXPECT_FLOAT_EQ(a.at<float>({1, 0}), 100.0f);

    // 越界参数被拒绝。
    EXPECT_THROW(a.narrow(0, 2, 2), std::exception);
    EXPECT_THROW(a.select(0, 3), std::exception);
    EXPECT_THROW(a.select(2, 0), std::exception);
}

TEST(View, SqueezeUnsqueeze) {
    Tensor a = Tensor::Fill({1, 3, 1}, 2.0f);
    Tensor sq = a.squeeze();
    EXPECT_EQ(sq.dims(), 1u);
    EXPECT_EQ(sq.shape(0), 3);
    EXPECT_EQ(a.dims(), 3u);  // 源不变

    // squeeze(dim) 只在长度为 1 的轴上生效。
    EXPECT_EQ(a.squeeze(0).dims(), 2u);
    EXPECT_EQ(a.squeeze(1).dims(), 3u);
    EXPECT_THROW(a.squeeze(3), std::exception);

    Tensor u = sq.unsqueeze(0);
    EXPECT_EQ(u.dims(), 2u);
    EXPECT_EQ(u.shape(0), 1);
    EXPECT_EQ(u.strides(0), 3);
    Tensor tail = sq.unsqueeze(1);
    EXPECT_EQ(tail.shape(1), 1);
    EXPECT_EQ(tail.strides(1), 1);
    EXPECT_THROW(sq.unsqueeze(5), std::exception);
}

TEST(View, ReshapeViewAndCopy) {
    Tensor a = Tensor::Fill({2, 3}, 1.0f);
    Tensor r = a.reshape({3, 2});
    EXPECT_EQ(r.dims(), 2u);
    EXPECT_EQ(a.dims(), 2u);  // reshape 不再原地改形状
    r.set<float>({0, 1}, 7.0f);
    EXPECT_FLOAT_EQ(a.at<float>({0, 1}), 7.0f);  // 连续张量上是视图

    // -1 推断一维。
    EXPECT_EQ(a.reshape({-1}).shape(0), 6);
    EXPECT_THROW(a.reshape({4}), std::exception);

    // 非连续张量的 reshape 返回新拷贝，源数据不变。
    Tensor base = Tensor::Fill({2, 3}, 1.0f);
    Tensor p = base.permute({1, 0});  // (3, 2) 非连续
    EXPECT_FALSE(p.is_contiguous());
    Tensor pc = p.reshape({6});
    EXPECT_TRUE(pc.is_contiguous());
    EXPECT_EQ(pc.numel(), 6u);
    pc.set<float>({0}, 42.0f);
    EXPECT_FLOAT_EQ(base.at<float>({0, 0}), 1.0f);  // 拷贝，不写穿
}

// ---------- FR-5：元素写入、nonzero 索引、masked 运算 ----------
TEST(Indexing, ElementWrite) {
    Tensor a = Tensor::Zeros({2, 2}, DataType::INT32);
    a.set<int32_t>({1, 0}, 5);
    EXPECT_EQ(a.at<int32_t>({1, 0}), 5);
    EXPECT_THROW(a.set<int32_t>({2, 0}, 1), std::exception);

    Tensor b = Tensor::Zeros({2}, DataType::BOOL);
    b.set<bool>({0}, true);
    EXPECT_EQ(b.at<bool>({0}), 1);
}

TEST(Indexing, NonzeroIndices) {
    std::vector<float> v{0, 2, 0, 3, 0, 0};
    Tensor a(v, {2, 3});
    Tensor idx = ops::Nonzero(a);
    EXPECT_EQ(idx.dtype(), DataType::INT64);
    EXPECT_EQ(idx.dims(), 2u);
    EXPECT_EQ(idx.shape(0), 2);
    EXPECT_EQ(idx.shape(1), 2);
    auto hv = host_values<int64_t>(idx);
    ASSERT_EQ(hv.size(), 4u);
    EXPECT_EQ(hv[0], 0); EXPECT_EQ(hv[1], 1);  // (0,1)
    EXPECT_EQ(hv[2], 1); EXPECT_EQ(hv[3], 0);  // (1,0)

    // 1 维输入返回 1 维索引。
    std::vector<float> one{0, 5, 0, 7};
    Tensor b(one, {4});
    Tensor one_idx = ops::Nonzero(b);
    EXPECT_EQ(one_idx.dims(), 1u);
    expect_close<int64_t>(host_values<int64_t>(one_idx), {1, 3}, "1d nonzero");

    // 空张量给出 (0, dims)。
    Tensor e = Tensor::Zeros({0, 3}, DataType::FLOAT32);
    EXPECT_EQ(ops::Nonzero(e).shape(0), 0);
    EXPECT_EQ(ops::Nonzero(e).shape(1), 2);

    // BOOL 张量同样可用。
    std::vector<int8_t> flags{1, 0, 1};
    Tensor f(flags, {3});
    EXPECT_EQ(ops::CountNonzero(f), 2u);
}

// 设备上的 nonzero 通过一次 host 回拷完成，语义与 CPU 一致。
TEST_F(DeviceOps, NonzeroOnDevice) {
    std::vector<float> v{0, 2, 0, 3, 0, 0};
    const Tensor a = moved_to(Tensor(v, {2, 3}), dev);
    const Tensor idx = ops::Nonzero(a);
    EXPECT_EQ(idx.dtype(), DataType::INT64);
    EXPECT_EQ(idx.shape(0), 2);
    const auto hv = host_values<int64_t>(idx);
    ASSERT_EQ(hv.size(), 4u);
    EXPECT_EQ(hv[0], 0); EXPECT_EQ(hv[1], 1);
    EXPECT_EQ(hv[2], 1); EXPECT_EQ(hv[3], 0);
}

TEST(Masked, SelectAndFill) {
    std::vector<float> v{1, 2, 3, 4, 5, 6};
    Tensor x(v, {2, 3});
    Tensor m = x > 3.0f;  // BOOL (2,3)

    Tensor sel = ops::MaskedSelect(x, m);
    EXPECT_EQ(sel.dims(), 1u);
    expect_close(host_values<float>(sel), {4.0f, 5.0f, 6.0f}, "masked_select");

    Tensor filled = ops::MaskedFill(x, m, -1.0f);
    EXPECT_EQ(filled.dtype(), DataType::FLOAT32);
    expect_close(host_values<float>(filled), {1.0f, 2.0f, 3.0f, -1.0f, -1.0f, -1.0f}, "masked_fill");
    EXPECT_FLOAT_EQ(x.at<float>({1, 0}), 4.0f);  // 不修改输入

    // 可广播的 mask：形状 (3) 作用到 (2,3)。
    Tensor row_mask = ops::GreaterEqual(x.select(0, 0), 2.0f);
    EXPECT_EQ(row_mask.dims(), 1u);
    Tensor bc = ops::MaskedFill(x, row_mask, 0.0f);
    expect_close(host_values<float>(bc), {1.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f}, "masked_fill broadcast");

    // mask 比输入大（会扩大结果）要拒绝。
    Tensor wide = Tensor::Zeros({2, 3}, DataType::BOOL);
    Tensor narrow_in = Tensor::Zeros({3}, DataType::FLOAT32);
    EXPECT_THROW(ops::MaskedFill(narrow_in, wide, 1.0f), std::exception);

    // mask 必须是 BOOL 或整型。
    Tensor float_mask = Tensor::Ones({2, 3}, DataType::FLOAT32);
    EXPECT_THROW(ops::MaskedSelect(x, float_mask), std::exception);
}

// ---------- FR-6：显式原地写入 ----------
TEST(Tensor, InplaceFamily) {
    Tensor a = Tensor::Fill({2, 2}, 2.0f);
    Tensor alias = a;  // 共享存储

    a.add_(1.0f);
    EXPECT_FLOAT_EQ(alias.at<float>({0, 0}), 3.0f);
    a.mul_(2.0f);
    EXPECT_FLOAT_EQ(alias.at<float>({1, 1}), 6.0f);
    a.sub_(1.0f).div_(5.0f);
    EXPECT_FLOAT_EQ(alias.at<float>({0, 0}), 1.0f);

    a.fill_(-4.0f);
    EXPECT_FLOAT_EQ(alias.at<float>({0, 0}), -4.0f);
    a.abs_();
    EXPECT_FLOAT_EQ(alias.at<float>({0, 0}), 4.0f);
    a.clamp_(0.0f, 1.0f);
    EXPECT_FLOAT_EQ(alias.at<float>({0, 0}), 1.0f);
    a.zero_();
    EXPECT_FLOAT_EQ(alias.at<float>({0, 0}), 0.0f);

    Tensor b = Tensor::Fill({2, 2}, 7.0f);
    a.copy_(b);
    EXPECT_FLOAT_EQ(alias.at<float>({0, 0}), 7.0f);
    EXPECT_THROW(a.copy_(Tensor::Fill({3}, 1.0f)), std::exception);

    // 整型张量不能被浮点标量原地提升（add_/add 同一规则）。
    Tensor i = Tensor::Fill({2}, 3.0f, DataType::INT32);
    EXPECT_THROW(i.add_(0.5f), std::exception);
    EXPECT_THROW(i.add_(2.0f), std::exception);
    Tensor src = Tensor::Fill({2}, 5.0f, DataType::INT32);
    i.copy_(src);
    EXPECT_EQ(i.at<int32_t>({0}), 5);
}

// ---------- FR-7：TensorOptions 与默认设备 ----------
TEST(Options, DtypeAndDeviceCarriedTogether) {
    Tensor a = Tensor::Zeros({2, 2}, TensorOptions(DataType::INT32));
    EXPECT_EQ(a.dtype(), DataType::INT32);
    EXPECT_EQ(a.device(), Device::CPU);

    Tensor b = Tensor::Ones({3}, TensorOptions(DataType::FLOAT64, Device::CPU));
    EXPECT_EQ(b.dtype(), DataType::FLOAT64);
    EXPECT_EQ(b.numel(), 3u);

    Tensor c({4}, TensorOptions(DataType::INT64));
    EXPECT_EQ(c.dtype(), DataType::INT64);
    EXPECT_EQ(c.numel(), 4u);

    // with_* 链式修改。
    TensorOptions o;
    o.with_dtype(DataType::FLOAT32).with_device(DeviceId{Device::CPU, 0});
    Tensor d = Tensor::Fill({2}, 1.5f, o);
    EXPECT_EQ(d.dtype(), DataType::FLOAT32);
    EXPECT_FLOAT_EQ(d.at<float>({1}), 1.5f);

    // ops 侧的 options 形式与 DeviceId 形式。
    Tensor e = ops::Ones({2, 2}, TensorOptions(DataType::INT8));
    EXPECT_EQ(e.dtype(), DataType::INT8);
    Tensor f = ops::Zeros({2}, DataType::FLOAT32, DeviceId{Device::CPU, 0});
    EXPECT_EQ(f.dtype(), DataType::FLOAT32);
}

TEST(Options, DefaultDevice) {
    const DeviceId prev = default_device();
    set_default_device(DeviceId{Device::CPU, 0});
    EXPECT_EQ(default_device().kind, Device::CPU);
    EXPECT_EQ(default_device().index, 0);

    // 默认设备参与 TensorOptions 的解析。
    EXPECT_EQ(TensorOptions{}.resolved_id().kind, Device::CPU);

    EXPECT_THROW(set_default_device(DeviceId{Device::CPU, 1}), std::exception);
    EXPECT_THROW(set_default_device(DeviceId{Device::CPU, -1}), std::exception);

    set_default_device(prev);
}
