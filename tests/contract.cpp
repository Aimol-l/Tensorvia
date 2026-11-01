#include "test_support.hpp"

// 契约用例：FR-1 空张量、FR-2 dtype、FR-3 归约与打印。
// ---------- 27. FR-2 dtype 契约：提升规则 ----------
TEST(DtypeContract, ComputeTypePromotion) {
    // BOOL 是最弱的整型：与任何类型相遇都由对方获胜。
    EXPECT_EQ(compute_type(DataType::BOOL, DataType::BOOL), DataType::BOOL);
    EXPECT_EQ(compute_type(DataType::BOOL, DataType::INT8), DataType::INT8);
    EXPECT_EQ(compute_type(DataType::INT32, DataType::BOOL), DataType::INT32);

    // 整型取更宽的一方。
    EXPECT_EQ(compute_type(DataType::INT8, DataType::INT32), DataType::INT32);
    EXPECT_EQ(compute_type(DataType::INT16, DataType::INT64), DataType::INT64);

    // 两个半精度浮点合到 FLOAT32，而不是取更宽的那个。
    EXPECT_EQ(compute_type(DataType::FLOAT16, DataType::BFLOAT16), DataType::FLOAT32);
    EXPECT_EQ(compute_type(DataType::FLOAT16, DataType::FLOAT16), DataType::FLOAT16);
    EXPECT_EQ(compute_type(DataType::FLOAT16, DataType::FLOAT64), DataType::FLOAT64);

    // 整型遇到浮点跟随浮点。
    EXPECT_EQ(compute_type(DataType::INT64, DataType::FLOAT32), DataType::FLOAT32);
    EXPECT_EQ(compute_type(DataType::FLOAT64, DataType::INT8), DataType::FLOAT64);

    // 弱标量：浮点张量保持自身 dtype，整型/BOOL 张量采用默认浮点 dtype。
    EXPECT_EQ(scalar_promote_type(DataType::FLOAT16), DataType::FLOAT16);
    EXPECT_EQ(scalar_promote_type(DataType::FLOAT64), DataType::FLOAT64);
    EXPECT_EQ(scalar_promote_type(DataType::INT32), default_dtype());
    EXPECT_EQ(scalar_promote_type(DataType::BOOL), default_dtype());
}

// 9×9 对照表：行 = 左操作数，列 = 右操作数。表是按 PyTorch result_type 写死的，
// 不调用 compute_type 自己推导——实现和测试同时改错时测试不会跟着一起错。
TEST(DtypeContract, FullPromotionMatrix) {
    using DT = DataType;
    const DT all[9] = {DT::BOOL,    DT::INT8,    DT::INT16,   DT::INT32,  DT::INT64,
                       DT::BFLOAT16, DT::FLOAT16, DT::FLOAT32, DT::FLOAT64};
    const DT expected[9][9] = {
        {DT::BOOL,     DT::INT8,     DT::INT16,     DT::INT32,     DT::INT64,
         DT::BFLOAT16, DT::FLOAT16,  DT::FLOAT32,   DT::FLOAT64},
        {DT::INT8,     DT::INT8,     DT::INT16,     DT::INT32,     DT::INT64,
         DT::BFLOAT16, DT::FLOAT16,  DT::FLOAT32,   DT::FLOAT64},
        {DT::INT16,    DT::INT16,    DT::INT16,     DT::INT32,     DT::INT64,
         DT::BFLOAT16, DT::FLOAT16,  DT::FLOAT32,   DT::FLOAT64},
        {DT::INT32,    DT::INT32,    DT::INT32,     DT::INT32,     DT::INT64,
         DT::BFLOAT16, DT::FLOAT16,  DT::FLOAT32,   DT::FLOAT64},
        {DT::INT64,    DT::INT64,    DT::INT64,     DT::INT64,     DT::INT64,
         DT::BFLOAT16, DT::FLOAT16,  DT::FLOAT32,   DT::FLOAT64},
        {DT::BFLOAT16, DT::BFLOAT16, DT::BFLOAT16,  DT::BFLOAT16,  DT::BFLOAT16,
         DT::BFLOAT16, DT::FLOAT32,  DT::FLOAT32,   DT::FLOAT64},
        {DT::FLOAT16,  DT::FLOAT16,  DT::FLOAT16,   DT::FLOAT16,   DT::FLOAT16,
         DT::FLOAT32,  DT::FLOAT16,  DT::FLOAT32,   DT::FLOAT64},
        {DT::FLOAT32,  DT::FLOAT32,  DT::FLOAT32,   DT::FLOAT32,   DT::FLOAT32,
         DT::FLOAT32,  DT::FLOAT32,  DT::FLOAT32,   DT::FLOAT64},
        {DT::FLOAT64,  DT::FLOAT64,  DT::FLOAT64,   DT::FLOAT64,   DT::FLOAT64,
         DT::FLOAT64,  DT::FLOAT64,  DT::FLOAT64,   DT::FLOAT64},
    };
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            EXPECT_EQ(compute_type(all[i], all[j]), expected[i][j]) << "row " << i << ", col " << j;
        }
    }
}

TEST(DtypeContract, TensorArithmeticPromotion) {
    Tensor i16  = Tensor::Ones({2, 2}, DataType::INT16);
    Tensor i64  = Tensor::Ones({2, 2}, DataType::INT64);
    Tensor f16  = Tensor::Ones({2, 2}, DataType::FLOAT16);
    Tensor bf16 = Tensor::Ones({2, 2}, DataType::BFLOAT16);
    Tensor f32  = Tensor::Ones({2, 2}, DataType::FLOAT32);
    Tensor f64  = Tensor::Ones({2, 2}, DataType::FLOAT64);

    EXPECT_EQ((i16 + i64).dtype(), DataType::INT64);
    EXPECT_EQ((f16 + bf16).dtype(), DataType::FLOAT32);
    EXPECT_EQ((f16 * f32).dtype(), DataType::FLOAT32);
    EXPECT_EQ((i64 / f32).dtype(), DataType::FLOAT32);
    EXPECT_EQ((f32 + f64).dtype(), DataType::FLOAT64);

    // 弱标量：浮点字面量不得把浮点张量抬宽。
    EXPECT_EQ((f16 + 2.0f).dtype(), DataType::FLOAT16);
    EXPECT_EQ((f64 * 2.0f).dtype(), DataType::FLOAT64);
    // 整型张量遇到浮点标量则采用默认浮点 dtype。
    Tensor i32 = Tensor::Ones({2, 2}, DataType::INT32);
    EXPECT_EQ((i32 + 0.5f).dtype(), default_dtype());
}


// ---------- 28. FR-2 dtype 契约：BOOL 算术被拒 ----------
TEST(DtypeContract, BoolArithmeticRejected) {
    std::vector<std::int32_t> bits{0, 1, 1, 0};
    const Tensor bits_i32(bits, {2, 2});
    Tensor b = ops::Typecast(bits_i32, DataType::BOOL);
    ASSERT_EQ(b.dtype(), DataType::BOOL);
    Tensor i8 = Tensor::Ones({2, 2}, DataType::INT8);

    EXPECT_THROW((b + i8), std::exception);        // 张量-张量
    EXPECT_THROW((b * 2.0f), std::exception);      // 张量-标量
    EXPECT_THROW(ops::Abs(b), std::exception);
    EXPECT_THROW(ops::Tanh(b), std::exception);
    EXPECT_THROW(ops::Mean(b), std::exception);
    EXPECT_THROW((ops::Mean(b, 0)), std::exception);
    EXPECT_THROW(ops::Prod(b), std::exception);
    EXPECT_THROW((ops::Var(b, 0)), std::exception);

    // 允许：比较、谓词与归约。
    EXPECT_EQ((b == b).dtype(), DataType::BOOL);
    EXPECT_EQ((b != b).dtype(), DataType::BOOL);
    EXPECT_FALSE(ops::All(b, 1.0f));   // 0 也在其中
    EXPECT_TRUE(ops::Any(b, 1.0f));
    EXPECT_EQ(ops::CountNonzero(b), 2u);
    EXPECT_EQ((ops::Sum(b, 0)).dtype(), DataType::INT64);
    EXPECT_EQ((ops::Max(b, 0)).dtype(), DataType::BOOL);
    EXPECT_EQ((ops::Min(b, 0)).dtype(), DataType::BOOL);
}


// ---------- 29. FR-2 dtype 契约：BOOL cast 与 0/1 规范化 ----------
TEST(DtypeContract, BoolCastsAndNormalization) {
    std::vector<std::int32_t> v{-1, 0, 2, 7};
    const Tensor i32(v, {4});
    Tensor b = ops::Typecast(i32, DataType::BOOL);
    EXPECT_EQ(b.dtype(), DataType::BOOL);
    EXPECT_EQ(calc_dtype_size(b.dtype()), 1u);  // 每元素 1 字节，从不按位打包
    EXPECT_EQ(b.at<bool>({0}), 1);  // -1 非零为真
    EXPECT_EQ(b.at<bool>({1}), 0);
    EXPECT_EQ(b.at<bool>({2}), 1);

    // BOOL -> 数值恒为 0/1，绝不泄出原始字节。
    Tensor back = b.to_type(DataType::INT32);
    EXPECT_EQ(back.dtype(), DataType::INT32);
    EXPECT_EQ(back.at<std::int32_t>({0}), 1);
    EXPECT_EQ(back.at<std::int32_t>({1}), 0);
    Tensor as_f64 = b.to_type(DataType::FLOAT64);
    EXPECT_DOUBLE_EQ(as_f64.at<double>({3}), 1.0);

    // 外部写入的非 0/1 字节：读取按非零为真，比较前先规范化。
    auto* raw = static_cast<std::uint8_t*>(b.data());
    raw[0] = 3;
    EXPECT_EQ(b.at<bool>({0}), 1);
    const Tensor& b_ref = b;
    EXPECT_EQ(ops::Typecast(b_ref, DataType::INT32).at<std::int32_t>({0}), 1);
    std::vector<std::int32_t> ones_v{1, 0, 1, 1};
    const Tensor ones_i32(ones_v, {4});
    Tensor one = ops::Typecast(ones_i32, DataType::BOOL);
    EXPECT_EQ((b == one).at<bool>({0}), 1);  // 3 与 1 同为真

    // 写入侧规范化：Fill / Zeros 只产生 0 或 1。
    Tensor filled = Tensor::Zeros({4}, DataType::BOOL);
    ops::Fill(filled, 5.0f);
    const auto* filled_raw = static_cast<const std::uint8_t*>(filled.data());
    EXPECT_EQ(filled_raw[0], 1);
    EXPECT_EQ(filled_raw[3], 1);
    Tensor quarter = Tensor::Fill({2}, 0.25f, DataType::BOOL);
    EXPECT_EQ(static_cast<const std::uint8_t*>(quarter.data())[0], 1);
}


// ---------- 30. FR-2 dtype 契约：默认 dtype ----------
TEST(DtypeContract, DefaultDtype) {
    EXPECT_EQ(default_dtype(), DataType::FLOAT32);

    set_default_dtype(DataType::FLOAT64);
    EXPECT_EQ(default_dtype(), DataType::FLOAT64);
    EXPECT_EQ(Tensor::Zeros({2, 2}).dtype(), DataType::FLOAT64);
    EXPECT_EQ(Tensor::Fill({2}, 1.0f).dtype(), DataType::FLOAT64);
    EXPECT_EQ(ops::Arange(0.0f, 3.0f).dtype(), DataType::FLOAT64);
    EXPECT_EQ(ops::Eye(2).dtype(), DataType::FLOAT64);
    // 默认 dtype 同时决定整型张量的弱标量提升。
    Tensor i32 = Tensor::Ones({2, 2}, DataType::INT32);
    EXPECT_EQ((i32 + 0.5f).dtype(), DataType::FLOAT64);

    // 只有浮点可作为默认 dtype。
    EXPECT_THROW(set_default_dtype(DataType::INT32), std::exception);
    EXPECT_THROW(set_default_dtype(DataType::BOOL), std::exception);

    set_default_dtype(DataType::FLOAT32);
    EXPECT_EQ(default_dtype(), DataType::FLOAT32);
}


// ---------- 31. FR-2 dtype 契约：归约结果 dtype ----------
TEST(DtypeContract, ReductionDtypes) {
    Tensor i8  = Tensor::Ones({2, 3}, DataType::INT8);
    Tensor i32 = Tensor::Ones({2, 3}, DataType::INT32);
    Tensor f32 = Tensor::Ones({2, 3}, DataType::FLOAT32);
    Tensor f64 = Tensor::Ones({2, 3}, DataType::FLOAT64);

    // 整型归约提升到 INT64（PyTorch reduction_type），浮点保持自身 dtype。
    auto s8 = ops::Sum(i8, 0);
    EXPECT_EQ(s8.dtype(), DataType::INT64);
    EXPECT_EQ(s8.at<std::int64_t>({0}), 2);
    EXPECT_EQ((ops::Sum(i32, 1)).dtype(), DataType::INT64);
    EXPECT_EQ((ops::Sum(f32, 1)).dtype(), DataType::FLOAT32);
    EXPECT_EQ((ops::Sum(f64, 1)).dtype(), DataType::FLOAT64);

    // Mean 走浮点：整型输入变 FLOAT32。
    EXPECT_EQ((ops::Mean(i32, 0)).dtype(), DataType::FLOAT32);
    EXPECT_EQ((ops::Mean(f64, 0)).dtype(), DataType::FLOAT64);

    // Min/Max 保持输入 dtype；argmax/argmin 恒为 INT64。
    EXPECT_EQ((ops::Min(i8, 0)).dtype(), DataType::INT8);
    EXPECT_EQ((ops::Max(f32, 0)).dtype(), DataType::FLOAT32);
    EXPECT_EQ((ops::Argmax(f32, 1)).dtype(), DataType::INT64);
    EXPECT_EQ((ops::Argmin(i32, 1)).dtype(), DataType::INT64);
}


// ---------- 32. FR-3 归约契约：0-dim 结果、item<T>()、keepdim、负轴 ----------
TEST(ReduceContract, FullReductionIsRankZero) {
    std::vector<float> v{1, 2, 3, 4, 5, 6};
    Tensor t(v, {2, 3});

    // 全归约 = 归约掉所有轴 -> 0-dim 张量，值仍在原设备上。
    Tensor s = ops::Sum(t);
    EXPECT_EQ(s.dims(), 0u);
    EXPECT_EQ(s.numel(), 1u);
    EXPECT_EQ(s.dtype(), DataType::FLOAT32);
    EXPECT_FLOAT_EQ(s.item<float>(), 21.0f);
    EXPECT_FLOAT_EQ(s.at<float>({}), 21.0f);  // 空索引列表读 0-dim

    // 与 1-D 输入 reduce 掉唯一一维的形状规则一致（0-dim 再全归约仍是 0-dim）。
    Tensor scalar_sum = ops::Sum(s);
    EXPECT_EQ(scalar_sum.dims(), 0u);
    EXPECT_FLOAT_EQ(scalar_sum.item<float>(), 21.0f);
    EXPECT_FLOAT_EQ(ops::Max(t).item<float>(), 6.0f);
    EXPECT_FLOAT_EQ(ops::Mean(t).item<float>(), 3.5f);

    // 0-dim 参与广播：等价于标量。
    Tensor shifted = t + s;
    EXPECT_EQ(shifted.dims(), 2u);
    EXPECT_FLOAT_EQ(shifted.at<float>({0, 0}), 22.0f);
    EXPECT_FLOAT_EQ(shifted.at<float>({1, 2}), 27.0f);
}

TEST(ReduceContract, KeepDimAndNegativeAxis) {
    std::vector<float> v{1, 2, 3, 4, 5, 6};
    Tensor t(v, {2, 3});

    Tensor keep = ops::Sum(t, 1, true);
    EXPECT_EQ(keep.dims(), 2u);
    EXPECT_EQ(keep.shape(0), 2);
    EXPECT_EQ(keep.shape(1), 1);
    EXPECT_FLOAT_EQ(keep.at<float>({0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(keep.at<float>({1, 0}), 15.0f);

    // keepdim 结果可直接与原张量广播，PyTorch 的常见写法。
    Tensor normed = t / keep;
    EXPECT_FLOAT_EQ(normed.at<float>({0, 0}), 1.0f / 6.0f);

    // 负轴从末尾数：-1 == 1。
    Tensor neg = ops::Sum(t, -1);
    EXPECT_EQ(neg.dims(), 1u);
    EXPECT_FLOAT_EQ(neg.at<float>({0}), 6.0f);
    EXPECT_FLOAT_EQ(neg.at<float>({1}), 15.0f);

    // 越界轴与 0-dim 上的轴都拒绝。
    EXPECT_THROW(ops::Sum(t, 2), std::exception);
    EXPECT_THROW(ops::Sum(t, -3), std::exception);
    EXPECT_THROW(ops::Sum(ops::Sum(t), 0), std::exception);
}

TEST(ReduceContract, ItemGuardsAndReductionDtypes) {
    Tensor two = Tensor::Zeros({2}, DataType::FLOAT32);
    EXPECT_THROW(two.item<float>(), std::exception);  // numel != 1（元素数不为 1）

    // item() 读的是“这个视图的那一个元素”，不是底层 buffer 起点。
    std::vector<float> vals{1, 2, 3};
    Tensor v(vals, {3});
    EXPECT_FLOAT_EQ(v.slice({{1, 2}}).item<float>(), 2.0f);      // offset 视图
    std::vector<float> mv{1, 2, 3, 4};
    Tensor m(mv, {2, 2});
    EXPECT_FLOAT_EQ(m.t().slice({{0, 1}, {0, 1}}).item<float>(), 1.0f);  // 非连续视图

    // item<T>() 从归约结果的 dtype 转换：INT16 -> INT64 归约，再取回 int/float。
    Tensor i16 = Tensor::Ones({4}, DataType::INT16);
    Tensor s16 = ops::Sum(i16);
    EXPECT_EQ(s16.dtype(), DataType::INT64);
    EXPECT_EQ(s16.item<int>(), 4);
    EXPECT_FLOAT_EQ(s16.item<float>(), 4.0f);

    // BOOL：Sum 走 INT64，Min/Max 保持 BOOL，Mean 拒绝。
    std::vector<std::int32_t> ones{1, 1, 1, 1};
    const Tensor ones_i32(ones, {2, 2});
    Tensor b = ops::Typecast(ones_i32, DataType::BOOL);
    Tensor bs = ops::Sum(b);
    EXPECT_EQ(bs.dtype(), DataType::INT64);
    EXPECT_EQ(bs.item<std::int64_t>(), 4);
    EXPECT_EQ(ops::Min(b).dtype(), DataType::BOOL);
    EXPECT_TRUE(ops::Min(b).item<bool>());
    EXPECT_THROW(ops::Mean(b), std::exception);

    // f64 全归约保持双精度；整型 Mean 走浮点。
    Tensor f64 = Tensor::Ones({2, 3}, DataType::FLOAT64);
    EXPECT_EQ(ops::Sum(f64).dtype(), DataType::FLOAT64);
    EXPECT_EQ(ops::Prod(f64).dtype(), DataType::FLOAT64);
    EXPECT_EQ(ops::Mean(f64).dtype(), DataType::FLOAT64);
    EXPECT_EQ(ops::Var(f64).dtype(), DataType::FLOAT64);
    EXPECT_EQ(ops::Std(f64).dtype(), DataType::FLOAT64);
    EXPECT_EQ(ops::Norm(f64, 2.0f).dtype(), DataType::FLOAT64);
    EXPECT_EQ(ops::Mean(i16).dtype(), DataType::FLOAT32);
}

TEST(ReduceContract, RankZeroConstruction) {
    Tensor z = Tensor::Zeros({}, DataType::FLOAT32);
    EXPECT_EQ(z.dims(), 0u);
    EXPECT_EQ(z.numel(), 1u);
    EXPECT_FLOAT_EQ(z.at<float>({}), 0.0f);
    EXPECT_EQ(ops::Sum(z).dims(), 0u);

    Tensor f = Tensor::Fill({}, 2.5f, DataType::FLOAT64);
    EXPECT_EQ(f.dtype(), DataType::FLOAT64);
    EXPECT_DOUBLE_EQ(f.at<double>({}), 2.5);
}



// ---------- 33. FR-1 空张量：形状、存储与算子语义 ----------
// numel == 0 是合法张量：shape/dtype/device 照常，data() 不为空。
TEST(Empty, ConstructionAndMetadata) {
    Tensor e = Tensor::Zeros({2, 0, 3}, DataType::FLOAT32);
    EXPECT_EQ(e.dims(), 3u);
    EXPECT_EQ(e.numel(), 0u);
    EXPECT_NE(e.data(), nullptr);
    EXPECT_EQ(e.shape(1), 0);
    EXPECT_TRUE(e.is_contiguous());
    // 行主序步长：空维把外层步长压成 0（{2,0,3} 的内侧连续块长度为 0）。
    EXPECT_EQ(e.strides(0), 0);
    EXPECT_EQ(e.strides(1), 3);
    EXPECT_EQ(e.strides(2), 1);

    Tensor z = Tensor::Zeros({0}, DataType::INT32);
    EXPECT_EQ(z.numel(), 0u);
    EXPECT_EQ(z.dims(), 1u);
    EXPECT_EQ(z.dtype(), DataType::INT32);
    EXPECT_NE(z.data(), nullptr);
    EXPECT_EQ(z.clone().numel(), 0u);

    // 负维度仍然拒绝；0-dim 仍然是 1 个元素。
    EXPECT_THROW(Tensor::Zeros({-1}, DataType::FLOAT32), std::exception);
    EXPECT_EQ(Tensor::Zeros({}, DataType::FLOAT32).numel(), 1u);

    // 拷贝 / to_host / 类型转换都保持空。
    Tensor h = e.clone();
    h.to_host();
    EXPECT_EQ(h.numel(), 0u);
    const Tensor& ce = e;
    Tensor cast = ops::Typecast(ce, DataType::INT64);
    EXPECT_EQ(cast.dtype(), DataType::INT64);
    EXPECT_EQ(cast.numel(), 0u);
    EXPECT_EQ(cast.dims(), 3u);

    EXPECT_NO_THROW(ops::println(e));
}

TEST(Empty, ElementwiseAndBroadcast) {
    Tensor e = Tensor::Zeros({3, 0}, DataType::FLOAT32);
    EXPECT_EQ(ops::Exp(e).numel(), 0u);
    EXPECT_EQ(ops::Exp(e).shape(0), 3);
    const Tensor& ce = e;
    EXPECT_EQ(ops::Abs(ce).numel(), 0u);

    // 整型输入的一元算子提升到 FLOAT32，即便没有元素。
    Tensor ei = Tensor::Zeros({2, 0}, DataType::INT32);
    const Tensor& cei = ei;
    EXPECT_EQ(ops::Sin(cei).dtype(), DataType::FLOAT32);
    EXPECT_EQ(ops::Relu(cei).dtype(), DataType::INT32);

    // 标量运算：出参为空、结果 dtype 按弱标量提升。
    EXPECT_EQ((e + 2.0f).numel(), 0u);
    EXPECT_EQ((e * 2.0f).shape(1), 0);
    Tensor inplace = e.clone();
    ops::Add(inplace, 2.0f);
    ops::Mul(inplace, 3.0f);
    EXPECT_EQ(inplace.numel(), 0u);

    // 0 与 1 广播成 0；0 与非 1 维不可广播。
    Tensor row = Tensor::Zeros({0}, DataType::FLOAT32);
    Tensor col = Tensor::Ones({5, 1}, DataType::FLOAT32);
    Tensor bc = row * col;
    EXPECT_EQ(bc.numel(), 0u);
    EXPECT_EQ(bc.dims(), 2u);
    EXPECT_EQ(bc.shape(0), 5);
    EXPECT_EQ(bc.shape(1), 0);
    Tensor bad = Tensor::Ones({4}, DataType::FLOAT32);
    EXPECT_THROW(row + bad, std::exception);

    // 0-dim 标量与空张量。
    Tensor scalar = Tensor::Fill({}, 2.0f, DataType::FLOAT32);
    Tensor shifted = row * scalar;
    EXPECT_EQ(shifted.numel(), 0u);

    // 比较 -> 空 BOOL；where 对空输入返回空。
    Tensor cmp = ops::Greater(e, 0.0f);
    EXPECT_EQ(cmp.dtype(), DataType::BOOL);
    EXPECT_EQ(cmp.numel(), 0u);
    EXPECT_EQ(ops::Where(cmp, e, e).numel(), 0u);
}

TEST(Empty, Reductions) {
    Tensor z = Tensor::Zeros({2, 0}, DataType::FLOAT32);

    // 全归约仍是 0-dim，取值是折叠的恒等元。
    Tensor s = ops::Sum(z);
    EXPECT_EQ(s.dims(), 0u);
    EXPECT_FLOAT_EQ(s.item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(ops::Prod(z).item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(ops::Norm(z, 2.0f).item<float>(), 0.0f);

    // mean/var/std 是 0/0：定义为 NaN。
    EXPECT_TRUE(std::isnan(ops::Mean(z).item<float>()));
    EXPECT_TRUE(std::isnan(ops::Var(z).item<float>()));
    EXPECT_TRUE(std::isnan(ops::Std(z).item<float>()));

    // Min/Max 没有恒等元，明确报错。
    EXPECT_THROW(ops::Min(z), std::exception);
    EXPECT_THROW(ops::Max(z), std::exception);

    // 空整型的 Sum 取归约 dtype。
    Tensor i = Tensor::Zeros({0}, DataType::INT16);
    Tensor isum = ops::Sum(i);
    EXPECT_EQ(isum.dtype(), DataType::INT64);
    EXPECT_EQ(isum.item<std::int64_t>(), 0);

    // 归约轴长度为 0：沿该轴得到恒等元。
    Tensor a = Tensor::Zeros({2, 0}, DataType::FLOAT32);
    Tensor s_axis = ops::Sum(a, 1);
    EXPECT_EQ(s_axis.dims(), 1u);
    EXPECT_EQ(s_axis.shape(0), 2);
    EXPECT_FLOAT_EQ(s_axis.at<float>({0}), 0.0f);
    EXPECT_FLOAT_EQ(s_axis.at<float>({1}), 0.0f);
    EXPECT_THROW(ops::Min(a, 1), std::exception);
    EXPECT_THROW(ops::Argmax(a, 1), std::exception);

    // 非归约轴长度为 0：结果是空张量，形状照常。
    Tensor b = Tensor::Zeros({0, 3}, DataType::FLOAT32);
    Tensor e_axis = ops::Sum(b, 1);
    EXPECT_EQ(e_axis.dims(), 1u);
    EXPECT_EQ(e_axis.numel(), 0u);
    EXPECT_EQ(e_axis.shape(0), 0);
    Tensor arg = ops::Argmax(b, 1);
    EXPECT_EQ(arg.dtype(), DataType::INT64);
    EXPECT_EQ(arg.numel(), 0u);

    // 归约轴为 0 但其他轴非 0：沿另一轴归约得到恒等元。
    Tensor s0 = ops::Sum(b, 0);
    EXPECT_EQ(s0.shape(0), 3);
    EXPECT_FLOAT_EQ(s0.at<float>({2}), 0.0f);
    EXPECT_TRUE(std::isnan(ops::Mean(b, 0).at<float>({0})));

    // keepdim 保留长度 1 的维。
    Tensor keep = ops::Sum(a, 1, true);
    EXPECT_EQ(keep.dims(), 2u);
    EXPECT_EQ(keep.shape(1), 1);
    EXPECT_FLOAT_EQ(keep.at<float>({0, 0}), 0.0f);
}

TEST(Empty, StructuralAndSupport) {
    Tensor z = Tensor::Zeros({2, 0}, DataType::FLOAT32);
    EXPECT_EQ(ops::Cumsum(z, 1).numel(), 0u);
    EXPECT_EQ(ops::Cumsum(z, 1).shape(0), 2);
    EXPECT_TRUE(ops::All(z, 0.0f));   // 空集合上 All 为真
    EXPECT_FALSE(ops::Any(z, 0.0f));  // Any 为假
    EXPECT_EQ(ops::CountNonzero(z), 0u);

    // slice / transpose / softmax 保持空。
    Tensor empty0 = Tensor::Zeros({0, 3}, DataType::FLOAT32);
    Tensor sl = ops::Slice(empty0, {{0, 0}, {1, 3}});
    EXPECT_EQ(sl.shape(0), 0);
    EXPECT_EQ(sl.shape(1), 2);
    Tensor tr = ops::Transpose(empty0, {1, 0});
    EXPECT_EQ(tr.shape(0), 3);
    EXPECT_EQ(tr.shape(1), 0);
    EXPECT_EQ(ops::Softmax(empty0, 1).numel(), 0u);
    EXPECT_EQ(ops::LogSoftmax(empty0, 1).numel(), 0u);

    // LayerNorm/RmsNorm：归一化块为空 -> 输出为空。
    Tensor w = Tensor::Zeros({0}, DataType::FLOAT32);
    Tensor bias = Tensor::Zeros({0}, DataType::FLOAT32);
    Tensor ln = ops::LayerNorm(z, w, bias, 1, 1e-5f);
    EXPECT_EQ(ln.numel(), 0u);
    EXPECT_EQ(ln.shape(0), 2);
    EXPECT_EQ(ops::RmsNorm(z, w, 1, 1e-5f).numel(), 0u);

    // 空内积 = 0；输出本来就没有元素时直接为空。
    Tensor m1 = Tensor::Zeros({2, 0}, DataType::FLOAT32);
    Tensor m2 = Tensor::Zeros({0, 3}, DataType::FLOAT32);
    Tensor mm = ops::MatMul(m1, m2);
    EXPECT_EQ(mm.dims(), 2u);
    EXPECT_EQ(mm.shape(0), 2);
    EXPECT_EQ(mm.shape(1), 3);
    EXPECT_FLOAT_EQ(mm.at<float>({1, 2}), 0.0f);
    Tensor m3 = Tensor::Zeros({0, 2}, DataType::FLOAT32);
    Tensor m4 = Tensor::Zeros({2, 3}, DataType::FLOAT32);
    EXPECT_EQ(ops::MatMul(m3, m4).numel(), 0u);
    EXPECT_EQ(ops::MatMul(m3, m4).shape(1), 3);

    // 3D 批内空内积。
    Tensor b1 = Tensor::Zeros({2, 3, 0}, DataType::FLOAT32);
    Tensor b2 = Tensor::Zeros({2, 0, 4}, DataType::FLOAT32);
    Tensor bm = ops::MatMul(b1, b2);
    EXPECT_EQ(bm.dims(), 3u);
    EXPECT_EQ(bm.shape(0), 2);
    EXPECT_EQ(bm.shape(2), 4);
    EXPECT_FLOAT_EQ(bm.at<float>({1, 2, 3}), 0.0f);

    // concat：空输入不贡献元素，但结果形状与取值正确。
    std::vector<float> vals{1, 2, 3, 4, 5, 6};
    Tensor full(vals, {2, 3});
    Tensor cat = ops::Concat({empty0, full}, 0);
    EXPECT_EQ(cat.shape(0), 2);
    EXPECT_EQ(cat.at<float>({0, 2}), 3.0f);
    Tensor cat2 = ops::Concat({full, empty0}, 0);
    EXPECT_EQ(cat2.at<float>({1, 0}), 4.0f);
    EXPECT_EQ(ops::Concat({empty0, empty0}, 0).shape(0), 0);

    // 空索引 -> 空输出。gather 的索引与输入同秩（与 PyTorch 一致），所以它的空
    // 索引要保留被索引以外的维度：{0, 3} 拿到的是形状 {0, 3} 的空结果。
    Tensor idx = Tensor::Zeros({0}, DataType::INT64);
    EXPECT_EQ(ops::IndexSelect(full, 0, idx).numel(), 0u);
    EXPECT_EQ(ops::IndexSelect(full, 0, idx).shape(1), 3);
    Tensor gidx = Tensor::Zeros({0, 3}, DataType::INT64);
    EXPECT_EQ(ops::Gather(full, 0, gidx).numel(), 0u);
    EXPECT_EQ(ops::Gather(full, 0, gidx).shape(1), 3);
}

// 设备 lane：空张量在 GPU 上同样可分配、可归约，并且结果留在原设备上。
TEST_F(DeviceOps, EmptyTensors) {
    Tensor e = Tensor::Zeros({2, 0}, DataType::FLOAT32, dev);
    EXPECT_EQ(e.device(), dev);
    EXPECT_EQ(e.numel(), 0u);
    EXPECT_NE(e.data(), nullptr);

    Tensor s = ops::Sum(e);
    EXPECT_EQ(s.device(), dev);
    EXPECT_EQ(s.dims(), 0u);
    EXPECT_FLOAT_EQ(s.item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(ops::Prod(e).item<float>(), 1.0f);
    EXPECT_TRUE(std::isnan(ops::Mean(e).item<float>()));

    auto axis_host = host_values<float>(ops::Sum(e, 1));
    EXPECT_EQ(axis_host.size(), 2u);
    EXPECT_FLOAT_EQ(axis_host[1], 0.0f);

    Tensor x = ops::Exp(e);
    EXPECT_EQ(x.device(), dev);
    EXPECT_EQ(x.numel(), 0u);

    Tensor cmp = ops::Greater(e, 0.0f);
    EXPECT_EQ(cmp.device(), dev);
    EXPECT_EQ(cmp.dtype(), DataType::BOOL);
    EXPECT_EQ(cmp.numel(), 0u);

    Tensor m1 = Tensor::Zeros({2, 0}, DataType::FLOAT32, dev);
    Tensor m2 = Tensor::Zeros({0, 3}, DataType::FLOAT32, dev);
    Tensor mm = ops::MatMul(m1, m2);
    EXPECT_EQ(mm.device(), dev);
    expect_close(host_values<float>(mm), {0, 0, 0, 0, 0, 0}, "matmul-empty-inner");

    Tensor empty0 = Tensor::Zeros({0, 3}, DataType::FLOAT32, dev);
    std::vector<float> vals{1, 2, 3, 4, 5, 6};
    Tensor full_cpu(vals, {2, 3});
    Tensor full = moved_to(full_cpu, dev);
    auto cat = ops::Concat({empty0, full}, 0);
    EXPECT_EQ(cat.device(), dev);
    expect_close(host_values<float>(cat), vals, "concat-empty-first");

    EXPECT_EQ(ops::Slice(empty0, {{0, 0}, {1, 3}}).numel(), 0u);
    EXPECT_EQ(ops::Transpose(empty0, {1, 0}).shape(0), 3);
    EXPECT_EQ(ops::Softmax(empty0, 1).numel(), 0u);
}


// ---------- 34. 打印：0-dim 张量必须显示它唯一的值 ----------
TEST(Print, RankZeroShowsScalarValue) {
    Tensor a = Tensor::Fill({4, 5}, 1.0f);
    Tensor total = ops::Sum(a);  // 20，0-dim
    testing::internal::CaptureStdout();
    ops::println(total);
    const std::string out = testing::internal::GetCapturedStdout();
    EXPECT_NE(out.find("20.000"), std::string::npos) << out;
    // 0-dim 是标量：不带方括号，避免看起来像长度为 1 的一维张量。
    EXPECT_EQ(out.find("["), std::string::npos) << out;
    EXPECT_NE(out.find("Tensor shape: ()"), std::string::npos) << out;

    // 形状 (1) 的一维张量仍按数组格式打印，两者可区分。
    Tensor one = Tensor::Fill({1}, 7.0f);
    testing::internal::CaptureStdout();
    ops::println(one);
    const std::string one_out = testing::internal::GetCapturedStdout();
    EXPECT_NE(one_out.find("[7.000]"), std::string::npos) << one_out;
    EXPECT_NE(one_out.find("Tensor shape: (1)"), std::string::npos) << one_out;

    // 整型 0-dim 按整数格式打印，不出现小数点。
    // 用 const 引用锁住按值返回的重载，避免选到就地修改的 void 版本。
    const Tensor iv = Tensor::Fill({1}, 7.0f);
    const Tensor& civ = iv;
    Tensor isum = ops::Sum(ops::Typecast(civ, DataType::INT32));
    testing::internal::CaptureStdout();
    ops::println(isum);
    const std::string int_out = testing::internal::GetCapturedStdout();
    EXPECT_NE(int_out.find("7\nTensor shape: ()"), std::string::npos) << int_out;
    EXPECT_EQ(int_out.find("7."), std::string::npos) << int_out;  // 整型不带小数点
}

