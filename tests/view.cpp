#include "test_support.hpp"

// 视图、offset 与引用语义：slice/permute/narrow/select/expand、写穿与非连续拒绝。
// ---------- 1~3. 带 offset 的连续切片：读 / out-of-place / in-place ----------
TEST(View, SliceReadAndInPlaceArithmetic) {
    Tensor a = matrix_3x3();
    auto v = a.slice({{1, 3}, {0, 3}});  // [[4,5,6],[7,8,9]]，偏移=3
    EXPECT_EQ(v.shape(0), 2);
    EXPECT_EQ(v.shape(1), 3);
    EXPECT_FLOAT_EQ(v.at<float>({0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(v.at<float>({0, 2}), 6.0f);
    EXPECT_FLOAT_EQ(v.at<float>({1, 0}), 7.0f);
    EXPECT_FLOAT_EQ(v.at<float>({1, 2}), 9.0f);

    // 视图上的 out-of-place 算术（不污染原张量）
    auto ones23 = Tensor::Ones({2, 3}, DataType::FLOAT32);
    auto r = v + ones23;  // [[5,6,7],[8,9,10]]
    EXPECT_FLOAT_EQ(r.at<float>({0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(r.at<float>({1, 2}), 10.0f);
    EXPECT_FLOAT_EQ(a.at<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(a.at<float>({2, 2}), 9.0f);

    // in-place 标量算子在连续 offset 视图上：写穿到原张量
    ops::Add(v, 1.0f);  // v -> [[5,6,7],[8,9,10]]，同时写穿 a 的后两行
    EXPECT_FLOAT_EQ(v.at<float>({0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(a.at<float>({0, 0}), 1.0f);  // 视图外的行不动
    EXPECT_FLOAT_EQ(a.at<float>({1, 0}), 5.0f);
    EXPECT_FLOAT_EQ(a.at<float>({2, 2}), 10.0f);
}


// ---------- 4~6. 非连续视图（转置）：读 / 算术 / reduction / softmax ----------
TEST(View, TransposedView) {
    Tensor x = matrix_3x3();
    auto xt = x.t();  // [[1,4,7],[2,5,8],[3,6,9]]，strides={1,3} 非连续
    EXPECT_FLOAT_EQ(xt.at<float>({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(xt.at<float>({2, 0}), 3.0f);

    auto ones33 = Tensor::Ones({3, 3}, DataType::FLOAT32);
    auto tr = xt + ones33;  // [[2,5,8],[3,6,9],[4,7,10]]
    EXPECT_FLOAT_EQ(tr.at<float>({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(tr.at<float>({0, 1}), 5.0f);
    EXPECT_FLOAT_EQ(tr.at<float>({2, 2}), 10.0f);
    EXPECT_FLOAT_EQ(x.at<float>({0, 0}), 1.0f);  // 原张量不受影响

    // 非连续视图上的 reduction
    EXPECT_NEAR(ops::Sum(xt).item<float>(), 45.0f, kEps);
    EXPECT_NEAR(ops::Max(xt).item<float>(), 9.0f, kEps);
    EXPECT_NEAR(ops::Mean(xt).item<float>(), 5.0f, kEps);

    // 非连续视图上的 softmax：每行和为 1
    auto sm = ops::Softmax(xt, -1);
    for (int i = 0; i < 3; ++i) {
        float s = 0.0f;
        for (int j = 0; j < 3; ++j) s += sm.at<float>({i, j});
        EXPECT_NEAR(s, 1.0f, kEps) << "row " << i;
    }
    // 第一行 = softmax([1,4,7])，逐值核对
    float e1 = std::exp(1.0f), e4 = std::exp(4.0f), e7 = std::exp(7.0f);
    float denom = e1 + e4 + e7;
    EXPECT_NEAR(sm.at<float>({0, 0}), e1 / denom, kEps);
    EXPECT_NEAR(sm.at<float>({0, 2}), e7 / denom, kEps);
}


// ---------- 7. 引用语义：拷贝共享数据（对齐 NumPy/PyTorch） ----------
TEST(Tensor, ReferenceSemantics) {
    std::vector<float> ref_v{1, 2, 3};
    Tensor rt(ref_v, {3});
    Tensor rc = rt;  // O(1) 共享句柄
    EXPECT_FLOAT_EQ(rc.at<float>({0}), 1.0f);
    ops::Add(rc, 10.0f);  // 写穿共享 buffer
    EXPECT_FLOAT_EQ(rc.at<float>({0}), 11.0f);
    EXPECT_FLOAT_EQ(rt.at<float>({0}), 11.0f);  // 原句柄同步可见

    auto rd = rt.clone();  // clone = 显式深拷贝
    ops::Add(rd, 100.0f);
    EXPECT_FLOAT_EQ(rd.at<float>({0}), 111.0f);
    EXPECT_FLOAT_EQ(rt.at<float>({0}), 11.0f);  // clone 不影响原

    Tensor rm = rt;  // meta 各自独立
    Tensor rv = rm.reshape({1, 3});  // reshape 返回视图，不修改 rm（FR-4）
    EXPECT_EQ(rv.shape().size(), 2u);
    EXPECT_EQ(rm.shape().size(), 1u);
    EXPECT_EQ(rt.shape().size(), 1u);
    EXPECT_FLOAT_EQ(rv.at<float>({0, 0}), rm.at<float>({0}));  // 共享存储

    Tensor a = matrix_3x3();
    auto v = a.slice({{1, 3}, {0, 3}});
    Tensor cv = v;  // 视图的拷贝同样共享
    EXPECT_FLOAT_EQ(cv.at<float>({0, 0}), v.at<float>({0, 0}));
}


// ---------- 8. 视图的 clone 与 typecast ----------
TEST(View, CloneAndTypecast) {
    Tensor a = matrix_3x3();
    auto v = a.slice({{1, 3}, {0, 3}});
    ops::Add(v, 1.0f);  // v -> [[5,6,7],[8,9,10]]

    auto vc = v.clone();
    EXPECT_FLOAT_EQ(vc.at<float>({0, 0}), 5.0f);
    EXPECT_TRUE(vc.is_contiguous());

    const Tensor& v_const = v;  // 非 const 左值会重载到 in-place 版本
    auto v64 = ops::Typecast(v_const, DataType::FLOAT64);
    EXPECT_EQ(v64.dtype(), DataType::FLOAT64);
    EXPECT_DOUBLE_EQ(v64.at<double>({0, 0}), 5.0);
    EXPECT_DOUBLE_EQ(v64.at<double>({1, 2}), 10.0);
}


// ---------- 9. permute（3D 非连续视图）算术 ----------
TEST(View, Permute3D) {
    std::vector<float> bv;
    for (int i = 0; i < 24; ++i) bv.push_back(static_cast<float>(i));
    Tensor b3(bv, {2, 3, 4});  // b3[0,1,2] = 6

    auto bp = b3.permute({2, 0, 1});  // 形状 {4,2,3}
    EXPECT_EQ(bp.shape(0), 4);
    EXPECT_EQ(bp.shape(1), 2);
    EXPECT_EQ(bp.shape(2), 3);
    EXPECT_FLOAT_EQ(bp.at<float>({2, 0, 1}), 6.0f);  // bp[i,j,k] = b3[j,k,i]

    auto bp2 = bp * 2.0f;
    EXPECT_FLOAT_EQ(bp2.at<float>({2, 0, 1}), 12.0f);
    EXPECT_NEAR(ops::Sum(bp).item<float>(), ops::Sum(b3).item<float>(), kEps);
}


// ---------- 10. in-place 写入非连续视图：必须拒绝 ----------
TEST(View, InPlaceRejectsNonContiguous) {
    Tensor x = matrix_3x3();
    auto xt = x.t();

    EXPECT_THROW(ops::Add(xt, 1.0f), std::exception);
    EXPECT_THROW(ops::Relu(xt), std::exception);
    EXPECT_THROW(ops::Fill(xt, 5.0f), std::exception);
    EXPECT_THROW(ops::Transpose(xt), std::exception);

    // 连续 offset 视图允许 in-place，普通张量不受影响
    Tensor ok_t = matrix_3x3();
    ops::Relu(ok_t);
    EXPECT_FLOAT_EQ(ok_t.at<float>({0, 0}), 1.0f);
}


// ---------- 11. dst 版本二元算子在视图上 ----------
TEST(View, DstBinaryOnTransposedView) {
    Tensor x = matrix_3x3();
    auto xt = x.t();
    auto ones33 = Tensor::Ones({3, 3}, DataType::FLOAT32);

    Tensor dst({3, 3}, DataType::FLOAT32);
    ops::Add(xt, ones33, dst);
    EXPECT_FLOAT_EQ(dst.at<float>({0, 1}), 5.0f);
    EXPECT_FLOAT_EQ(dst.at<float>({2, 2}), 10.0f);
}


// ---------- 19. 元数据布局（DimArray） ----------
TEST(Metadata, DimArray) {
    // unsqueeze 返回视图，strides 正确重算（FR-4）
    Tensor uq = Tensor::Ones({2, 3}, DataType::FLOAT32);
    Tensor uqv = uq.unsqueeze(0);
    EXPECT_EQ(uq.shape().size(), 2u);
    EXPECT_EQ(uqv.shape().size(), 3u);
    EXPECT_EQ(uqv.strides(0), 6);
    EXPECT_FLOAT_EQ(uqv.at<float>({0, 1, 2}), 1.0f);

    // 维度数超过 8 拒绝
    EXPECT_THROW((Tensor({1, 1, 1, 1, 1, 1, 1, 1, 1}, DataType::FLOAT32)), std::exception);

    // span 返回 + range-for
    std::int64_t prod = 1;
    for (std::int64_t d : uq.shape()) prod *= d;
    EXPECT_EQ(prod, 6);
}

