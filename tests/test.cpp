#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <vector>

#include "interface/tensorvia.hpp"

namespace {

constexpr float kEps = 1e-5f;

// Recreate the [[1,2,3],[4,5,6],[7,8,9]] fixture shared by several view tests.
Tensor matrix_3x3() {
    static std::vector<float> values{1, 2, 3, 4, 5, 6, 7, 8, 9};
    return Tensor(values, {3, 3});
}

}  // namespace

// ---------- 原有冒烟 ----------
TEST(Tensorvia, BasicArithmetic) {
    auto sa = Tensor::Ones({2, 3}, DataType::FLOAT32);
    auto sb = Tensor::Fill({2, 3}, 2.0f, DataType::FLOAT32);
    auto sc = sa + sb;
    EXPECT_FLOAT_EQ(sc.at<float>({0, 0}), 3.0f);
    auto sd = ops::MatMul(sa.view({3, 2}), sb.view({2, 3}));
    EXPECT_EQ(sd.shape(0), 3);
    EXPECT_EQ(sd.shape(1), 3);
}

// ---------- 1~3. 带 offset 的连续切片：读 / out-of-place / in-place ----------
TEST(View, SliceReadAndInPlaceArithmetic) {
    Tensor a = matrix_3x3();
    auto v = a.slice({{1, 3}, {0, 3}});  // [[4,5,6],[7,8,9]] offset=3
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
    EXPECT_NEAR(ops::Sum(xt), 45.0f, kEps);
    EXPECT_NEAR(ops::Max(xt), 9.0f, kEps);
    EXPECT_NEAR(ops::Mean(xt), 5.0f, kEps);

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
    rm.reshape({1, 3});
    EXPECT_EQ(rm.shape().size(), 2u);
    EXPECT_EQ(rt.shape().size(), 1u);

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

    auto bp = b3.permute({2, 0, 1});  // shape {4,2,3}
    EXPECT_EQ(bp.shape(0), 4);
    EXPECT_EQ(bp.shape(1), 2);
    EXPECT_EQ(bp.shape(2), 3);
    EXPECT_FLOAT_EQ(bp.at<float>({2, 0, 1}), 6.0f);  // bp[i,j,k] = b3[j,k,i]

    auto bp2 = bp * 2.0f;
    EXPECT_FLOAT_EQ(bp2.at<float>({2, 0, 1}), 12.0f);
    EXPECT_NEAR(ops::Sum(bp), ops::Sum(b3), kEps);
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

// ---------- 12. 混合 dtype 慢路径 ----------
TEST(DType, MixedPromotion) {
    std::vector<float> mf_v{1, 2, 3, 4};
    std::vector<double> md_v{0.5, 0.5, 0.5, 0.5};
    Tensor mf(mf_v, {4});  // FLOAT32
    Tensor md(md_v, {4});  // FLOAT64

    auto mr = mf + md;  // F32 + F64 -> F64 = [1.5, 2.5, 3.5, 4.5]
    EXPECT_EQ(mr.dtype(), DataType::FLOAT64);
    EXPECT_DOUBLE_EQ(mr.at<double>({0}), 1.5);
    EXPECT_DOUBLE_EQ(mr.at<double>({3}), 4.5);

    auto mr2 = md + mf;  // 交换操作数顺序（另一侧提升）
    EXPECT_DOUBLE_EQ(mr2.at<double>({0}), 1.5);
    EXPECT_DOUBLE_EQ(mr2.at<double>({3}), 4.5);
    EXPECT_DOUBLE_EQ((mf - md).at<double>({0}), 0.5);
    EXPECT_DOUBLE_EQ((mf * md).at<double>({1}), 1.0);
    EXPECT_DOUBLE_EQ((mf / md).at<double>({3}), 8.0);

    std::vector<std::int32_t> mi_v{1, 2, 3, 4};
    Tensor mi(mi_v, {4});  // INT32
    auto ir = mi + mf;     // INT32 + F32 -> F32 = [2, 4, 6, 8]
    EXPECT_EQ(ir.dtype(), DataType::FLOAT32);
    EXPECT_FLOAT_EQ(ir.at<float>({0}), 2.0f);
    EXPECT_FLOAT_EQ(ir.at<float>({3}), 8.0f);

    // 视图（f32）+ 混合 dtype + dst 变体组合
    Tensor m3 = matrix_3x3();
    auto row = m3.slice({{1, 2}, {0, 3}});  // [[4,5,6]] f32 带 offset
    Tensor ones_f64 = Tensor::Ones({1, 3}, DataType::FLOAT64);
    Tensor d_out({1, 3}, DataType::FLOAT64);
    ops::Add(row, ones_f64, d_out);
    EXPECT_DOUBLE_EQ(d_out.at<double>({0, 0}), 5.0);
    EXPECT_DOUBLE_EQ(d_out.at<double>({0, 2}), 7.0);

    // fp16 matmul：mul 内部先把半精度提升到 F32
    std::vector<float16> hf_v{float16(1.0f), float16(2.0f), float16(3.0f), float16(4.0f)};
    Tensor hf(hf_v, {2, 2});
    auto hr = ops::MatMul(hf, hf);  // [[7,10],[15,22]]，结果 FLOAT32
    EXPECT_EQ(hr.dtype(), DataType::FLOAT32);
    EXPECT_FLOAT_EQ(hr.at<float>({0, 0}), 7.0f);
    EXPECT_FLOAT_EQ(hr.at<float>({0, 1}), 10.0f);
    EXPECT_FLOAT_EQ(hr.at<float>({1, 1}), 22.0f);
}

// ---------- 13. matmul micro-kernel：naive 三重循环对照 ----------
TEST(MatMul, MatchesNaiveGemm) {
    // 覆盖满 8x8 块、rows/cols 不被 8 整除的余块、3D batch
    auto naive_gemm = [](Tensor A, Tensor B) {
        const bool is3d = A.shape().size() == 3;
        const std::int64_t batch = is3d ? A.shape(0) : 1;
        const std::int64_t rows = is3d ? A.shape(1) : A.shape(0);
        const std::int64_t inner = is3d ? A.shape(2) : A.shape(1);
        const std::int64_t cols = is3d ? B.shape(2) : B.shape(1);
        std::vector<float> out(std::size_t(batch * rows * cols), 0.0f);
        for (std::int64_t b = 0; b < batch; ++b)
            for (std::int64_t i = 0; i < rows; ++i)
                for (std::int64_t j = 0; j < cols; ++j) {
                    float s = 0.0f;
                    for (std::int64_t k = 0; k < inner; ++k) {
                        const float av = is3d ? A.at<float>({b, i, k}) : A.at<float>({i, k});
                        const float bv = is3d ? B.at<float>({b, k, j}) : B.at<float>({k, j});
                        s += av * bv;
                    }
                    out[std::size_t(b * rows * cols + i * cols + j)] = s;
                }
        return out;
    };

    auto check_matmul = [&](const Tensor& A, const Tensor& B, const char* label) {
        SCOPED_TRACE(label);
        Tensor got = ops::MatMul(A, B);
        std::vector<float> want = naive_gemm(A, B);
        ASSERT_EQ(got.numel(), want.size());
        const float* gd = static_cast<const float*>(got.data());
        float max_err = 0.0f;
        for (std::size_t i = 0; i < want.size(); ++i)
            max_err = std::max(max_err, std::abs(gd[i] - want[i]));
        EXPECT_NEAR(max_err, 0.0f, kEps);
    };

    check_matmul(Tensor::Random({16, 16}, 0.0f, 1.0f, DataType::FLOAT32),
                 Tensor::Random({16, 16}, 0.0f, 1.0f, DataType::FLOAT32),
                 "full 8x8 blocks");
    check_matmul(Tensor::Random({5, 3}, 0.0f, 1.0f, DataType::FLOAT32),
                 Tensor::Random({3, 7}, 0.0f, 1.0f, DataType::FLOAT32),
                 "remainder blocks");
    check_matmul(Tensor::Random({17, 9}, 0.0f, 1.0f, DataType::FLOAT32),
                 Tensor::Random({9, 5}, 0.0f, 1.0f, DataType::FLOAT32),
                 "mixed boundary");
    check_matmul(Tensor::Random({64, 64}, 0.0f, 1.0f, DataType::FLOAT32),
                 Tensor::Random({64, 300}, 0.0f, 1.0f, DataType::FLOAT32),
                 "K blocking boundary");
    check_matmul(Tensor::Random({2, 5, 6}, 0.0f, 1.0f, DataType::FLOAT32),
                 Tensor::Random({2, 6, 4}, 0.0f, 1.0f, DataType::FLOAT32),
                 "3D batch");
}

// ---------- 14. argmax/argmin 返回 INT32 索引 ----------
TEST(Reduce, ArgmaxArgminIndexType) {
    std::vector<float> am_v{3, 1, 2,
                            9, 7, 8,
                            4, 6, 5};
    Tensor am(am_v, {3, 3});

    auto amax1 = ops::Argmax(am, 1);
    EXPECT_EQ(amax1.dtype(), DataType::INT32);
    EXPECT_EQ(amax1.at<std::int32_t>({0}), 0);  // 行 [3,1,2] 最大在 0
    EXPECT_EQ(amax1.at<std::int32_t>({1}), 0);  // 行 [9,7,8] 最大在 0
    EXPECT_EQ(amax1.at<std::int32_t>({2}), 1);  // 行 [4,6,5] 最大在 1

    auto amin1 = ops::Argmin(am, 1);
    EXPECT_EQ(amin1.dtype(), DataType::INT32);
    EXPECT_EQ(amin1.at<std::int32_t>({0}), 1);
    EXPECT_EQ(amin1.at<std::int32_t>({1}), 1);
    EXPECT_EQ(amin1.at<std::int32_t>({2}), 0);

    auto amax0 = ops::Argmax(am, 0);
    EXPECT_EQ(amax0.at<std::int32_t>({0}), 1);  // 列 [3,9,4] 最大在 1
    EXPECT_EQ(amax0.at<std::int32_t>({1}), 1);  // 列 [1,7,6] 最大在 1
    EXPECT_EQ(amax0.at<std::int32_t>({2}), 1);  // 列 [2,8,5] 最大在 1

    // int8 溢出回归：索引 150 超出 int8 范围
    std::vector<std::int8_t> big_v(200, std::int8_t{-1});
    big_v[150] = 0;
    Tensor big(big_v, {200});
    auto bimax = ops::Argmax(big, 0);
    EXPECT_EQ(bimax.dtype(), DataType::INT32);
    EXPECT_EQ(bimax.shape(0), 1);
    EXPECT_EQ(bimax.at<std::int32_t>({0}), 150);
}

// ---------- 15. 1D 带轴 reduce 全家族 ----------
TEST(Reduce, OneDimAxis) {
    std::vector<float> r1d_v{1, 2, 3, 4};
    Tensor r1d(r1d_v, {4});
    EXPECT_NEAR(ops::Sum(r1d, 0).at<float>({0}), 10.0f, kEps);
    EXPECT_NEAR(ops::Mean(r1d, 0).at<float>({0}), 2.5f, kEps);
    EXPECT_NEAR(ops::Min(r1d, 0).at<float>({0}), 1.0f, kEps);
    EXPECT_NEAR(ops::Max(r1d, 0).at<float>({0}), 4.0f, kEps);
    EXPECT_EQ(ops::Argmax(r1d, 0).at<std::int32_t>({0}), 3);
    EXPECT_EQ(ops::Argmin(r1d, 0).at<std::int32_t>({0}), 0);

    // 视图组合：转置视图上的 argmax（am.t() = [[3,9,4],[1,7,6],[2,8,5]]）
    std::vector<float> am_v{3, 1, 2, 9, 7, 8, 4, 6, 5};
    Tensor am(am_v, {3, 3});
    auto tamax = ops::Argmax(am.t(), 0);
    EXPECT_EQ(tamax.at<std::int32_t>({0}), 0);  // 列 [3,1,2] 最大在 0
    EXPECT_EQ(tamax.at<std::int32_t>({1}), 0);  // 列 [9,7,8] 最大在 0
}

// ---------- 16. 广播（NumPy 规则） ----------
TEST(Broadcast, Arithmetic) {
    // (3,1) + (1,4) -> (3,4)
    std::vector<float> ca_v{1, 2, 3};
    std::vector<float> cb_v{10, 20, 30, 40};
    Tensor ca(ca_v, {3, 1});
    Tensor cb(cb_v, {1, 4});
    auto cab = ca + cb;
    EXPECT_EQ(cab.shape(0), 3);
    EXPECT_EQ(cab.shape(1), 4);
    EXPECT_FLOAT_EQ(cab.at<float>({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(cab.at<float>({0, 3}), 41.0f);
    EXPECT_FLOAT_EQ(cab.at<float>({2, 0}), 13.0f);
    EXPECT_FLOAT_EQ(cab.at<float>({2, 3}), 43.0f);

    // (4,) + (3,4)：行向量广播（连续段快路径）
    std::vector<float> row_v{1, 2, 3, 4};
    Tensor brow(row_v, {4});
    std::vector<float> mat_v;
    for (int i = 0; i < 12; ++i) mat_v.push_back(float(i));
    Tensor bmat(mat_v, {3, 4});
    auto mab = brow + bmat;  // out[i][j] = row[j] + mat[i][j]
    EXPECT_EQ(mab.shape(0), 3);
    EXPECT_EQ(mab.shape(1), 4);
    EXPECT_FLOAT_EQ(mab.at<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(mab.at<float>({1, 2}), 9.0f);
    EXPECT_FLOAT_EQ(mab.at<float>({2, 3}), 15.0f);

    // (3,1) + (3,4)：列向量广播
    Tensor col(ca_v, {3, 1});
    auto mcol = col + bmat;  // out[i][j] = col[i] + mat[i][j]
    EXPECT_FLOAT_EQ(mcol.at<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(mcol.at<float>({0, 3}), 4.0f);
    EXPECT_FLOAT_EQ(mcol.at<float>({2, 0}), 11.0f);
    EXPECT_FLOAT_EQ(mcol.at<float>({2, 3}), 14.0f);

    // 高维 + 低维：(2,3,4) + (4,)
    std::vector<float> cube_v;
    for (int i = 0; i < 24; ++i) cube_v.push_back(float(i));
    Tensor cube(cube_v, {2, 3, 4});
    auto cube_b = cube + brow;
    EXPECT_EQ(cube_b.shape(0), 2);
    EXPECT_EQ(cube_b.shape(1), 3);
    EXPECT_EQ(cube_b.shape(2), 4);
    EXPECT_FLOAT_EQ(cube_b.at<float>({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(cube_b.at<float>({1, 2, 3}), 27.0f);  // 23 + 4

    // dst 变体 + 广播
    Tensor bdst({3, 4}, DataType::FLOAT32);
    ops::Add(brow, bmat, bdst);
    EXPECT_FLOAT_EQ(bdst.at<float>({1, 2}), 9.0f);
    EXPECT_FLOAT_EQ(bdst.at<float>({2, 3}), 15.0f);

    // 不可广播必须报错
    Tensor bad({5}, DataType::FLOAT32);
    EXPECT_THROW((brow + bad), std::exception);
    // dst 形状不等于广播结果必须报错
    Tensor bdst2({4, 3}, DataType::FLOAT32);
    EXPECT_THROW(ops::Add(brow, bmat, bdst2), std::exception);

    // 广播 + 混合 dtype + 视图组合：f32 视图 (3,1) + f64 (1,4) -> F64
    Tensor f32base = Tensor::Fill({3, 1}, 1.0f, DataType::FLOAT32);
    Tensor f64row = Tensor::Fill({1, 4}, 0.5, DataType::FLOAT64);
    auto viewcol = f32base.slice({{0, 3}, {0, 1}});
    auto mixed = viewcol + f64row;
    EXPECT_EQ(mixed.dtype(), DataType::FLOAT64);
    EXPECT_DOUBLE_EQ(mixed.at<double>({0, 0}), 1.5);
    EXPECT_DOUBLE_EQ(mixed.at<double>({2, 3}), 1.5);

    // 广播不修改操作数
    EXPECT_FLOAT_EQ(col.at<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(bmat.at<float>({0, 0}), 0.0f);
}

// ---------- 17. 逻辑 op 广播 ----------
TEST(Broadcast, Comparison) {
    std::vector<float> mat_v;
    for (int i = 0; i < 12; ++i) mat_v.push_back(float(i));
    Tensor bmat(mat_v, {3, 4});  // [[0..3],[4..7],[8..11]]

    std::vector<float> lrow_v{5, 5, 5, 5};
    Tensor lrow(lrow_v, {4});
    auto lt = lrow < bmat;
    EXPECT_EQ(lt.dtype(), DataType::INT8);
    EXPECT_EQ(lt.shape(0), 3);
    EXPECT_EQ(lt.shape(1), 4);
    EXPECT_EQ(lt.at<std::int8_t>({0, 0}), 0);  // 5 < 0 false
    EXPECT_EQ(lt.at<std::int8_t>({1, 0}), 0);  // 5 < 4 false
    EXPECT_EQ(lt.at<std::int8_t>({2, 0}), 1);  // 5 < 8 true
    EXPECT_EQ(lt.at<std::int8_t>({0, 3}), 0);  // 5 < 3 false
    EXPECT_EQ(lt.at<std::int8_t>({1, 1}), 0);  // 5 < 5 false（严格小于）

    // 列向量相等比较：(3,1) == (3,4)，混合 dtype（f32 列 vs f64 矩阵）
    Tensor leq_col = Tensor::Fill({3, 1}, 5.0f, DataType::FLOAT32);
    Tensor leq_mat = Tensor::Fill({3, 4}, 5.0, DataType::FLOAT64);
    auto leq = leq_col == leq_mat;
    EXPECT_EQ(leq.shape(0), 3);
    EXPECT_EQ(leq.shape(1), 4);
    EXPECT_EQ(leq.at<std::int8_t>({0, 0}), 1);
    EXPECT_EQ(leq.at<std::int8_t>({2, 3}), 1);

    // 双向广播：(2,1) vs (1,3) Greater
    Tensor g2a = Tensor::Fill({2, 1}, 1.0f, DataType::FLOAT32);
    Tensor g2b = Tensor::Fill({1, 3}, 0.0f, DataType::FLOAT32);
    auto gt = g2a > g2b;
    EXPECT_EQ(gt.shape(0), 2);
    EXPECT_EQ(gt.shape(1), 3);
    EXPECT_EQ(gt.at<std::int8_t>({1, 2}), 1);
}

// ---------- 18. 标量比较（零临时张量路径） ----------
TEST(Broadcast, ScalarComparison) {
    std::vector<float> sc_v{1, 5, 3, 5};
    Tensor sct(sc_v, {4});

    auto seq = sct == 5.0f;  // 容差内相等
    EXPECT_EQ(seq.dtype(), DataType::INT8);
    EXPECT_EQ(seq.at<std::int8_t>({0}), 0);
    EXPECT_EQ(seq.at<std::int8_t>({1}), 1);
    EXPECT_EQ(seq.at<std::int8_t>({3}), 1);

    auto slt = sct < 4.0f;
    EXPECT_EQ(slt.at<std::int8_t>({0}), 1);  // 1 < 4
    EXPECT_EQ(slt.at<std::int8_t>({1}), 0);

    auto sge = sct >= 3.0f;
    EXPECT_EQ(sge.at<std::int8_t>({2}), 1);
    EXPECT_EQ(sge.at<std::int8_t>({0}), 0);  // 1 >= 3 false

    auto sne = sct != 5.0f;
    EXPECT_EQ(sne.at<std::int8_t>({1}), 0);
    EXPECT_EQ(sne.at<std::int8_t>({2}), 1);

    // int 张量 vs float 标量（混合 dtype，PromotedType 比较）
    std::vector<std::int32_t> si_v{2, 4, 6};
    Tensor sit(si_v, {3});
    auto sgt = sit > 3.5f;
    EXPECT_EQ(sgt.at<std::int8_t>({0}), 0);  // 2 > 3.5
    EXPECT_EQ(sgt.at<std::int8_t>({1}), 1);
    EXPECT_EQ(sgt.at<std::int8_t>({2}), 1);

    // 视图上的标量比较
    auto sv = sct.slice({{1, 3}});
    EXPECT_EQ((sv == 5.0f).at<std::int8_t>({0}), 1);
}

// ---------- 19. 元数据布局（DimArray） ----------
TEST(Metadata, DimArray) {
    // unsqueeze 后 strides 正确重算
    Tensor uq = Tensor::Ones({2, 3}, DataType::FLOAT32);
    uq.unsqueeze(0);
    EXPECT_EQ(uq.shape().size(), 3u);
    EXPECT_FLOAT_EQ(uq.at<float>({0, 1, 2}), 1.0f);

    // 维度数超过 8 拒绝
    EXPECT_THROW((Tensor({1, 1, 1, 1, 1, 1, 1, 1, 1}, DataType::FLOAT32)), std::exception);

    // span 返回 + range-for
    std::int64_t prod = 1;
    for (std::int64_t d : uq.shape()) prod *= d;
    EXPECT_EQ(prod, 6);
}

// ---------- 20. 超越函数 SIMD 精度回归 ----------
TEST(Numeric, Transcendentals) {
    // exp：|x| 控制到 <= 4，保证误差落在 kEps 绝对容差内
    std::vector<float> ex_xs{-4.f, -3.f, -1.f, -0.001f, 0.f, 0.001f, 1.f, 2.f, 3.f, 4.f};
    Tensor et(ex_xs, {static_cast<std::int64_t>(ex_xs.size())});
    const Tensor& etc = et;
    auto ex = ops::Exp(etc);
    for (std::size_t i = 0; i < ex_xs.size(); ++i)
        EXPECT_NEAR(ex.at<float>({static_cast<std::int64_t>(i)}), std::exp(ex_xs[i]), kEps) << "i=" << i;

    // sigmoid / tanh / silu：输出有界，全范围（含 ±9 饱和边界）核对
    std::vector<float> axs{-9.f, -5.f, -1.f, -0.001f, 0.f, 0.001f, 1.f, 5.f, 9.f};
    Tensor at(axs, {static_cast<std::int64_t>(axs.size())});
    const Tensor& atc = at;

    auto sg = ops::Sigmoid(atc);
    for (std::size_t i = 0; i < axs.size(); ++i)
        EXPECT_NEAR(sg.at<float>({static_cast<std::int64_t>(i)}),
                    1.0f / (1.0f + std::exp(-axs[i])), kEps) << "i=" << i;

    auto th = ops::Tanh(atc);
    for (std::size_t i = 0; i < axs.size(); ++i)
        EXPECT_NEAR(th.at<float>({static_cast<std::int64_t>(i)}), std::tanh(axs[i]), kEps) << "i=" << i;

    auto sl = ops::Silu(atc);
    for (std::size_t i = 0; i < axs.size(); ++i)
        EXPECT_NEAR(sl.at<float>({static_cast<std::int64_t>(i)}),
                    axs[i] / (1.0f + std::exp(-axs[i])), kEps) << "i=" << i;
}

// ---------- 21. 整型 softmax：结果提升为 FLOAT32 ----------
TEST(Numeric, IntegerSoftmax) {
    std::vector<std::int32_t> ism_v{1, 2, 3, 4};
    Tensor ism(ism_v, {4});
    auto ism_out = ops::Softmax(ism, 0);
    EXPECT_EQ(ism_out.dtype(), DataType::FLOAT32);

    float isum = 0.0f;
    for (std::size_t i = 0; i < ism_v.size(); ++i)
        isum += std::exp(static_cast<float>(ism_v[i]));
    for (std::size_t i = 0; i < ism_v.size(); ++i) {
        float want = std::exp(static_cast<float>(ism_v[i])) / isum;
        EXPECT_NEAR(ism_out.at<float>({static_cast<std::int64_t>(i)}), want, kEps);
    }
}

// ---------- 22. argmax/argmin 并列值 tie-break：取首个极值 ----------
TEST(Reduce, ArgmaxTieBreak) {
    std::vector<float> tmax{0.f, 0.f, 0.f, 9.f, 0.f, 0.f, 0.f, 0.f, 0.f, 9.f};
    Tensor ttmax(tmax, {10});
    EXPECT_EQ(ops::Argmax(ttmax, 0).at<std::int32_t>({0}), 3);  // 9 出现在 3 与 9，取 3

    std::vector<float> tmin{9.f, 9.f, 9.f, 1.f, 9.f, 9.f, 9.f, 9.f, 9.f, 1.f};
    Tensor ttmin(tmin, {10});
    EXPECT_EQ(ops::Argmin(ttmin, 0).at<std::int32_t>({0}), 3);

    // 跨越两个完整向量块（n_axis >= 16），并列极值仍取首个
    std::vector<float> tmax16(16, 0.f);
    tmax16[2] = 5.f;
    tmax16[13] = 5.f;
    Tensor ttmax16(tmax16, {16});
    EXPECT_EQ(ops::Argmax(ttmax16, 0).at<std::int32_t>({0}), 2);

    std::vector<float> tmin16(16, 9.f);
    tmin16[2] = 1.f;
    tmin16[13] = 1.f;
    Tensor ttmin16(tmin16, {16});
    EXPECT_EQ(ops::Argmin(ttmin16, 0).at<std::int32_t>({0}), 2);
}

// ---------- 23. 创建：arange / linspace / eye ----------
TEST(Create, ArangeLinspaceEye) {
    auto ar = ops::Arange(0, 5);  // 0,1,2,3,4
    EXPECT_EQ(ar.shape(0), 5);
    EXPECT_FLOAT_EQ(ar.at<float>({0}), 0.0f);
    EXPECT_FLOAT_EQ(ar.at<float>({4}), 4.0f);

    auto ar2 = ops::Arange(1.0f, 5.0f, 2.0f);  // 1,3
    EXPECT_EQ(ar2.shape(0), 2);
    EXPECT_FLOAT_EQ(ar2.at<float>({0}), 1.0f);
    EXPECT_FLOAT_EQ(ar2.at<float>({1}), 3.0f);

    auto ar3 = ops::Arange(5, 0, -1);  // 5,4,3,2,1
    EXPECT_EQ(ar3.shape(0), 5);
    EXPECT_FLOAT_EQ(ar3.at<float>({0}), 5.0f);
    EXPECT_FLOAT_EQ(ar3.at<float>({4}), 1.0f);

    auto ar4 = ops::Arange(0, 4, 1.0f, DataType::INT32);
    EXPECT_EQ(ar4.dtype(), DataType::INT32);
    EXPECT_EQ(ar4.at<std::int32_t>({3}), 3);

    EXPECT_THROW(ops::Arange(0, 5, 0.0f), std::exception);   // step 为零
    EXPECT_THROW(ops::Arange(0, 5, -1.0f), std::exception);  // 方向与 step 符号矛盾

    auto ls = ops::Linspace(0.0f, 1.0f, 5);  // 0,0.25,0.5,0.75,1
    EXPECT_EQ(ls.shape(0), 5);
    EXPECT_NEAR(ls.at<float>({0}), 0.0f, kEps);
    EXPECT_NEAR(ls.at<float>({4}), 1.0f, kEps);
    EXPECT_NEAR(ls.at<float>({2}), 0.5f, kEps);

    auto ls1 = ops::Linspace(3.0f, 3.0f, 1);
    EXPECT_EQ(ls1.shape(0), 1);
    EXPECT_NEAR(ls1.at<float>({0}), 3.0f, kEps);

    auto e = ops::Eye(3);
    EXPECT_EQ(e.shape(0), 3);
    EXPECT_EQ(e.shape(1), 3);
    EXPECT_FLOAT_EQ(e.at<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(e.at<float>({1, 2}), 0.0f);
    EXPECT_FLOAT_EQ(e.at<float>({2, 2}), 1.0f);

    auto e2 = ops::Eye(2, 3);  // 2x3 非方阵
    EXPECT_EQ(e2.shape(0), 2);
    EXPECT_EQ(e2.shape(1), 3);
    EXPECT_FLOAT_EQ(e2.at<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(e2.at<float>({1, 1}), 1.0f);
    EXPECT_FLOAT_EQ(e2.at<float>({0, 2}), 0.0f);
}

// ---------- 24. where / index_select / gather ----------
TEST(Indexing, WhereIndexSelectGather) {
    std::vector<float> wv{1, 5, 3, 5};
    Tensor wt(wv, {4});
    std::vector<float> wa_v{10, 20, 30, 40};
    Tensor wa(wa_v, {4});
    std::vector<float> wb_v{-1, -2, -3, -4};
    Tensor wb(wb_v, {4});
    auto wmask = wt > 3.0f;  // [0,1,0,1] INT8

    auto w1 = ops::Where(wmask, wa, wb);  // [-1, 20, -3, 40]
    EXPECT_FLOAT_EQ(w1.at<float>({0}), -1.0f);
    EXPECT_FLOAT_EQ(w1.at<float>({1}), 20.0f);
    EXPECT_FLOAT_EQ(w1.at<float>({2}), -3.0f);
    EXPECT_FLOAT_EQ(w1.at<float>({3}), 40.0f);

    auto w2 = ops::Where(wmask, 0.0f, wa);  // cond?0:wa = [10,0,30,0]
    EXPECT_FLOAT_EQ(w2.at<float>({0}), 10.0f);
    EXPECT_FLOAT_EQ(w2.at<float>({1}), 0.0f);
    EXPECT_FLOAT_EQ(w2.at<float>({2}), 30.0f);
    EXPECT_FLOAT_EQ(w2.at<float>({3}), 0.0f);

    auto w3 = ops::Where(wmask, wa, 0.0f);  // cond?wa:0 = [0,20,0,40]
    EXPECT_FLOAT_EQ(w3.at<float>({0}), 0.0f);
    EXPECT_FLOAT_EQ(w3.at<float>({1}), 20.0f);
    EXPECT_FLOAT_EQ(w3.at<float>({2}), 0.0f);
    EXPECT_FLOAT_EQ(w3.at<float>({3}), 40.0f);

    // 混合 dtype：INT32 + FLOAT32 -> FLOAT32
    std::vector<std::int32_t> wi_v{1, 2, 3, 4};
    Tensor wi(wi_v, {4});
    std::vector<float> wf_v{0.5f, 0.5f, 0.5f, 0.5f};
    Tensor wf(wf_v, {4});
    auto wmix = ops::Where(wi > 2.0f, wi, wf);  // [0.5, 0.5, 3.0, 4.0]
    EXPECT_EQ(wmix.dtype(), DataType::FLOAT32);
    EXPECT_FLOAT_EQ(wmix.at<float>({0}), 0.5f);
    EXPECT_FLOAT_EQ(wmix.at<float>({2}), 3.0f);
    EXPECT_FLOAT_EQ(wmix.at<float>({3}), 4.0f);

    // 3 路广播：cond(2,1) + a(2,3) + b(2,3)
    std::vector<float> bc_v{1, 0};
    Tensor bcond(bc_v, {2, 1});
    std::vector<float> ba_v{1, 2, 3, 4, 5, 6};
    Tensor ba(ba_v, {2, 3});
    std::vector<float> bneg_v{-1, -2, -3, -4, -5, -6};
    Tensor bneg(bneg_v, {2, 3});
    auto wbcast = ops::Where(bcond, ba, bneg);
    EXPECT_EQ(wbcast.shape(0), 2);
    EXPECT_EQ(wbcast.shape(1), 3);
    EXPECT_FLOAT_EQ(wbcast.at<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(wbcast.at<float>({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(wbcast.at<float>({1, 0}), -4.0f);
    EXPECT_FLOAT_EQ(wbcast.at<float>({1, 2}), -6.0f);

    // index_select：embedding 查表（dim=0）
    std::vector<float> emb_v{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    Tensor emb(emb_v, {4, 3});
    std::vector<std::int32_t> idx_v{2, 0, 2};
    Tensor idx(idx_v, {3});
    auto sel = ops::IndexSelect(emb, 0, idx);
    EXPECT_EQ(sel.shape(0), 3);
    EXPECT_EQ(sel.shape(1), 3);
    EXPECT_FLOAT_EQ(sel.at<float>({0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(sel.at<float>({0, 2}), 8.0f);
    EXPECT_FLOAT_EQ(sel.at<float>({1, 0}), 0.0f);
    EXPECT_FLOAT_EQ(sel.at<float>({2, 1}), 7.0f);

    // index_select：dim=1
    std::vector<std::int32_t> idx2_v{2, 0};
    Tensor idx2(idx2_v, {2});
    auto sel2 = ops::IndexSelect(emb, 1, idx2);
    EXPECT_EQ(sel2.shape(0), 4);
    EXPECT_EQ(sel2.shape(1), 2);
    EXPECT_FLOAT_EQ(sel2.at<float>({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(sel2.at<float>({0, 1}), 0.0f);
    EXPECT_FLOAT_EQ(sel2.at<float>({3, 0}), 11.0f);
    EXPECT_FLOAT_EQ(sel2.at<float>({3, 1}), 9.0f);

    // gather：PyTorch 语义（index 与 input 同秩）
    std::vector<float> gv{1, 2, 3, 4};
    Tensor g(gv, {2, 2});
    std::vector<std::int32_t> gi_v{0, 0, 1, 0};
    Tensor gi(gi_v, {2, 2});
    auto go1 = ops::Gather(g, 1, gi);  // [[1,1],[4,3]]
    EXPECT_FLOAT_EQ(go1.at<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(go1.at<float>({0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(go1.at<float>({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(go1.at<float>({1, 1}), 3.0f);

    std::vector<std::int32_t> gi0_v{0, 1, 1, 0};
    Tensor gi0(gi0_v, {2, 2});
    auto go0 = ops::Gather(g, 0, gi0);  // [[1,4],[3,2]]
    EXPECT_FLOAT_EQ(go0.at<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(go0.at<float>({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(go0.at<float>({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(go0.at<float>({1, 1}), 2.0f);

    // 错误路径：非 1D index / 越界 index
    EXPECT_THROW(ops::IndexSelect(emb, 0, gi), std::exception);  // gi 是 2D
    std::vector<std::int32_t> idx_bad_v{5};
    Tensor idx_bad(idx_bad_v, {1});
    EXPECT_THROW(ops::IndexSelect(emb, 0, idx_bad), std::exception);  // 越界（axis_len=4）
}

// ---------- 25. 数值覆盖：prod / var / std / norm / cumsum / log_softmax / erf ----------
TEST(Numeric, ReductionsAndSpecialFunctions) {
    std::vector<float> nv{1, 2, 3, 4, 5};
    Tensor n1(nv, {5});
    EXPECT_NEAR(ops::Prod(n1), 120.0f, kEps);
    // var / std（总体 ddof=0）：mean=3, var=2, std=sqrt(2)
    EXPECT_NEAR(ops::Var(n1), 2.0f, kEps);
    EXPECT_NEAR(ops::Std(n1), std::sqrt(2.0f), kEps);
    // norm：L1 = 15，L2 = sqrt(55)
    EXPECT_NEAR(ops::Norm(n1, 1.0f), 15.0f, kEps);
    EXPECT_NEAR(ops::Norm(n1, 2.0f), std::sqrt(55.0f), kEps);

    auto cs1 = ops::Cumsum(n1, 0);
    EXPECT_NEAR(cs1.at<float>({0}), 1.0f, kEps);
    EXPECT_NEAR(cs1.at<float>({1}), 3.0f, kEps);
    EXPECT_NEAR(cs1.at<float>({4}), 15.0f, kEps);

    std::vector<float> m2v{1, 2, 3, 4, 5, 6};
    Tensor m2(m2v, {2, 3});  // [[1,2,3],[4,5,6]]

    // prod axis：dim=0 -> [4,10,18]；dim=1 -> [6,120]
    auto pa0 = ops::Prod(m2, 0);
    EXPECT_NEAR(pa0.at<float>({0}), 4.0f, kEps);
    EXPECT_NEAR(pa0.at<float>({1}), 10.0f, kEps);
    EXPECT_NEAR(pa0.at<float>({2}), 18.0f, kEps);
    auto pa1 = ops::Prod(m2, 1);
    EXPECT_NEAR(pa1.at<float>({0}), 6.0f, kEps);
    EXPECT_NEAR(pa1.at<float>({1}), 120.0f, kEps);

    // var / std axis：每行 var = 2/3
    auto va1 = ops::Var(m2, 1);
    EXPECT_EQ(va1.dtype(), DataType::FLOAT32);
    EXPECT_NEAR(va1.at<float>({0}), 2.0f / 3.0f, kEps);
    EXPECT_NEAR(va1.at<float>({1}), 2.0f / 3.0f, kEps);
    auto sa1 = ops::Std(m2, 1);
    EXPECT_NEAR(sa1.at<float>({0}), std::sqrt(2.0f / 3.0f), kEps);
    EXPECT_NEAR(sa1.at<float>({1}), std::sqrt(2.0f / 3.0f), kEps);

    // norm axis：L1 -> [6,15]；L2 -> [sqrt(14), sqrt(77)]
    auto na1 = ops::Norm(m2, 1.0f, 1);
    EXPECT_NEAR(na1.at<float>({0}), 6.0f, kEps);
    EXPECT_NEAR(na1.at<float>({1}), 15.0f, kEps);
    auto na2 = ops::Norm(m2, 2.0f, 1);
    EXPECT_NEAR(na2.at<float>({0}), std::sqrt(14.0f), kEps);
    EXPECT_NEAR(na2.at<float>({1}), std::sqrt(77.0f), kEps);

    // cumsum axis 1 -> [[1,3,6],[4,9,15]]
    auto cs2 = ops::Cumsum(m2, 1);
    EXPECT_NEAR(cs2.at<float>({0, 2}), 6.0f, kEps);
    EXPECT_NEAR(cs2.at<float>({1, 2}), 15.0f, kEps);
    // cumsum axis 0 -> [[1,2,3],[5,7,9]]
    auto cs0 = ops::Cumsum(m2, 0);
    EXPECT_NEAR(cs0.at<float>({1, 0}), 5.0f, kEps);
    EXPECT_NEAR(cs0.at<float>({1, 2}), 9.0f, kEps);

    // log_softmax == log(softmax)
    std::vector<float> lv{1, 2, 3};
    Tensor l1(lv, {3});
    auto lsm = ops::LogSoftmax(l1, 0);
    auto smf = ops::Softmax(l1, 0);
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(lsm.at<float>({i}), std::log(smf.at<float>({i})), kEps) << "i=" << i;

    // erf 对标 std::erf
    std::vector<float> ev{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    Tensor et(ev, {5});
    auto er = ops::Erf(et);
    for (int i = 0; i < 5; ++i)
        EXPECT_NEAR(er.at<float>({i}), std::erf(ev[i]), kEps) << "i=" << i;
}

// ---------- 26. layer_norm / rms_norm / randn / mse / nll / cross_entropy ----------
TEST(Numeric, NormsAndLosses) {
    // layer_norm：[[1,2,3],[4,5,6]]，axis=1 对每行减均值除标准差（weight=1, bias=0）
    std::vector<float> lnv{1, 2, 3, 4, 5, 6};
    Tensor ln(lnv, {2, 3});
    std::vector<float> ln_wv{1, 1, 1};
    Tensor lw(ln_wv, {3});
    std::vector<float> ln_bv{0, 0, 0};
    Tensor lb(ln_bv, {3});
    auto ln_out = ops::LayerNorm(ln, lw, lb, 1, 1e-5f);
    const float ln_scale = 1.0f / std::sqrt(2.0f / 3.0f);
    EXPECT_NEAR(ln_out.at<float>({0, 0}), (1.0f - 2.0f) * ln_scale, kEps);
    EXPECT_NEAR(ln_out.at<float>({0, 1}), 0.0f, kEps);
    EXPECT_NEAR(ln_out.at<float>({0, 2}), (3.0f - 2.0f) * ln_scale, kEps);
    EXPECT_NEAR(ln_out.at<float>({1, 0}), (4.0f - 5.0f) * ln_scale, kEps);
    EXPECT_NEAR(ln_out.at<float>({1, 2}), (6.0f - 5.0f) * ln_scale, kEps);

    // rms_norm：weight=1，每行除以 sqrt(mean(x²)+eps)
    auto rn_out = ops::RmsNorm(ln, lw, 1, 1e-5f);
    const float rms0 = std::sqrt(14.0f / 3.0f);
    const float rms1 = std::sqrt(77.0f / 3.0f);
    EXPECT_NEAR(rn_out.at<float>({0, 0}), 1.0f / rms0, kEps);
    EXPECT_NEAR(rn_out.at<float>({0, 2}), 3.0f / rms0, kEps);
    EXPECT_NEAR(rn_out.at<float>({1, 1}), 5.0f / rms1, kEps);

    // randn：统计校验 mean≈0, std≈1（大样本，不 flaky）
    auto rn = ops::Randn({100000}, DataType::FLOAT32, 0.0f, 1.0f);
    float rn_mean = ops::Mean(rn);
    float rn_std = ops::Std(rn);
    EXPECT_GT(rn_mean, -0.05f);
    EXPECT_LT(rn_mean, 0.05f);
    EXPECT_GT(rn_std, 0.95f);
    EXPECT_LT(rn_std, 1.05f);

    // mse_loss：mean((x-y)²)
    std::vector<float> pv{1, 2, 3};
    Tensor p(pv, {3});
    std::vector<float> qv{1, 3, 5};
    Tensor q(qv, {3});
    EXPECT_NEAR(ops::MseLoss(p, q), (0.0f + 1.0f + 4.0f) / 3.0f, kEps);

    // nll_loss：-mean(log_prob[n, target[n]])
    std::vector<float> lpv{std::log(0.5f), std::log(0.3f), std::log(0.2f),
                           std::log(0.1f), std::log(0.7f), std::log(0.2f)};
    Tensor logp(lpv, {2, 3});
    std::vector<std::int64_t> tv{0, 1};
    Tensor tgt(tv, {2});
    EXPECT_NEAR(ops::NllLoss(logp, tgt), -(std::log(0.5f) + std::log(0.7f)) / 2.0f, kEps);

    // cross_entropy = nll(log_softmax(logits))
    std::vector<float> lgv{1, 2, 3, 0, 1, 2};
    Tensor logits(lgv, {2, 3});
    EXPECT_NEAR(ops::CrossEntropyLoss(logits, tgt),
                ops::NllLoss(ops::LogSoftmax(logits, 1), tgt), kEps);
}
