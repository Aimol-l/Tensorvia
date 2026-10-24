#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include "tensorvia.hpp"

namespace {

constexpr float kEps = 1e-4f;

// Copy to host through the regular reference-semantics path; `at` only works on
// CPU tensors.
Tensor host_of(const Tensor& t) {
    Tensor h = t;
    h.to_host();
    return h;
}

}  // namespace

// The machine may have no driver or no device: probing with a one-element SYCL
// tensor is the only reliable check, so every test skips when the probe fails.
class SyclBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        try {
            Tensor probe = Tensor::Fill({1}, 0.0f, DataType::FLOAT32, Device::SYCL);
            (void)probe;
        } catch (const std::exception& e) {
            GTEST_SKIP() << "no usable SYCL device: " << e.what();
        }
    }
};

// ---------- creation + host round trip ----------
TEST_F(SyclBackendTest, CreationAndRoundTrip) {
    Tensor a = Tensor::Fill({2, 3}, 2.0f, DataType::FLOAT32, Device::SYCL);
    EXPECT_EQ(a.device(), Device::SYCL);
    EXPECT_EQ(a.numel(), 6u);
    Tensor h = host_of(a);
    EXPECT_EQ(h.device(), Device::CPU);
    EXPECT_NEAR(h.at<float>({0, 0}), 2.0f, kEps);
    EXPECT_NEAR(h.at<float>({1, 2}), 2.0f, kEps);

    std::vector<float> v{1, 2, 3, 4};
    Tensor c(v, {2, 2});
    c.to_device(Device::SYCL);
    EXPECT_EQ(c.device(), Device::SYCL);
    c.to_host();
    EXPECT_EQ(c.device(), Device::CPU);
    EXPECT_NEAR(c.at<float>({0, 1}), 2.0f, kEps);
    EXPECT_NEAR(c.at<float>({1, 1}), 4.0f, kEps);
}

TEST_F(SyclBackendTest, FilledConstructors) {
    Tensor z = Tensor::Zeros({3, 2}, DataType::FLOAT32, Device::SYCL);
    Tensor o = Tensor::Ones({3, 2}, DataType::FLOAT32, Device::SYCL);
    Tensor zh = host_of(z);
    Tensor oh = host_of(o);
    for (int64_t i = 0; i < 3; ++i) {
        for (int64_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(zh.at<float>({i, j}), 0.0f, kEps);
            EXPECT_NEAR(oh.at<float>({i, j}), 1.0f, kEps);
        }
    }

    // Integer and half-precision fills go through the same island path.
    Tensor i64 = host_of(Tensor::Fill({4}, 3.0f, DataType::INT64, Device::SYCL));
    EXPECT_EQ(i64.dtype(), DataType::INT64);
    EXPECT_EQ(i64.at<int64_t>({0}), 3);
    Tensor f16 = host_of(Tensor::Fill({4}, 1.5f, DataType::FLOAT16, Device::SYCL));
    EXPECT_EQ(f16.dtype(), DataType::FLOAT16);
}

TEST_F(SyclBackendTest, RandomStaysInRange) {
    Tensor r = host_of(Tensor::Random({64}, -2.0f, 3.0f, DataType::FLOAT32, Device::SYCL));
    for (int64_t i = 0; i < 64; ++i) {
        const float v = r.at<float>({i});
        EXPECT_GE(v, -2.0f);
        EXPECT_LT(v, 3.0f);
    }
    // Not all identical: a broken seed path would fill a constant.
    std::vector<float> vals;
    for (int64_t i = 0; i < 64; ++i) vals.push_back(r.at<float>({i}));
    EXPECT_NE(*std::min_element(vals.begin(), vals.end()),
              *std::max_element(vals.begin(), vals.end()));
}

// ---------- deep copy + strided materialization ----------
TEST_F(SyclBackendTest, CloneIsDeep) {
    Tensor a = Tensor::Fill({3, 3}, 1.0f, DataType::FLOAT32, Device::SYCL);
    Tensor b = a.clone();
    EXPECT_EQ(b.device(), Device::SYCL);
    EXPECT_NE(a.data(), b.data());
    Tensor hb = host_of(b);
    EXPECT_NEAR(hb.at<float>({2, 2}), 1.0f, kEps);
}

TEST_F(SyclBackendTest, CloneOfPermutedViewRepacks) {
    std::vector<float> v{0, 1, 2, 3, 4, 5};
    Tensor t(v, {2, 3});
    t.to_device(Device::SYCL);
    Tensor p = t.permute({1, 0});
    EXPECT_FALSE(p.is_contiguous());

    Tensor c = p.clone();
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_EQ(c.device(), Device::SYCL);
    Tensor h = host_of(c);
    EXPECT_EQ(h.shape(0), 3);
    EXPECT_EQ(h.shape(1), 2);
    EXPECT_NEAR(h.at<float>({0, 0}), 0.0f, kEps);
    EXPECT_NEAR(h.at<float>({0, 1}), 3.0f, kEps);
    EXPECT_NEAR(h.at<float>({1, 0}), 1.0f, kEps);
    EXPECT_NEAR(h.at<float>({2, 1}), 5.0f, kEps);
}

// ---------- arithmetic ----------
TEST_F(SyclBackendTest, ScalarArithmetic) {
    Tensor a = Tensor::Fill({2, 3}, 2.0f, DataType::FLOAT32, Device::SYCL);

    // In-place forms mutate the device buffer; the result form must leave the
    // source untouched.
    Tensor inplace = a.clone();
    ops::Add(inplace, 1.0f);
    Tensor hi = host_of(inplace);
    EXPECT_NEAR(hi.at<float>({0, 0}), 3.0f, kEps);
    EXPECT_NEAR(hi.at<float>({0, 1}), 3.0f, kEps);

    const Tensor& ac = a;
    Tensor sub = host_of(ops::Sub(ac, 5.0f));
    EXPECT_NEAR(sub.at<float>({0, 0}), -3.0f, kEps);
    Tensor mul = host_of(ops::Mul(ac, 4.0f));
    EXPECT_NEAR(mul.at<float>({1, 2}), 8.0f, kEps);
    Tensor div = host_of(ops::Div(ac, 4.0f));
    EXPECT_NEAR(div.at<float>({1, 2}), 0.5f, kEps);

    Tensor hsrc = host_of(a);
    EXPECT_NEAR(hsrc.at<float>({0, 0}), 2.0f, kEps);
}

TEST_F(SyclBackendTest, TensorArithmeticAndBroadcast) {
    std::vector<float> av{1, 2, 3, 4, 5, 6};
    std::vector<float> bv{10, 20, 30, 40, 50, 60};
    Tensor a(av, {2, 3});
    Tensor b(bv, {2, 3});
    a.to_device(Device::SYCL);
    b.to_device(Device::SYCL);

    Tensor sum = host_of(ops::Add(a, b));
    Tensor diff = host_of(ops::Sub(a, b));
    Tensor prod = host_of(ops::Mul(a, b));
    Tensor quot = host_of(ops::Div(b, a));
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            const float x = av[static_cast<size_t>(i * 3 + j)];
            const float y = bv[static_cast<size_t>(i * 3 + j)];
            EXPECT_NEAR(sum.at<float>({i, j}), x + y, kEps);
            EXPECT_NEAR(diff.at<float>({i, j}), x - y, kEps);
            EXPECT_NEAR(prod.at<float>({i, j}), x * y, kEps);
            EXPECT_NEAR(quot.at<float>({i, j}), y / x, kEps);
        }
    }

    // [2,3] + [3] broadcasts over rows; [2,1] + [1,3] broadcasts on both axes.
    Tensor row = Tensor::Fill({3}, 100.0f, DataType::FLOAT32, Device::SYCL);
    Tensor bcast = host_of(ops::Add(a, row));
    EXPECT_NEAR(bcast.at<float>({0, 0}), 101.0f, kEps);
    EXPECT_NEAR(bcast.at<float>({1, 2}), 106.0f, kEps);

    Tensor col = Tensor::Fill({2, 1}, 1.0f, DataType::FLOAT32, Device::SYCL);
    Tensor row2 = Tensor::Fill({1, 3}, 2.0f, DataType::FLOAT32, Device::SYCL);
    Tensor grid = host_of(ops::Mul(col, row2));
    EXPECT_EQ(grid.shape(0), 2);
    EXPECT_EQ(grid.shape(1), 3);
    EXPECT_NEAR(grid.at<float>({1, 2}), 2.0f, kEps);

    // dst form writes into a caller-provided contiguous buffer.
    Tensor dst({2, 3}, DataType::FLOAT32, Device::SYCL);
    ops::Add(a, b, dst);
    Tensor hdst = host_of(dst);
    EXPECT_NEAR(hdst.at<float>({1, 1}), 55.0f, kEps);
}

TEST_F(SyclBackendTest, MixedDtypePromotion) {
    std::vector<float> fv{1, 2, 3, 4};
    std::vector<double> dv{0.5, 0.5, 0.5, 0.5};
    Tensor f(fv, {4});
    Tensor d(dv, {4});
    f.to_device(Device::SYCL);
    d.to_device(Device::SYCL);

    Tensor mixed = host_of(ops::Add(f, d));
    EXPECT_EQ(mixed.dtype(), DataType::FLOAT64);
    EXPECT_DOUBLE_EQ(mixed.at<double>({0}), 1.5);
    EXPECT_DOUBLE_EQ(mixed.at<double>({3}), 4.5);
}

TEST_F(SyclBackendTest, IntegerDivisionByZeroIsZero) {
    std::vector<int32_t> nv{1, -2, 0, 7};
    std::vector<int32_t> zv{0, 0, 0, 0};
    Tensor n(nv, {4});
    Tensor z(zv, {4});
    n.to_device(Device::SYCL);
    z.to_device(Device::SYCL);

    Tensor q = host_of(ops::Div(n, z));
    EXPECT_EQ(q.dtype(), DataType::INT32);
    for (int64_t i = 0; i < 4; ++i) EXPECT_EQ(q.at<int32_t>({i}), 0);
}

// ---------- unary ----------
TEST_F(SyclBackendTest, UnaryOperators) {
    std::vector<float> v{-2.0f, -0.5f, 0.0f, 0.5f, 2.0f, 4.0f};
    Tensor t(v, {6});
    t.to_device(Device::SYCL);
    const Tensor& tc = t;

    Tensor abs_h = host_of(ops::Abs(tc));
    Tensor sin_h = host_of(ops::Sin(tc));
    Tensor cos_h = host_of(ops::Cos(tc));
    Tensor tan_h = host_of(ops::Tan(tc));
    Tensor exp_h = host_of(ops::Exp(tc));
    for (int64_t i = 0; i < 6; ++i) {
        const float x = v[static_cast<size_t>(i)];
        EXPECT_NEAR(abs_h.at<float>({i}), std::abs(x), kEps);
        EXPECT_NEAR(sin_h.at<float>({i}), std::sin(x), kEps);
        EXPECT_NEAR(cos_h.at<float>({i}), std::cos(x), kEps);
        EXPECT_NEAR(tan_h.at<float>({i}), std::tan(x), kEps);
        EXPECT_NEAR(exp_h.at<float>({i}), std::exp(x), 1e-3f);
    }

    Tensor powed = host_of(ops::Pow(
        Tensor::Fill({6}, 9.0f, DataType::FLOAT32, Device::SYCL), 2.0f));
    EXPECT_NEAR(powed.at<float>({0}), 81.0f, kEps);

    // Sqrt/Log run through the Any(a < 0) guard, which needs comparison.
    std::vector<float> pos_v{1.0f, 4.0f, 9.0f, 16.0f};
    Tensor pos(pos_v, {4});
    pos.to_device(Device::SYCL);
    const Tensor& pos_c = pos;
    Tensor sqrt_h = host_of(ops::Sqrt(pos_c));
    for (int64_t i = 0; i < 4; ++i)
        EXPECT_NEAR(sqrt_h.at<float>({i}), std::sqrt(pos_v[static_cast<size_t>(i)]), kEps);
    Tensor log_h = host_of(ops::Log(pos_c, 2.0f));
    for (int64_t i = 0; i < 4; ++i)
        EXPECT_NEAR(log_h.at<float>({i}), std::log2(pos_v[static_cast<size_t>(i)]), kEps);

    Tensor clamped = host_of(ops::Clamp(tc, -1.0f, 1.0f));
    EXPECT_NEAR(clamped.at<float>({0}), -1.0f, kEps);
    EXPECT_NEAR(clamped.at<float>({5}), 1.0f, kEps);

    // In-place unary keeps the buffer identity but updates the values.
    Tensor abs_in = Tensor::Fill({4}, -3.0f, DataType::FLOAT32, Device::SYCL);
    ops::Abs(abs_in);
    Tensor abs_h2 = host_of(abs_in);
    for (int64_t i = 0; i < 4; ++i) EXPECT_NEAR(abs_h2.at<float>({i}), 3.0f, kEps);
}

TEST_F(SyclBackendTest, UnaryIntegerPromotesToFloat32) {
    std::vector<int32_t> iv{0, 1, 2, 3};
    Tensor t(iv, {4});
    t.to_device(Device::SYCL);
    const Tensor& tc = t;
    Tensor out = ops::Sin(tc);
    EXPECT_EQ(out.dtype(), DataType::FLOAT32);
    Tensor h = host_of(out);
    EXPECT_NEAR(h.at<float>({1}), std::sin(1.0f), kEps);
    // Abs keeps the input dtype.
    EXPECT_EQ(ops::Abs(tc).dtype(), DataType::INT32);
}

TEST_F(SyclBackendTest, NanPropagatesThroughClamp) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> v{nan, 1.0f, -1.0f};
    Tensor t(v, {3});
    t.to_device(Device::SYCL);
    const Tensor& tc = t;
    Tensor hc = host_of(ops::Clamp(tc, -0.5f, 0.5f));
    EXPECT_TRUE(std::isnan(hc.at<float>({0})));
    EXPECT_NEAR(hc.at<float>({1}), 0.5f, kEps);
    EXPECT_NEAR(hc.at<float>({2}), -0.5f, kEps);
}

// ---------- matmul ----------
namespace {

// Small explicit reference so a kernel indexing mistake is unambiguous.
std::vector<float> naive_matmul(Tensor& A, Tensor& B) {
    const bool is3d = A.shape().size() == 3;
    const int64_t batch = is3d ? A.shape(0) : 1;
    const int64_t rows  = is3d ? A.shape(1) : A.shape(0);
    const int64_t k     = is3d ? A.shape(2) : A.shape(1);
    const int64_t cols  = is3d ? B.shape(2) : B.shape(1);
    std::vector<float> out(static_cast<size_t>(batch * rows * cols), 0.0f);
    for (int64_t b = 0; b < batch; ++b)
        for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = 0; j < cols; ++j) {
                float s = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    const float av = is3d ? A.at<float>({b, i, p}) : A.at<float>({i, p});
                    const float bv = is3d ? B.at<float>({b, p, j}) : B.at<float>({p, j});
                    s += av * bv;
                }
                out[static_cast<size_t>(b * rows * cols + i * cols + j)] = s;
            }
    return out;
}

}  // namespace

TEST_F(SyclBackendTest, MatMul2D) {
    Tensor A = Tensor::Random({17, 9}, -1.0f, 1.0f, DataType::FLOAT32, Device::SYCL);
    Tensor B = Tensor::Random({9, 23}, -1.0f, 1.0f, DataType::FLOAT32, Device::SYCL);
    Tensor Gh = host_of(A);
    Tensor Hh = host_of(B);
    const std::vector<float> want = naive_matmul(Gh, Hh);

    Tensor got = ops::MatMul(A, B);
    EXPECT_EQ(got.shape(0), 17);
    EXPECT_EQ(got.shape(1), 23);
    Tensor h = host_of(got);
    float max_err = 0.0f;
    for (size_t i = 0; i < want.size(); ++i)
        max_err = std::max(max_err, std::abs(static_cast<const float*>(h.data())[i] - want[i]));
    EXPECT_NEAR(max_err, 0.0f, 1e-4f);
}

TEST_F(SyclBackendTest, MatMulBatched3D) {
    Tensor A = Tensor::Random({3, 5, 7}, -1.0f, 1.0f, DataType::FLOAT32, Device::SYCL);
    Tensor B = Tensor::Random({3, 7, 4}, -1.0f, 1.0f, DataType::FLOAT32, Device::SYCL);
    Tensor Gh = host_of(A);
    Tensor Hh = host_of(B);
    const std::vector<float> want = naive_matmul(Gh, Hh);

    Tensor got = ops::MatMul(A, B);
    EXPECT_EQ(got.shape(0), 3);
    EXPECT_EQ(got.shape(1), 5);
    EXPECT_EQ(got.shape(2), 4);
    Tensor h = host_of(got);
    float max_err = 0.0f;
    for (size_t i = 0; i < want.size(); ++i)
        max_err = std::max(max_err, std::abs(static_cast<const float*>(h.data())[i] - want[i]));
    EXPECT_NEAR(max_err, 0.0f, 1e-4f);
}

TEST_F(SyclBackendTest, MatMulHalfPromotesToFloat32) {
    std::vector<float16> v{float16(1.0f), float16(2.0f), float16(3.0f), float16(4.0f)};
    Tensor hf(v, {2, 2});
    hf.to_device(Device::SYCL);
    Tensor r = ops::MatMul(hf, hf);
    EXPECT_EQ(r.dtype(), DataType::FLOAT32);
    Tensor h = host_of(r);
    EXPECT_NEAR(h.at<float>({0, 0}), 7.0f, 1e-3f);
    EXPECT_NEAR(h.at<float>({0, 1}), 10.0f, 1e-3f);
    EXPECT_NEAR(h.at<float>({1, 0}), 15.0f, 1e-3f);
    EXPECT_NEAR(h.at<float>({1, 1}), 22.0f, 1e-3f);
}

TEST_F(SyclBackendTest, NonContiguousInPlaceRejected) {
    std::vector<float> v{0, 1, 2, 3, 4, 5};
    Tensor t(v, {2, 3});
    t.to_device(Device::SYCL);
    Tensor p = t.permute({1, 0});
    EXPECT_THROW(ops::Add(p, 1.0f), std::runtime_error);
    EXPECT_THROW(ops::Abs(p), std::runtime_error);
}

// ---------- activations ----------
TEST_F(SyclBackendTest, Activations) {
    std::vector<float> v{-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
    Tensor t(v, {5});
    t.to_device(Device::SYCL);
    const Tensor& ct = t;

    Tensor relu = host_of(ops::Relu(ct));
    Tensor silu = host_of(ops::Silu(ct));
    Tensor tanh = host_of(ops::Tanh(ct));
    Tensor sig  = host_of(ops::Sigmoid(ct));
    for (size_t i = 0; i < v.size(); ++i) {
        const float x = v[i];
        EXPECT_NEAR(relu.at<float>({static_cast<int64_t>(i)}), std::max(0.0f, x), kEps);
        EXPECT_NEAR(silu.at<float>({static_cast<int64_t>(i)}), x / (1.0f + std::exp(-x)), kEps);
        EXPECT_NEAR(tanh.at<float>({static_cast<int64_t>(i)}), std::tanh(x), kEps);
        EXPECT_NEAR(sig.at<float>({static_cast<int64_t>(i)}), 1.0f / (1.0f + std::exp(-x)), kEps);
    }

    // In-place forms update the same tensor.
    ops::Sigmoid(t);
    const Tensor& it = t;
    Tensor inplace = host_of(it);
    EXPECT_NEAR(inplace.at<float>({0}), 1.0f / (1.0f + std::exp(2.0f)), kEps);
    EXPECT_NEAR(inplace.at<float>({4}), 1.0f / (1.0f + std::exp(-2.0f)), kEps);
}

TEST_F(SyclBackendTest, ActivationIntegerPromotesToFloat32) {
    std::vector<int32_t> v{-1, 0, 1};
    Tensor t(v, {3});
    t.to_device(Device::SYCL);
    const Tensor& ct = t;
    Tensor r = ops::Silu(ct);
    EXPECT_EQ(r.dtype(), DataType::FLOAT32);
    Tensor h = host_of(r);
    EXPECT_NEAR(h.at<float>({0}), -1.0f / (1.0f + std::exp(1.0f)), kEps);
    EXPECT_NEAR(h.at<float>({2}), 1.0f / (1.0f + std::exp(-1.0f)), kEps);
}

TEST_F(SyclBackendTest, SoftmaxMatchesReference) {
    std::vector<float> v{1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f};
    Tensor t(v, {2, 3});
    t.to_device(Device::SYCL);
    Tensor r = ops::Softmax(t, 1);
    EXPECT_EQ(r.dtype(), DataType::FLOAT32);
    Tensor h = host_of(r);

    const float e1 = std::exp(1.0f), e2 = std::exp(2.0f), e3 = std::exp(3.0f);
    const float denom = e1 + e2 + e3;
    EXPECT_NEAR(h.at<float>({0, 0}), e1 / denom, kEps);
    EXPECT_NEAR(h.at<float>({0, 1}), e2 / denom, kEps);
    EXPECT_NEAR(h.at<float>({0, 2}), e3 / denom, kEps);
    EXPECT_NEAR(h.at<float>({1, 0}), 1.0f / 3.0f, kEps);
    EXPECT_NEAR(h.at<float>({1, 1}), 1.0f / 3.0f, kEps);
    EXPECT_NEAR(h.at<float>({1, 2}), 1.0f / 3.0f, kEps);

    // Negative axis is normalized.
    Tensor r2 = host_of(ops::Softmax(t, -1));
    EXPECT_NEAR(r2.at<float>({0, 2}), e3 / denom, kEps);
}

TEST_F(SyclBackendTest, SoftmaxIntegerPromotesToFloat32) {
    std::vector<int32_t> v{0, 0, 0, 0};
    Tensor t(v, {2, 2});
    t.to_device(Device::SYCL);
    Tensor r = ops::Softmax(t, 0);
    EXPECT_EQ(r.dtype(), DataType::FLOAT32);
    Tensor h = host_of(r);
    EXPECT_NEAR(h.at<float>({0, 0}), 0.5f, kEps);
    EXPECT_NEAR(h.at<float>({1, 1}), 0.5f, kEps);
}

// ---------- reductions ----------
TEST_F(SyclBackendTest, GlobalReductions) {
    std::vector<float> v{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor t(v, {2, 3});
    t.to_device(Device::SYCL);
    EXPECT_NEAR(ops::Sum(t), 21.0f, kEps);
    EXPECT_NEAR(ops::Mean(t), 3.5f, kEps);
    EXPECT_NEAR(ops::Min(t), 1.0f, kEps);
    EXPECT_NEAR(ops::Max(t), 6.0f, kEps);
}

TEST_F(SyclBackendTest, GlobalReductionsOnNonContiguousView) {
    std::vector<float> v{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor t(v, {2, 3});
    t.to_device(Device::SYCL);
    Tensor p = t.permute({1, 0});  // 3x2 view, same six elements
    EXPECT_NEAR(ops::Sum(p), 21.0f, kEps);
    EXPECT_NEAR(ops::Max(p), 6.0f, kEps);
}

TEST_F(SyclBackendTest, AxisSumMeanMinMax) {
    std::vector<float> v{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor t(v, {2, 3});
    t.to_device(Device::SYCL);

    Tensor col = host_of(ops::Sum(t, 0));  // over rows -> {3}
    Tensor row = host_of(ops::Sum(t, 1));  // over cols -> {2}
    EXPECT_NEAR(col.at<float>({0}), 5.0f, kEps);
    EXPECT_NEAR(col.at<float>({1}), 7.0f, kEps);
    EXPECT_NEAR(col.at<float>({2}), 9.0f, kEps);
    EXPECT_NEAR(row.at<float>({0}), 6.0f, kEps);
    EXPECT_NEAR(row.at<float>({1}), 15.0f, kEps);

    Tensor mean = ops::Mean(t, 1);
    EXPECT_EQ(mean.dtype(), DataType::FLOAT32);
    Tensor mh = host_of(mean);
    EXPECT_NEAR(mh.at<float>({0}), 2.0f, kEps);
    EXPECT_NEAR(mh.at<float>({1}), 5.0f, kEps);

    Tensor mn = host_of(ops::Min(t, 0));
    Tensor mx = host_of(ops::Max(t, 0));
    EXPECT_NEAR(mn.at<float>({2}), 3.0f, kEps);
    EXPECT_NEAR(mx.at<float>({2}), 6.0f, kEps);
}

TEST_F(SyclBackendTest, AxisSumWidensIntegerAccumulator) {
    std::vector<int8_t> v{1, 2, 3, 4};
    Tensor t(v, {2, 2});
    t.to_device(Device::SYCL);
    Tensor r = ops::Sum(t, 1);
    EXPECT_EQ(r.dtype(), DataType::INT32);
    Tensor h = host_of(r);
    EXPECT_EQ(h.at<int32_t>({0}), 3);
    EXPECT_EQ(h.at<int32_t>({1}), 7);
}

TEST_F(SyclBackendTest, AxisMinPreservesIntegerDtype) {
    std::vector<int32_t> v{5, 2, 9, 1};
    Tensor t(v, {2, 2});
    t.to_device(Device::SYCL);
    Tensor r = ops::Min(t, 0);
    EXPECT_EQ(r.dtype(), DataType::INT32);
    Tensor h = host_of(r);
    // rows are [5, 2] and [9, 1]; the axis-0 minimum per column is [5, 1].
    EXPECT_EQ(h.at<int32_t>({0}), 5);
    EXPECT_EQ(h.at<int32_t>({1}), 1);
}

TEST_F(SyclBackendTest, ReductionsPropagateNan) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> v{1.0f, nan, 3.0f};
    Tensor t(v, {3});
    t.to_device(Device::SYCL);
    EXPECT_TRUE(std::isnan(ops::Min(t)));
    EXPECT_TRUE(std::isnan(ops::Max(t)));
    Tensor axis = host_of(ops::Max(t, 0));
    EXPECT_TRUE(std::isnan(axis.at<float>({0})));
}

TEST_F(SyclBackendTest, ArgMaxArgMin) {
    std::vector<float> v{1.0f, 5.0f, 5.0f, 2.0f, 0.0f, 7.0f};
    Tensor t(v, {2, 3});
    t.to_device(Device::SYCL);

    Tensor am = ops::Argmax(t, 1);
    EXPECT_EQ(am.dtype(), DataType::INT32);
    Tensor ah = host_of(am);
    // Ties keep the smaller index (first 5 is index 1, not 2).
    EXPECT_EQ(ah.at<int32_t>({0}), 1);
    EXPECT_EQ(ah.at<int32_t>({1}), 2);

    Tensor in = host_of(ops::Argmin(t, 0));
    // Columns are (1,2), (5,0), (5,7): the minima sit at rows 0, 1, 0.
    EXPECT_EQ(in.at<int32_t>({0}), 0);
    EXPECT_EQ(in.at<int32_t>({1}), 1);
    EXPECT_EQ(in.at<int32_t>({2}), 0);
}

TEST_F(SyclBackendTest, ArgMaxFirstNanWins) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> v{1.0f, nan, 3.0f, nan};
    Tensor t(v, {2, 2});
    t.to_device(Device::SYCL);
    Tensor h = host_of(ops::Argmax(t, 1));
    EXPECT_EQ(h.at<int32_t>({0}), 1);
    EXPECT_EQ(h.at<int32_t>({1}), 1);
    Tensor hmin = host_of(ops::Argmin(t, 1));
    EXPECT_EQ(hmin.at<int32_t>({0}), 1);
}

TEST_F(SyclBackendTest, AllAnyAndNonzero) {
    std::vector<float> v{1.0f, 1.0f, 1.000001f, 2.0f};
    Tensor t(v, {2, 2});
    t.to_device(Device::SYCL);
    EXPECT_TRUE(ops::All(t, 1.0f) == false);  // 2.0 does not match
    EXPECT_TRUE(ops::Any(t, 2.0f));

    std::vector<float> ones(4, 1.0f);
    Tensor o(ones, {2, 2});
    o.to_device(Device::SYCL);
    EXPECT_TRUE(ops::All(o, 1.0f));
    EXPECT_FALSE(ops::Any(o, 2.0f));

    std::vector<float> mixed{0.0f, 1.5f, 0.0f, -2.0f};
    Tensor m(mixed, {4});
    m.to_device(Device::SYCL);
    EXPECT_EQ(ops::Nonzero(m), 2u);
}

// ---------- comparison / logical ----------
TEST_F(SyclBackendTest, ComparisonsBroadcast) {
    std::vector<float> av{1, 2, 3, 4, 5, 6};
    std::vector<float> bv{1, 0, 7};
    Tensor a(av, {2, 3});
    Tensor b(bv, {3});
    a.to_device(Device::SYCL);
    b.to_device(Device::SYCL);
    const Tensor& ac = a;
    const Tensor& bc = b;

    Tensor eq = ops::Equal(ac, bc);
    EXPECT_EQ(eq.dtype(), DataType::INT8);
    Tensor eqh = host_of(eq);
    EXPECT_EQ(eqh.at<int8_t>({0, 0}), 1);
    EXPECT_EQ(eqh.at<int8_t>({0, 1}), 0);
    EXPECT_EQ(eqh.at<int8_t>({1, 2}), 0);

    Tensor gt = host_of(ops::Greater(ac, bc));
    EXPECT_EQ(gt.at<int8_t>({0, 0}), 0);  // 1 > 1 is false
    EXPECT_EQ(gt.at<int8_t>({0, 1}), 1);  // 2 > 0
    EXPECT_EQ(gt.at<int8_t>({1, 2}), 0);  // 6 > 7 is false

    Tensor le = host_of(ops::LessEqual(ac, bc));
    EXPECT_EQ(le.at<int8_t>({0, 0}), 1);
    EXPECT_EQ(le.at<int8_t>({1, 2}), 1);
}

TEST_F(SyclBackendTest, ScalarComparisonsAndTolerance) {
    std::vector<float> v{1.0f, 2.0f, 3.0f, 1.0f + 1e-6f};
    Tensor t(v, {4});
    t.to_device(Device::SYCL);
    const Tensor& tc = t;

    Tensor eq = host_of(ops::Equal(tc, 1.0f));
    EXPECT_EQ(eq.at<int8_t>({0}), 1);
    EXPECT_EQ(eq.at<int8_t>({1}), 0);
    // Equal uses the FLOAT32 tier (1e-5), so 1 + 1e-6 still matches.
    EXPECT_EQ(eq.at<int8_t>({3}), 1);

    Tensor lt = host_of(ops::Less(tc, 2.0f));
    EXPECT_EQ(lt.at<int8_t>({0}), 1);
    EXPECT_EQ(lt.at<int8_t>({2}), 0);
}

// ---------- shape / data movement ----------
TEST_F(SyclBackendTest, ConcatAxis0And1) {
    std::vector<float> av{1, 2, 3, 4};
    std::vector<float> bv{5, 6, 7, 8};
    Tensor a(av, {2, 2});
    Tensor b(bv, {2, 2});
    a.to_device(Device::SYCL);
    b.to_device(Device::SYCL);

    Tensor c0 = host_of(ops::Concat({a, b}, 0));
    EXPECT_EQ(c0.shape(0), 4);
    EXPECT_EQ(c0.shape(1), 2);
    EXPECT_NEAR(c0.at<float>({0, 0}), 1.0f, kEps);
    EXPECT_NEAR(c0.at<float>({2, 0}), 5.0f, kEps);
    EXPECT_NEAR(c0.at<float>({3, 1}), 8.0f, kEps);

    Tensor c1 = host_of(ops::Concat({a, b}, 1));
    EXPECT_EQ(c1.shape(0), 2);
    EXPECT_EQ(c1.shape(1), 4);
    EXPECT_NEAR(c1.at<float>({0, 1}), 2.0f, kEps);
    EXPECT_NEAR(c1.at<float>({0, 2}), 5.0f, kEps);
    EXPECT_NEAR(c1.at<float>({1, 3}), 8.0f, kEps);
}

TEST_F(SyclBackendTest, SliceRowsAndCols) {
    std::vector<float> v{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    Tensor t(v, {3, 4});
    t.to_device(Device::SYCL);
    const Tensor& tc = t;

    Tensor s = host_of(ops::Slice(tc, {{1, 3}, {0, 2}}));
    EXPECT_EQ(s.shape(0), 2);
    EXPECT_EQ(s.shape(1), 2);
    EXPECT_NEAR(s.at<float>({0, 0}), 4.0f, kEps);
    EXPECT_NEAR(s.at<float>({0, 1}), 5.0f, kEps);
    EXPECT_NEAR(s.at<float>({1, 0}), 8.0f, kEps);
    EXPECT_NEAR(s.at<float>({1, 1}), 9.0f, kEps);
}

TEST_F(SyclBackendTest, Transpose2DInPlace) {
    std::vector<float> v{1, 2, 3, 4, 5, 6};
    Tensor t(v, {2, 3});
    t.to_device(Device::SYCL);
    ops::Transpose(t);
    EXPECT_EQ(t.shape(0), 3);
    EXPECT_EQ(t.shape(1), 2);
    Tensor h = host_of(t);
    EXPECT_NEAR(h.at<float>({0, 0}), 1.0f, kEps);
    EXPECT_NEAR(h.at<float>({0, 1}), 4.0f, kEps);
    EXPECT_NEAR(h.at<float>({2, 0}), 3.0f, kEps);
    EXPECT_NEAR(h.at<float>({2, 1}), 6.0f, kEps);
}

TEST_F(SyclBackendTest, TransposeNDWithAxes) {
    std::vector<float> v(24);
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i);
    Tensor t(v, {2, 3, 4});
    t.to_device(Device::SYCL);
    const Tensor& tc = t;

    // axes {2, 0, 1}: output shape is {4, 2, 3}.
    Tensor r = ops::Transpose(tc, {2, 0, 1});
    EXPECT_EQ(r.shape(0), 4);
    EXPECT_EQ(r.shape(1), 2);
    EXPECT_EQ(r.shape(2), 3);
    Tensor h = host_of(r);
    for (int64_t i = 0; i < 2; ++i)
        for (int64_t j = 0; j < 3; ++j)
            for (int64_t k = 0; k < 4; ++k)
                EXPECT_NEAR(h.at<float>({k, i, j}), static_cast<float>(i * 12 + j * 4 + k), kEps);
}

TEST_F(SyclBackendTest, TransposeIntoDst) {
    std::vector<float> v{1, 2, 3, 4, 5, 6};
    Tensor t(v, {2, 3});
    t.to_device(Device::SYCL);
    Tensor dst = Tensor::Zeros({3, 2}, DataType::FLOAT32, Device::SYCL);
    ops::Transpose(t, dst, {1, 0});
    Tensor h = host_of(dst);
    EXPECT_NEAR(h.at<float>({0, 0}), 1.0f, kEps);
    EXPECT_NEAR(h.at<float>({1, 0}), 2.0f, kEps);
    EXPECT_NEAR(h.at<float>({2, 1}), 6.0f, kEps);
}

// ---------- in-place fill + typecast ----------
TEST_F(SyclBackendTest, InPlaceFillZerosOnes) {
    Tensor t = Tensor::Random({3, 4}, -1.0f, 1.0f, DataType::FLOAT32, Device::SYCL);
    ops::Fill(t, 2.5f);
    Tensor h = host_of(t);
    for (int64_t i = 0; i < 3; ++i)
        for (int64_t j = 0; j < 4; ++j)
            EXPECT_NEAR(h.at<float>({i, j}), 2.5f, kEps);

    ops::Zeros(t);
    h = host_of(t);
    EXPECT_NEAR(h.at<float>({0, 0}), 0.0f, kEps);
    EXPECT_NEAR(h.at<float>({2, 3}), 0.0f, kEps);

    ops::Ones(t);
    h = host_of(t);
    EXPECT_NEAR(h.at<float>({1, 2}), 1.0f, kEps);

    // Integer tensors take the same path.
    Tensor i = Tensor::Zeros({4}, DataType::INT32, Device::SYCL);
    ops::Fill(i, 7.0f);
    Tensor ih = host_of(i);
    EXPECT_EQ(ih.at<int32_t>({3}), 7);
}

TEST_F(SyclBackendTest, InPlaceFillOnOffsetView) {
    std::vector<float> v{0, 1, 2, 3, 4, 5, 6, 7};
    Tensor base(v, {2, 4});
    base.to_device(Device::SYCL);
    // Row 1 is a contiguous view whose origin has a non-zero offset.
    Tensor view = base.slice({{1, 2}, {0, 4}});
    EXPECT_TRUE(view.is_contiguous());
    ops::Fill(view, 9.0f);
    Tensor h = host_of(base);
    EXPECT_NEAR(h.at<float>({0, 0}), 0.0f, kEps);
    EXPECT_NEAR(h.at<float>({1, 0}), 9.0f, kEps);
    EXPECT_NEAR(h.at<float>({1, 3}), 9.0f, kEps);
}

TEST_F(SyclBackendTest, TypecastRoundTrip) {
    std::vector<float> v{0.5f, -1.25f, 4.0f, 12.75f};
    Tensor t(v, {4});
    t.to_device(Device::SYCL);
    const Tensor& tc = t;

    Tensor f16 = ops::Typecast(tc, DataType::FLOAT16);
    EXPECT_EQ(f16.dtype(), DataType::FLOAT16);
    EXPECT_EQ(f16.device(), Device::SYCL);
    const Tensor& f16c = f16;
    Tensor back = host_of(ops::Typecast(f16c, DataType::FLOAT32));
    for (int64_t i = 0; i < 4; ++i)
        EXPECT_NEAR(back.at<float>({i}), v[static_cast<size_t>(i)], 1e-2f);

    Tensor i32 = host_of(ops::Typecast(tc, DataType::INT32));
    EXPECT_EQ(i32.dtype(), DataType::INT32);
    EXPECT_EQ(i32.at<int32_t>({0}), 0);
    EXPECT_EQ(i32.at<int32_t>({1}), -1);
    EXPECT_EQ(i32.at<int32_t>({3}), 12);
}

TEST_F(SyclBackendTest, TypecastInPlace) {
    Tensor t = Tensor::Fill({3}, 2.5f, DataType::FLOAT32, Device::SYCL);
    ops::Typecast(t, DataType::FLOAT64);
    EXPECT_EQ(t.dtype(), DataType::FLOAT64);
    Tensor h = host_of(t);
    EXPECT_NEAR(h.at<double>({2}), 2.5, 1e-12);
    ops::Typecast(t, DataType::FLOAT64);  // same dtype: no-op
    EXPECT_EQ(t.dtype(), DataType::FLOAT64);
}

TEST_F(SyclBackendTest, RandomOtherDtypesStayInRange) {
    Tensor i = host_of(Tensor::Random({32}, 0.0f, 10.0f, DataType::INT32, Device::SYCL));
    for (int64_t n = 0; n < 32; ++n) {
        const int32_t v = i.at<int32_t>({n});
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 10);
    }
    Tensor h = host_of(Tensor::Random({32}, -1.0f, 1.0f, DataType::FLOAT16, Device::SYCL));
    EXPECT_EQ(h.dtype(), DataType::FLOAT16);
    const Tensor& hc = h;
    Tensor f32 = host_of(ops::Typecast(hc, DataType::FLOAT32));
    for (int64_t n = 0; n < 32; ++n) {
        EXPECT_GE(f32.at<float>({n}), -1.0f);
        EXPECT_LT(f32.at<float>({n}), 1.0f);
    }
}

// ---------- unimplemented operators fail loudly ----------
TEST_F(SyclBackendTest, UnimplementedOperatorThrows) {
    Tensor a = Tensor::Ones({2, 2}, DataType::FLOAT32, Device::SYCL);
    Tensor w = Tensor::Ones({2}, DataType::FLOAT32, Device::SYCL);
    Tensor b = Tensor::Zeros({2}, DataType::FLOAT32, Device::SYCL);
    // LayerNorm is outside the ported operator set; it must report SYCL as
    // unimplemented rather than silently running on the CPU.
    EXPECT_THROW((void)ops::LayerNorm(a, w, b, 1), std::runtime_error);
}
