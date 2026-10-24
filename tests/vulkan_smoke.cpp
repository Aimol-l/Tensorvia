#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <exception>
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

// The machine may have no ICD (or no Vulkan loader at all): probing with a
// one-element VULKAN tensor is the only reliable check, so every test skips
// when the probe fails.
class VulkanBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        try {
            Tensor probe = Tensor::Fill({1}, 0.0f, DataType::FLOAT32, Device::VULKAN);
            (void)probe;
        } catch (const std::exception& e) {
            GTEST_SKIP() << "no usable Vulkan device: " << e.what();
        }
    }
};

// ---------- creation + host round trip ----------
TEST_F(VulkanBackendTest, CreationAndRoundTrip) {
    Tensor a = Tensor::Fill({2, 3}, 2.0f, DataType::FLOAT32, Device::VULKAN);
    EXPECT_EQ(a.device(), Device::VULKAN);
    EXPECT_EQ(a.numel(), 6u);
    Tensor h = host_of(a);
    EXPECT_EQ(h.device(), Device::CPU);
    EXPECT_NEAR(h.at<float>({0, 0}), 2.0f, kEps);
    EXPECT_NEAR(h.at<float>({1, 2}), 2.0f, kEps);

    // CPU -> VULKAN -> CPU keeps the values.
    std::vector<float> v{1, 2, 3, 4};
    Tensor c(v, {2, 2});
    c.to_device(Device::VULKAN);
    EXPECT_EQ(c.device(), Device::VULKAN);
    c.to_host();
    EXPECT_EQ(c.device(), Device::CPU);
    EXPECT_NEAR(c.at<float>({0, 1}), 2.0f, kEps);
    EXPECT_NEAR(c.at<float>({1, 1}), 4.0f, kEps);
}

// ---------- broadcasting arithmetic ----------
TEST_F(VulkanBackendTest, BroadcastingArithmetic) {
    Tensor a = Tensor::Fill({3, 1}, 1.0f, DataType::FLOAT32, Device::VULKAN);
    Tensor b = Tensor::Fill({1, 4}, 10.0f, DataType::FLOAT32, Device::VULKAN);
    Tensor c = a + b;
    EXPECT_EQ(c.shape(0), 3);
    EXPECT_EQ(c.shape(1), 4);
    EXPECT_EQ(c.device(), Device::VULKAN);
    Tensor h = host_of(c);
    EXPECT_NEAR(h.at<float>({0, 0}), 11.0f, kEps);
    EXPECT_NEAR(h.at<float>({2, 3}), 11.0f, kEps);

    Tensor d = c * 2.0f;
    Tensor hd = host_of(d);
    EXPECT_NEAR(hd.at<float>({1, 1}), 22.0f, kEps);

    // dst variant writes into an existing VULKAN tensor.
    Tensor dst({3, 4}, DataType::FLOAT32, Device::VULKAN);
    ops::Sub(a, b, dst);
    Tensor hdst = host_of(dst);
    EXPECT_NEAR(hdst.at<float>({0, 0}), -9.0f, kEps);
    EXPECT_NEAR(hdst.at<float>({2, 2}), -9.0f, kEps);
}

// ---------- unary / activation ----------
TEST_F(VulkanBackendTest, UnaryAndActivation) {
    std::vector<float> v{-1.0f, 0.0f, 2.0f};
    Tensor t(v, {3});
    t.to_device(Device::VULKAN);
    const Tensor& ct = t;  // pick the out-of-place overload (Tensor& means in-place)
    Tensor h = host_of(ops::Relu(ct));
    EXPECT_NEAR(h.at<float>({0}), 0.0f, kEps);
    EXPECT_NEAR(h.at<float>({1}), 0.0f, kEps);
    EXPECT_NEAR(h.at<float>({2}), 2.0f, kEps);

    Tensor s = host_of(ops::Softmax(t, 0));
    EXPECT_NEAR(s.at<float>({0}) + s.at<float>({1}) + s.at<float>({2}), 1.0f, kEps);
    EXPECT_GT(s.at<float>({2}), s.at<float>({1}));
}

// ---------- reductions ----------
TEST_F(VulkanBackendTest, Reductions) {
    Tensor ones = Tensor::Ones({2, 3}, DataType::FLOAT32, Device::VULKAN);
    EXPECT_NEAR(ops::Sum(ones), 6.0f, kEps);
    EXPECT_NEAR(ops::Mean(ones), 1.0f, kEps);
    EXPECT_NEAR(ops::Max(ones), 1.0f, kEps);

    std::vector<float> v{1, 2, 3, 4, 5, 6};
    Tensor t(v, {2, 3});
    t.to_device(Device::VULKAN);
    Tensor axis1 = host_of(ops::Sum(t, 1));
    EXPECT_EQ(axis1.numel(), 2u);
    EXPECT_NEAR(axis1.at<float>({0}), 6.0f, kEps);
    EXPECT_NEAR(axis1.at<float>({1}), 15.0f, kEps);

    Tensor max0 = host_of(ops::Max(t, 0));
    EXPECT_EQ(max0.numel(), 3u);
    EXPECT_NEAR(max0.at<float>({0}), 4.0f, kEps);
    EXPECT_NEAR(max0.at<float>({2}), 6.0f, kEps);
}

// ---------- matmul ----------
TEST_F(VulkanBackendTest, MatMul) {
    Tensor a = Tensor::Fill({2, 3}, 1.0f, DataType::FLOAT32, Device::VULKAN);
    Tensor b = Tensor::Fill({3, 2}, 2.0f, DataType::FLOAT32, Device::VULKAN);
    Tensor c = ops::MatMul(a, b);
    EXPECT_EQ(c.shape(0), 2);
    EXPECT_EQ(c.shape(1), 2);
    Tensor h = host_of(c);
    EXPECT_NEAR(h.at<float>({0, 0}), 6.0f, kEps);
    EXPECT_NEAR(h.at<float>({1, 1}), 6.0f, kEps);
}

// ---------- concat ----------
TEST_F(VulkanBackendTest, Concat) {
    Tensor a = Tensor::Fill({2, 2}, 1.0f, DataType::FLOAT32, Device::VULKAN);
    Tensor b = Tensor::Fill({2, 2}, 2.0f, DataType::FLOAT32, Device::VULKAN);
    Tensor c = ops::Concat({a, b}, 0);
    EXPECT_EQ(c.shape(0), 4);
    EXPECT_EQ(c.shape(1), 2);
    Tensor h = host_of(c);
    EXPECT_NEAR(h.at<float>({0, 0}), 1.0f, kEps);
    EXPECT_NEAR(h.at<float>({3, 1}), 2.0f, kEps);
}

// ---------- views: slice + transpose + to_host materialization ----------
TEST_F(VulkanBackendTest, Views) {
    std::vector<float> v{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    Tensor t(v, {3, 4});
    t.to_device(Device::VULKAN);

    Tensor view = t.slice({{1, 3}, {0, 2}});
    EXPECT_EQ(view.device(), Device::VULKAN);
    EXPECT_NE(view.offset(), 0);
    Tensor hv = host_of(view);
    EXPECT_EQ(hv.shape(0), 2);
    EXPECT_EQ(hv.shape(1), 2);
    EXPECT_NEAR(hv.at<float>({0, 0}), 4.0f, kEps);
    EXPECT_NEAR(hv.at<float>({0, 1}), 5.0f, kEps);
    EXPECT_NEAR(hv.at<float>({1, 0}), 8.0f, kEps);
    EXPECT_NEAR(hv.at<float>({1, 1}), 9.0f, kEps);

    Tensor tr = ops::Transpose(t, {1, 0});
    EXPECT_EQ(tr.shape(0), 4);
    EXPECT_EQ(tr.shape(1), 3);
    Tensor ht = host_of(tr);
    EXPECT_NEAR(ht.at<float>({0, 1}), 4.0f, kEps);
    EXPECT_NEAR(ht.at<float>({2, 2}), 10.0f, kEps);

    Tensor base = Tensor::Fill({3, 4}, 1.0f, DataType::FLOAT32, Device::VULKAN);
    Tensor sliced = base.slice({{1, 3}, {2, 4}});
    EXPECT_NEAR(ops::Sum(sliced), 4.0f, kEps);
}

// ---------- logical + nonzero ----------
TEST_F(VulkanBackendTest, LogicalAndNonzero) {
    std::vector<float> v{0, 1, 0, 2, 3};
    Tensor t(v, {5});
    t.to_device(Device::VULKAN);

    Tensor mask = ops::Greater(t, 0.5f);
    EXPECT_EQ(mask.dtype(), DataType::INT8);
    Tensor hm = host_of(mask);
    EXPECT_EQ(hm.at<std::int8_t>({0}), 0);
    EXPECT_EQ(hm.at<std::int8_t>({1}), 1);
    EXPECT_EQ(hm.at<std::int8_t>({4}), 1);

    EXPECT_EQ(ops::Nonzero(t), 3u);
}

// ---------- typecast ----------
TEST_F(VulkanBackendTest, Typecast) {
    Tensor a = Tensor::Fill({2}, 1.5f, DataType::FLOAT32, Device::VULKAN);
    Tensor d = a.to_type(DataType::FLOAT64);
    EXPECT_EQ(d.dtype(), DataType::FLOAT64);
    EXPECT_EQ(a.dtype(), DataType::FLOAT32);  // to_type must not mutate
    Tensor h = host_of(d);
    EXPECT_NEAR(static_cast<float>(h.at<double>({0})), 1.5f, kEps);
}

// ---------- CPU-only ops reject device tensors instead of silently falling back ----------
TEST_F(VulkanBackendTest, CpuOnlyOpRejected) {
    Tensor a = Tensor::Fill({2}, 1.0f, DataType::FLOAT32, Device::VULKAN);
    EXPECT_THROW(ops::Erf(a), std::exception);
}
