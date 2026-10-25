// CMake 路径的头文件形态冒烟测试。
//
// tests/ 下其余文件是 import 形态（`import tensorvia;`），服务 mcpp 路径的
// 行为测试（mcpp test）；CMake 没有模块支持，这里用安装面同款的
// #include <tensorvia.hpp> 验证 CMake 产物与安装头树的行为闭环。
#include <gtest/gtest.h>

#include <tensorvia.hpp>

TEST(HeaderSurface, CpuArithmeticAndSoftmax) {
    auto a = Tensor::Ones({2, 3}, DataType::FLOAT32);
    auto b = Tensor::Fill({2, 3}, 2.0f, DataType::FLOAT32);
    EXPECT_NEAR(ops::Sum(a + b), 18.0f, 1e-4f);

    auto v = Tensor::Fill({4}, 1.0f, DataType::FLOAT32);
    EXPECT_NEAR(ops::Sum(ops::Softmax(v, 0)), 1.0f, 1e-4f);
}
