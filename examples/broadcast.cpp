#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <format>
#include <print>
#include <string>
#include <string_view>
#include <vector>
#include "interface/tensorvia.hpp"

int main() {
    // 1. 广播：(3,1) + (4,) -> (3,4)，out[i][j] = col[i] + row[j]
    auto col = Tensor::Fill({3, 1}, 1.0f, DataType::FLOAT32);
    auto row = ops::Arange(1.0f, 5.0f);  // [1,2,3,4]
    auto bc = col + row;
    ops::println("(3,1) + (4,) =");
    ops::println(bc);

    // 2. 混合 dtype：结果自动提升
    auto f32 = Tensor::Fill({4}, 1.0f, DataType::FLOAT32);
    auto f64 = Tensor::Fill({4}, 0.5, DataType::FLOAT64);
    auto mixed = f32 + f64;              // FLOAT32 + FLOAT64 -> FLOAT64
    ops::println("f32 + f64[0] = {}", mixed.at<double>({0}));  // 1.5

    std::vector<std::int32_t> iv{2, 2, 2, 2};
    Tensor i32(iv, {4});
    auto iw = i32 + f32;                 // INT32 + FLOAT32 -> FLOAT32
    ops::println("i32 + f32[0] = {}", iw.at<float>({0}));      // 3.0

    // 3. 比较（返回 INT8 掩码）+ where 选择
    auto mask = row > 2.0f;              // [0,0,1,1]
    auto picked = ops::Where(mask, row, 0.0f);  // [0,0,3,4]
    ops::println("where(row>2, row, 0) =");
    ops::println(picked);

    return 0;
}
