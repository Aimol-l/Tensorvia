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
import tensorvia;

int main() {
    // 热循环里复用同一个 Tensor 容器，shape+dtype 不变时零分配
    Tensor buf;
    const int N = 5;

    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        buf.reset({256}, DataType::FLOAT32);  // 复用容量，避免反复 malloc
        ops::Zeros(buf);                       // in-place 填零
        ops::Add(buf, static_cast<float>(i));  // in-place 全加 i
        acc += ops::Sum(buf);                  // 每次 sum = 256 * i
    }

    ops::println("acc      = {}", acc);
    ops::println("expected = {}", 10.0f * 256.0f);  // 256*(0+1+2+3+4)

    return 0;
}
