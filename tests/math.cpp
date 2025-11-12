#include "ops.h"
#include <print>


int main() {

    Tensor a = Tensor::Random({3,2048,2592},-10,10,DataType::FLOAT32);
    Tensor b = Tensor::Random({3,2048,2592},-10,10,DataType::FLOAT32);
    Tensor result(a.shape(), a.dtype(), Device::CPU);

    auto start = std::chrono::high_resolution_clock::now();
    // for(int i = 0;i<1000;i++)
        // ops::Relu(a);
    auto res = ops::Concat({a,b},0);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

    ops::println(res);
    std::println("avg times = {}us",duration);

    return 0;
}