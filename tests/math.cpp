#include "ops.h"
#include <format>

int main() {
    Tensor a = Tensor::Random({1000,1000},-10,10,DataType::FLOAT32);
    Tensor b = Tensor::Random({1000,1000},-10,10,DataType::FLOAT32);
    Tensor result(a.shape(), a.dtype(), Device::CPU);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0;i<10000;i++)
        ops::Add(a,b,result);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 10000;
    std::cout<<std::format("total time = {}us", duration)<<std::endl;
    return 0;
}