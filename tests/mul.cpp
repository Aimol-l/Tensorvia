#include <print>
#include <iostream>
#include "ops.h"
using namespace via;


// 获取所有 DataType 值的函数
std::vector<DataType> GetAllDataTypes() {
    return {
        DataType::INT8, DataType::INT16, DataType::INT32,DataType::INT64,
        DataType::BFLOAT16, DataType::FLOAT16,DataType::FLOAT32, DataType::FLOAT64
    };
}


int main() {
    Tensor a = Tensor::Random({2592,2048},-10,10,DataType::FLOAT32);
    Tensor b = Tensor::Random({2048,4096},-10,10,DataType::FLOAT32);

    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < 100; i++){
        RUNNING_TIME(ops::Mul(a,b));
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 100;

    std::println("avg times = {}ms",duration);

    return 0;
}