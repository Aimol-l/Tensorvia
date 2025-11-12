#include <print>
#include <iostream>

#include "ops.h"

// 获取所有 DataType 值的函数
std::vector<DataType> GetAllDataTypes() {
    return {
        DataType::INT8, DataType::INT16, DataType::INT32,
        DataType::INT64, DataType::BFLOAT16, DataType::FLOAT16,
        DataType::FLOAT32, DataType::FLOAT64
    };
}


int main() {

    Tensor a = Tensor::Random({3,2000,2500},-10,10,DataType::FLOAT32);
    Tensor res = Tensor::Random({3,2000,2500},-10,10,DataType::FLOAT32);


    ops::println(a);
    auto start = std::chrono::high_resolution_clock::now();

    ops::Transpose(a,res,{0,2,1});

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    
    ops::println(res);

    std::println("avg times = {}us",duration);
    return 0;
}