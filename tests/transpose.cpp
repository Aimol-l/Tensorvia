#include <print>
#include <iostream>

#include "ops.h"
using namespace via;

// 获取所有 DataType 值的函数
std::vector<DataType> GetAllDataTypes() {
    return {
        DataType::INT8, DataType::INT16, DataType::INT32,
        DataType::INT64, DataType::BFLOAT16, DataType::FLOAT16,
        DataType::FLOAT32, DataType::FLOAT64
    };
}


int main() {

    Tensor a = Tensor::Random({2000,2500},-10,10,DataType::FLOAT32);
    auto b = a.slice({{1,800},{1,500}});
    ops::println(b);

    return 0;
}