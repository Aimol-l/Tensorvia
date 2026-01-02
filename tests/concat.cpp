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

    for (auto type : GetAllDataTypes()) {
        Tensor a = Tensor::Random({2,4,3},-10,10,type);
        Tensor b = Tensor::Random({3,4,3},-10,10,type);
    
        ops::println(a);
        ops::println(b);
        ops::println(ops::Concat({a,b},0));
        LOG_INFO("Data type: " << dtype_to_string(type));
    }
    return 0;
}