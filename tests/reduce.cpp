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
    for(auto& dtype : GetAllDataTypes()) {
        Tensor temp = Tensor::Random({3000,3000},-10,10,dtype);
        RUNNING_TIME(ops::Min(static_cast<const Tensor&>(temp)));
        RUNNING_TIME(ops::Max(static_cast<const Tensor&>(temp)));
        RUNNING_TIME(ops::Mean(static_cast<const Tensor&>(temp)));
        RUNNING_TIME(ops::Sum(static_cast<const Tensor&>(temp)));
    }
    return 0;
}