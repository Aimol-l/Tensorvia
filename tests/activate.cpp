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
    for (auto type : GetAllDataTypes()) {
        Tensor temp = Tensor::Random({5,5},-10,10,type);
        ops::println(temp);
        ops::println(ops::Softmax(static_cast<const Tensor&>(temp),1));
        LOG_INFO("Data type: " << dtype_to_string(type));
    }
    return 0;
}