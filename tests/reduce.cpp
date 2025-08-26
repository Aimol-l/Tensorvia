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
        Tensor temp = Tensor::Random({3,3},-10,10,dtype);
        ops::println(temp);
        LOG_INFO("res: " << ops::Any(static_cast<const Tensor&>(temp), 0));
        // ops::println(ops::Argmax(temp, 1));
        LOG_INFO("Data type: " << dtype_to_string(dtype));
    }
    return 0;
}