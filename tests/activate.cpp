#include <print>
#include <iostream>
#include "ops.h"


// 获取所有 DataType 值的函数
std::vector<DataType> GetAllDataTypes() {
    return {
        DataType::INT8, DataType::INT16, DataType::INT32,DataType::INT64,
        DataType::BFLOAT16, DataType::FLOAT16,DataType::FLOAT32, DataType::FLOAT64
    };
}

int main() {
    // relu silu tanh sigmoid softmax
    for (auto type : GetAllDataTypes()) {
        auto temp = Tensor::Random({700, 700},-10,10,type);
        ops::println(temp);
        ops::println(ops::Relu(static_cast<const Tensor&>(temp)));
        ops::println(ops::Silu(static_cast<const Tensor&>(temp)));
        ops::println(ops::Sigmoid(static_cast<const Tensor&>(temp)));
        ops::println(ops::Softmax(static_cast<const Tensor&>(temp),0));
        LOG_INFO("----------------------------------------------------------------");
    }
    return 0;
}