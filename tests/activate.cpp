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

void test(DataType type){
    Tensor temp = Tensor::Random({3,1000,1000}, 0,10, type);
    // temp.to_host();
    // RUNNING_TIME(ops::Softmax(static_cast<const Tensor&>(temp), 0));
    auto res = ops::Softmax(static_cast<const Tensor&>(temp), 0);
    ops::println(res);
}
int main() {
    for (auto type : GetAllDataTypes()) {
        test(type);
    }
    return 0;
}