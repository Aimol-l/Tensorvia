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
    for(auto& dtype : GetAllDataTypes()) {
        Tensor a = Tensor::Random({10,1024,1024},-10,10,dtype);
        Tensor b = Tensor::Random({10,1024,1024},-10,10,dtype);
        RUNNING_TIME(ops::Mul(a,b));
    }
    return 0;
}