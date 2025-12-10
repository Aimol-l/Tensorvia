#include "ops.h"
#include <print>

int main() {
    Tensor a = Tensor::Fill({1,100,100},2,DataType::FLOAT32); // 0
    Tensor b = Tensor::Fill({1,100,100},2,DataType::FLOAT32); // 10000
    Tensor c = Tensor::Fill({1,100,100},2,DataType::FLOAT32); // 20000
    Tensor d = Tensor::Fill({1,100,100},2,DataType::FLOAT32); // 30000

    auto res = ops::Concat({a,b,c,d},0);
    

    return 0;
}