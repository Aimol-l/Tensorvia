#include "ops.h"
#include <print>
using namespace via;

int main() {
    Tensor a = Tensor::Random({1,7,7},0,10,DataType::FLOAT32); // 0
    Tensor b = Tensor::Fill({1,7,7},2,DataType::FLOAT32); // 10000
    Tensor c = Tensor::Fill({1,7,7},3,DataType::FLOAT32); // 20000
    Tensor d = Tensor::Fill({1,7,7},4,DataType::FLOAT32); // 30000

    auto res = ops::Concat({a,b,c,d},0);

    ops::println(res);

    return 0;
}                       