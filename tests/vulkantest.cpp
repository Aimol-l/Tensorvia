#include "ops.h"
#include <print>

int main() {
    Tensor a = Tensor::Fill({3,2592,2048},2,DataType::FLOAT32);
    Tensor b = Tensor::Fill({3,2048,2592},4,DataType::FLOAT32);

    auto c = ops::Mul(a,b);

    ops::println(c);
    
    return 0;
}