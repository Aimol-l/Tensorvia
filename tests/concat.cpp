#include <print>
#include <iostream>
#include "ops.h"

int main() {
    Tensor a = Tensor::Random({5,5,6},0,10,DataType::FLOAT32);
    Tensor b = Tensor::Random({3,5,6},0,10,DataType::FLOAT32);

    auto c = ops::concat({a,b},0);

    ops::println(a);
    ops::println(b);
    ops::println(c);
    return 0;
}