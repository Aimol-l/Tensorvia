#include <print>
#include <iostream>
#include "ops.h"

int main() {
    Tensor a = Tensor::Random({50,500,500},0,10,DataType::FLOAT32);
    Tensor b = Tensor::Random({33,500,500},0,10,DataType::FLOAT32);

    RUNNING_TIME(ops::concat({a,b},0));

    // ops::println(a);
    // ops::println(b);
    // ops::println(c);
    return 0;
}