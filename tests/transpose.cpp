#include <print>
#include <iostream>

#include "ops.h"


int main() {
    Tensor a = Tensor::Random({10,20,10},0,10,DataType::FLOAT16);

    ops::println(a);

    auto b = ops::transpose(a,{1,0,2});

    ops::println(b);

    return 0;
}