#include <print>
#include <iostream>

#include "ops.h"


int main() {
    Tensor a = Tensor::Random({3,6,6},0,10,DataType::FLOAT16);

    auto c = ops::slice(a,{{0,2},{0,5},{0,5}});

    ops::println(a);
    ops::println(c);

    return 0;
}