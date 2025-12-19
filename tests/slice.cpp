#include <print>
#include <iostream>

#include "ops.h"

int main() {

    Tensor a = Tensor::Random({2048,2592},-10,10,DataType::FLOAT32);

    ops::println(a);
    
    auto b = ops::Slice(a,{{0,300},{0,1024}});

    ops::println(b);

    return 0;
}