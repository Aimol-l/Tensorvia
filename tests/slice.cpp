#include <print>
#include <iostream>

#include "ops.h"

int main() {

    Tensor a = Tensor::Random({7,7},-10,10,DataType::FLOAT32);

    ops::println(a);
    
    auto b = a.slice({{1,5},{2,7}}).clone();
    ops::println(b);

    return 0;
}