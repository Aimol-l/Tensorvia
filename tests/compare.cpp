#include <print>
#include <iostream>
#include "ops.h"

int main() {
    Tensor a = Tensor::Random({5,5},0,1,DataType::FLOAT16);
    Tensor b = Tensor::Random({5,5},0,1,DataType::FLOAT32);

    auto c = ops::less(a,b);
    auto d = ops::greater(a,b);
    auto e = ops::equal(a,b);
    auto f = ops::not_equal(a,b);

    ops::println(a);
    ops::println(b);

    ops::println(f);
    return 0;
}