#include <print>
#include <iostream>
#include "ops.h"
using namespace via;

int main() {
    Tensor a = Tensor::Random({5,6},10,100,DataType::INT16);
    Tensor b = Tensor::Random({5,6},10,100,DataType::FLOAT16);

    auto c = ops::Less(a,b);
    auto d = ops::Greater(a,b);
    auto e = ops::Equal(a,b);
    auto f = ops::NotEqual(a,b);

    ops::println(a);
    ops::println(b);
    ops::println(c);
    ops::println(d);
    ops::println(e);
    ops::println(f);
    std::cout<<ops::Nonzero(f)<<std::endl;
    return 0;
}