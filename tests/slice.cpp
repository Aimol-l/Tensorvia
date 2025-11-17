#include <print>
#include <iostream>

#include "ops.h"

int main() {

    Tensor a = Tensor::Random({2048,2592},-10,10,DataType::FLOAT32);

    ops::println(a);
    
    auto b = a.slice({{100,1000},{230,1024}}); // 非连续的
    auto c = a.slice({{100,1000},{230,1024}}).clone(); // 经过克隆，是连续的

    ops::println(b);
    ops::println(c);

    return 0;
}