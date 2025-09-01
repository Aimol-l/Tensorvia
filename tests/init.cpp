#include <print>
#include <iostream>
#include <vector>

#include "ops.h"


int main() {

    Tensor a = Tensor::Ones({5,5},DataType::INT8);
    Tensor b = Tensor::Zeros({5,5});
    Tensor c = Tensor::Random({200,300,400},0,10,DataType::FLOAT32);
    Tensor d = Tensor::Fill({5,5},3.1415f);

    std::vector vec = {1,2,3,4,5, 6};
    std::vector shape = {2,3};

    Tensor e(vec.data(), shape, DataType::INT32, Device::CPU);

    Tensor f(vec, shape);

    // ops::println(a);
    // ops::println(b);
    // ops::println(c);
    // ops::println(d);
    ops::println(e);
    ops::println(f);

}