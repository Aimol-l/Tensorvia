#include <print>
#include <iostream>

#include "ops.h"
using namespace via;


int main() {

    Tensor a = Tensor::Ones({5,5},DataType::INT8);
    Tensor b = Tensor::Zeros({5,5});
    Tensor c = Tensor::Random({200,300,400},0,10,DataType::FLOAT32);
    Tensor d = Tensor::Fill({5,5},3.1415f);

    ops::println(a);
    ops::println(b);
    ops::println(c);
    ops::println(d);
}