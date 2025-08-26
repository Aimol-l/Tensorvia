#include "ops.h"

int main() {
    Tensor a = Tensor::Random({5,5},0,1,DataType::INT8);
    Tensor b = Tensor::Random({5,5},0,1,DataType::INT16);
    Tensor c = Tensor::Random({5,5},0,1,DataType::INT32);
    Tensor d = Tensor::Random({5,5},0,1,DataType::INT64);
    Tensor e = Tensor::Random({5,5},0,1,DataType::FLOAT16);
    Tensor f = Tensor::Random({5,5},0,1,DataType::BFLOAT16);
    Tensor g = Tensor::Random({5,5},0,1,DataType::FLOAT32);
    Tensor h = Tensor::Random({5,5},0,1,DataType::FLOAT64);

    // ops...
    Tensor i = a + b;
    Tensor j = c - d;
    Tensor k = e * f;
    Tensor l = g / h;

    // println
    ops::println(i);
    ops::println(j);
    ops::println(k);
    ops::println(l);
    return 0;
}