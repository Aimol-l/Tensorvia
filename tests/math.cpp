#include "ops.h"

int main() {
    Tensor a = Tensor::Random({5,5},-10,10,DataType::INT8);
    Tensor b = Tensor::Random({5,5},-10,10,DataType::INT16);
    Tensor c = Tensor::Random({5,5},-10,10,DataType::INT32);
    Tensor d = Tensor::Random({5,5},-10,10,DataType::INT64);
    Tensor e = Tensor::Random({5,5},-10,10,DataType::FLOAT16);
    Tensor f = Tensor::Random({5,5},-10,10,DataType::BFLOAT16);
    Tensor g = Tensor::Random({5,5},-10,10,DataType::FLOAT32);
    Tensor h = Tensor::Random({5,5},-10,10,DataType::FLOAT64);

    ops::println(a + b);
    ops::println(c - d);
    ops::println(e * f);
    ops::println(g / h);
    return 0;
}