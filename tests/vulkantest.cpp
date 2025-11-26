#include "ops.h"
#include <print>

int main() {
    Tensor a = Tensor::Fill({3,2592,2048},3.1415926f,DataType::FLOAT32);

    Tensor b = Tensor::Fill({3,2592,2048},2.71818f,DataType::FLOAT32);

    auto c = ops::Dot(a,b);

    ops::println(c);

    return 0;
}