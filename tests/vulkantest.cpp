#include "ops.h"
#include <print>

int main() {
    Tensor a = Tensor::Random({3,2592,2048},0,1,DataType::FLOAT16);

    ops::println(a);

    auto b = ops::Softmax(a,0);

    ops::println(b);

    return 0;
}