#include "ops.h"
#include <print>


int main() {

    Tensor a = Tensor::Random({3,2048,2592},-10,10,DataType::FLOAT32);

    ops::println(a);

    auto b = a.view({3,2592,2048});

    ops::println(b);

    return 0;
}