#include "ops.h"
#include <print>

int main() {
    Tensor a = Tensor::Fill({3,2592,2048},3.1415926f,DataType::FLOAT32);

    auto b = ops::Silu(static_cast<const Tensor&>(a));

    ops::println(b);

    return 0;
}