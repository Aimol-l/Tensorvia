#include "ops.h"
#include <print>

int main() {
    Tensor a = Tensor::Fill({3,2592,2048},2,DataType::FLOAT32);

    auto b = ops::Pow(static_cast<const Tensor&>(a),3);

    ops::println(b);

    return 0;
}