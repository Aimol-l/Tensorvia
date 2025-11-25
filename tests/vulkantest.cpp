#include "ops.h"
#include <print>

int main() {
    // Tensor a = Tensor::Fill({3,100,100},3,DataType::FLOAT32);
    Tensor b = Tensor::Random({3,9,9},-127,127,DataType::INT8);

    // ops::Relu(b);

    ops::println(b);

    return 0;
}