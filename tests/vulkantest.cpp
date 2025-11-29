#include "ops.h"
#include <print>

int main() {
    Tensor a = Tensor::Random({3,2592,2048},1,1000,DataType::FLOAT32);

    auto b = ops::Softmax(a,0);

    return 0;
}