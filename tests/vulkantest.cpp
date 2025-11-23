#include "ops.h"
#include <print>

int main() {
    Tensor a = Tensor::Fill({3,100,100},3.1415f,DataType::FLOAT32);

    ops::println(a);
    return 0;
}