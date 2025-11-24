#include "ops.h"
#include <print>

int main() {
    Tensor a = Tensor::Fill({3,100,100},3.1415,DataType::FLOAT16);

    ops::println(a);

    return 0;
}