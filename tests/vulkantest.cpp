#include "ops.h"
#include <print>

int main() {
    Tensor a = Tensor::Fill({3,2592,2048},10,DataType::FLOAT32);
    // Tensor b = Tensor::Fill({3,100,100},100,DataType::FLOAT32);

    RUNNING_TIME(ops::Add(a,3.24));
    ops::println(a);
    return 0;
}