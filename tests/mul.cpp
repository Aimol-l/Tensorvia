#include <print>
#include <iostream>

#include "ops.h"


int main() {
    Tensor a = Tensor::Random({100,300,500},0,1,DataType::FLOAT32);
    Tensor b = Tensor::Random({100,500,300},0,1,DataType::FLOAT32);

    RUNNING_TIME(ops::Mul(a,b));
    RUNNING_TIME(ops::Mul(a,b));
    return 0;
}