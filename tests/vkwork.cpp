#include "ops.h"
#include <print>
using namespace via;

int main() {

    Tensor a = Tensor::Fill({1024,2048},-1,DataType::FLOAT32);

    ops::Temp(a);

    ops::export_csv(a,"a.csv");

    return 0;
}                       