#include "tensorvia.hpp"

int main(){

    Tensor a = Tensor::Fill({100,100},1,DataType::INT32);

    auto res = ops::Sum(a);

    if(res.item<int32_t>() > 1000){
        ops::println(res);
    }
    
    return 0;
}
