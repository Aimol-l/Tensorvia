#include "ops.h"
#include "backend/cpu/ops/temp.h"

using namespace via;

namespace ops {
void TempImpl<Device::CPU>::execute(Tensor& a){
 
    // todo...

}

template struct TempImpl<Device::CPU>;
}