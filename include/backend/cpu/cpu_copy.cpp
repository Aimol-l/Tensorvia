
#include "core/factory.h"

using namespace via;


#ifdef BACKEND_CPU // 默认指用gcc13+ 编译,msvc似乎还不支持浮点数扩展

void copy_device_to_host(std::shared_ptr<TensorImpl> src,std::shared_ptr<TensorImpl> dst,DataType dtype) {
    // 似乎什么都不用做，因为tensor只能在cpu里面,这个函数不会被调用
}

void copy_host_to_device(std::shared_ptr<TensorImpl> src,std::shared_ptr<TensorImpl> dst,DataType dtype) {
    // 似乎什么都不用做，因为tensor只能在cpu里面，这个函数不会被调用
}
#endif
