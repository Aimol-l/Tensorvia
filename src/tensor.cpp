#include "ops.h"
#include "factory.h"
#include "tensor.h"
#include <utility>
#include <algorithm>

Tensor::Tensor(){
    this->m_meta.numel = 0;
#ifdef BACKEND_CPU
    this->m_meta.device = Device::CPU;
#endif
#ifdef BACKEND_CUDA
    this->m_meta.device = Device::CUDA;
#endif
#ifdef BACKEND_SYCL
    this->m_meta.device = Device::SYCL;
#endif
#ifdef BACKEND_VULKAN
    this->m_meta.device = Device::VULKAN;
#endif
    this->m_meta.shape = {};
    this->m_meta.dtype = DataType::FLOAT32;
    this->m_impl = nullptr;
}
Tensor::Tensor(std::vector<int64_t> shape,DataType dtype = DataType::FLOAT32){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    // 使用默认数据类型 float32 ,使用编译选择的设备
#ifdef BACKEND_CPU
    this->m_meta.device = Device::CPU;
#endif
#ifdef BACKEND_CUDA
    this->m_meta.device = Device::CUDA;
#endif
#ifdef BACKEND_SYCL
    this->m_meta.device = Device::SYCL;
#endif
#ifdef BACKEND_VULKAN
    this->m_meta.device = Device::VULKAN;
#endif
    this->m_meta.shape = shape;
    this->m_meta.dtype = dtype;
    this->m_meta.numel = calc_numel(shape);
    this->m_impl = create_tensor_impl(shape,this->m_meta.dtype, this->m_meta.device);
}
Tensor::Tensor(std::initializer_list<int64_t> shape,DataType dtype = DataType::FLOAT32){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");

    std::vector<int64_t> shape_(shape);
    // 使用默认数据类型 float32 ,使用编译选择的设备
#ifdef BACKEND_CPU
    this->m_meta.device = Device::CPU;
#endif
#ifdef BACKEND_CUDA
    this->m_meta.device = Device::CUDA;
#endif
#ifdef BACKEND_SYCL
    this->m_meta.device = Device::SYCL;
#endif
#ifdef BACKEND_VULKAN
    this->m_meta.device = Device::VULKAN;
#endif
    this->m_meta.dtype = dtype;
    this->m_meta.shape = shape_;
    this->m_meta.numel = calc_numel(shape_);
    m_impl = create_tensor_impl(shape_,this->m_meta.dtype, this->m_meta.device);
}
Tensor::Tensor(void *ptr, std::initializer_list<int64_t> shape, DataType dtype, Device device){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");

    std::vector<int64_t> shape_(shape);
    // 使用默认数据类型 float32 ,使用编译选择的设备
#ifdef BACKEND_CPU
    this->m_meta.device = Device::CPU;
#endif
#ifdef BACKEND_CUDA
    this->m_meta.device = Device::CUDA;
#endif
#ifdef BACKEND_SYCL
    this->m_meta.device = Device::SYCL;
#endif
#ifdef BACKEND_VULKAN
    this->m_meta.device = Device::VULKAN;
#endif
    this->m_meta.dtype = DataType::FLOAT32;
    this->m_meta.shape = shape_;
    this->m_meta.numel = calc_numel(shape_);
    m_impl = create_tensor_impl(ptr,shape_,this->m_meta.dtype, this->m_meta.device);
}
Tensor::Tensor(void* ptr, std::vector<int64_t> shape,DataType dtype,Device device) {
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    // 使用默认数据类型 float32 ,使用编译选择的设备
#ifdef BACKEND_CPU
    this->m_meta.device = Device::CPU;
#endif
#ifdef BACKEND_CUDA
    this->m_meta.device = Device::CUDA;
#endif
#ifdef BACKEND_SYCL
    this->m_meta.device = Device::SYCL;
#endif
#ifdef BACKEND_VULKAN
    this->m_meta.device = Device::VULKAN;
#endif
    this->m_meta.dtype = dtype;
    this->m_meta.shape = shape;
    this->m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(ptr,shape,this->m_meta.dtype, this->m_meta.device);
}

Tensor::Tensor(std::vector<int64_t> shape, DataType dtype, Device device){
    if(shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    this->m_meta.device = device;
    this->m_meta.dtype = dtype;
    this->m_meta.shape = shape;
    this->m_meta.numel = calc_numel(shape);
    this->m_impl = create_tensor_impl(shape, dtype, device);
}
Tensor::Tensor(std::initializer_list<int64_t> shape, DataType dtype, Device device){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    this->m_meta.device = device;
    this->m_meta.dtype = dtype;
    this->m_meta.shape = shape;
    this->m_meta.numel = calc_numel(shape);
    std::vector<int64_t> shape_(shape);
    this->m_impl = create_tensor_impl(shape_, dtype, device);
}
template <typename T>
Tensor::Tensor(std::vector<T> &vec, std::initializer_list<int64_t> shape){
    // create a tensor from a vector
    if(vec.empty()) throw std::runtime_error("Cannot create a tensor from an empty vector");

#ifdef BACKEND_CPU
    this->m_meta.device = Device::CPU;
#endif
#ifdef BACKEND_CUDA
    this->m_meta.device = Device::CUDA;
#endif
#ifdef BACKEND_SYCL
    this->m_meta.device = Device::SYCL;
#endif
#ifdef BACKEND_VULKAN
    this->m_meta.device = Device::VULKAN;
#endif
    // 根据T的类型设置m_dtype
    if constexpr (std::is_same_v<T, float32>) {
        this->m_meta.dtype = DataType::FLOAT32;
    } else if constexpr (std::is_same_v<T, float64>) {
        this->m_meta.dtype = DataType::FLOAT64;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        this->m_meta.dtype = DataType::INT64;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        this->m_meta.dtype = DataType::INT32;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        this->m_meta.dtype = DataType::INT16;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        this->m_meta.dtype = DataType::INT8;
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        this->m_meta.dtype = DataType::BFLOAT16;
    }else if constexpr (std::is_same_v<T, float16>) {
        this->m_meta.dtype = DataType::FLOAT16;
    }
    this->m_meta.shape = shape;
    this->m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(vec.data(),this->m_meta.shape,this->m_meta.dtype, this->m_meta.device);
}

template<typename T>
Tensor::Tensor(std::vector<T>& vec, std::vector<int64_t> shape) {
    // create a tensor from a vector
    if(vec.empty()) throw std::runtime_error("Cannot create a tensor from an empty vector");
#ifdef BACKEND_CPU
    this->m_meta.device = Device::CPU;
#endif
#ifdef BACKEND_CUDA
    this->m_meta.device = Device::CUDA;
#endif
#ifdef BACKEND_SYCL
    this->m_meta.device = Device::SYCL;
#endif
#ifdef BACKEND_VULKAN
    this->m_meta.device = Device::VULKAN;
#endif
    // 根据T的类型设置m_dtype
    if constexpr (std::is_same_v<T, float32>) {
        this->m_meta.dtype = DataType::FLOAT32;
    }else if (std::is_same_v<T, float64>) {
        this->m_meta.dtype = DataType::FLOAT64;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        this->m_meta.dtype = DataType::INT64;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        this->m_meta.dtype = DataType::INT32;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        this->m_meta.dtype = DataType::INT16;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        this->m_meta.dtype = DataType::INT8;
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        this->m_meta.dtype = DataType::BFLOAT16;
    }else if constexpr (std::is_same_v<T, float16>) {
        this->m_meta.dtype = DataType::FLOAT16;
    }
    this->m_meta.shape = shape;
    this->m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(vec.data(),this->m_meta.shape,this->m_meta.dtype, this->m_meta.device);
}

Tensor Tensor::clone() const{
    Tensor temp(this->m_meta.shape,this->m_meta.dtype,this->m_meta.device);
    if (this->m_impl && temp.m_impl)    
        this->m_impl->copy_to(temp.m_impl->data()); // 同设备复制
    return temp;
}

void* Tensor::data(){
    return m_impl->data();
}
const void* Tensor::data() const{
    return m_impl->data();
}
// 拷贝构造
Tensor::Tensor(const Tensor& other){
    this->m_meta.device = other.device();
    this->m_meta.dtype =  other.dtype();
    this->m_meta.shape =  other.shape();
    this->m_meta.numel =  calc_numel(other.shape());
    this->m_impl =  other.m_impl ? other.m_impl->clone() : nullptr;
}
// 移动构造
Tensor::Tensor(Tensor &&other) noexcept{
    this->m_meta = std::move(other.m_meta);
    this->m_impl = std::move(other.m_impl);
}

    // 拷贝赋值运算符
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        this->m_meta = std::move(other.m_meta);
        this->m_impl = other.m_impl ? other.m_impl->clone() : nullptr;
    }
    return *this;
}

// 移动赋值运算符
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        this->m_meta = std::move(other.m_meta);
        this->m_impl = std::move(other.m_impl);
    }
    return *this;
}
Device Tensor::device()const{return this->m_meta.device;}

std::vector<int64_t> Tensor::shape() const{return this->m_meta.shape;}
void Tensor::reshape(std::vector<int64_t> &newshape){
    int old_total = std::accumulate(this->m_meta.shape.begin(), this->m_meta.shape.end(), 1, std::multiplies<int>());
    int new_total = std::accumulate(newshape.begin(), newshape.end(), 1, std::multiplies<int>());
    if(old_total != new_total){
        std::string info = std::format("new elements count must be equal to old elements count. {} != {}",old_total,new_total);
        throw std::runtime_error(info);
    }
    this->m_meta.shape.clear();
    this->m_meta.shape.assign(newshape.begin(), newshape.end());
    m_impl->reshape(newshape);
}
void Tensor::reshape(std::initializer_list<int64_t> newshape){
    int old_total = std::accumulate(this->m_meta.shape.begin(), this->m_meta.shape.end(), 1, std::multiplies<int>());
    int new_total = std::accumulate(newshape.begin(), newshape.end(), 1, std::multiplies<int>());
    if(old_total != new_total){
        throw std::runtime_error("new elements count must be equal to old elements count");
    }
    this->m_meta.shape.clear();
    this->m_meta.shape.assign(newshape.begin(), newshape.end());
    m_impl->reshape(newshape);
}
size_t Tensor::dims()const{
    return this->m_meta.shape.size();
}
int64_t Tensor::shape(int i)
{
    if(i<0) return this->m_meta.shape[this->m_meta.shape.size()+i];
    return this->m_meta.shape[i];
}
int64_t Tensor::shape(int i) const{
    if(i<0) return this->m_meta.shape[this->m_meta.shape.size()+i];
    return this->m_meta.shape[i];
}
DataType Tensor::dtype() const { return this->m_meta.dtype; }

Tensor Tensor::to_type(DataType dst){
   return ops::Typecast(*this,dst);
}

// cuda|sycl|vulkan -> cpu
void Tensor::to_host(){
    if(this->m_meta.device == Device::CPU) return;
    auto cpu_impl =  create_tensor_impl(this->m_meta.shape, this->m_meta.dtype, Device::CPU);
    copy_device_to_host(m_impl,cpu_impl,this->m_meta.dtype); // 
    m_impl = cpu_impl;
    this->m_meta.device = Device::CPU;
}

// cpu --> cuda|sycl|vulkan
void Tensor::to_device(uint32_t id){
    if(this->m_meta.device != Device::CPU) return;
#ifdef BACKEND_CPU
    return ;
#endif
#ifdef BACKEND_SYCL
    auto device_impl = create_tensor_impl(this->m_meta.shape, this->m_meta.dtype, Device::SYCL);
    copy_host_to_device(m_impl,device_impl,this->m_meta.dtype);
    m_impl = std::move(device_impl);
    this->m_meta.device = Device::SYCL;
#endif
#ifdef BACKEND_CUDA
    auto device_impl = create_tensor_impl(this->m_meta.shape, this->m_meta.dtype, Device::CUDA);
    copy_host_to_device(m_impl,device_impl,this->m_meta.dtype);
    m_impl = std::move(device_impl);
    this->m_meta.device = Device::CUDA;
#endif
#ifdef BACKEND_VULKAN
    auto device_impl = create_tensor_impl(this->m_meta.shape, this->m_meta.dtype, Device::VULKAN);
    copy_host_to_device(m_impl,device_impl,this->m_meta.dtype);
    m_impl = std::move(device_impl);
    this->m_meta.device = Device::VULKAN;
#endif
}

Tensor Tensor::empty_like(Tensor& tensor) const{
    return Tensor(tensor.shape(),this->m_meta.dtype,this->m_meta.device);
}

std::shared_ptr<TensorImpl> Tensor::get_impl() const{
    return m_impl;
}
Tensor Tensor::slice(const std::vector<std::pair<int, int>> &ranges) const{
    return ops::Slice(*this,ranges);
}

Tensor Tensor::operator+(const Tensor& other) const{
    return ops::Add(*this,other);
}

Tensor Tensor::operator-(const Tensor& other) const{
    return ops::Sub(*this,other);
}


Tensor Tensor::operator*(const Tensor& other) const{
    return ops::Dot(*this,other);
}

Tensor operator+(float a, const Tensor& other) {
    return ops::Add(other, a);
}

Tensor Tensor::operator+(float a) const{
    return ops::Add(*this, a);
}

Tensor operator-(float a, const Tensor& other) {
    return ops::Sub(other, a);
}

Tensor Tensor::operator-(float a) const{
    return ops::Sub(*this, a);
}

Tensor operator*(float a, const Tensor& other) {
    return ops::Dot(other, a);
}

Tensor Tensor::operator*(float a) const{
    return ops::Dot(*this, a);
}
Tensor Tensor::operator/(const Tensor& other) const{
    return ops::Div(*this,other);
}

Tensor Tensor::operator/(float a) const {
    return ops::Div(*this,a);
}

Tensor Tensor::Zeros(std::initializer_list<int64_t> shape, DataType dtype){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    std::vector<int64_t> shape_(shape);
    return ops::Zeros(shape_,dtype);
}
Tensor Tensor::Zeros(std::vector<int64_t> shape, DataType dtype){
    if(shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    return ops::Zeros(shape,dtype);
}
Tensor Tensor::Ones(std::initializer_list<int64_t> shape, DataType dtype){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    std::vector<int64_t> shape_(shape);
    return ops::Ones(shape_,dtype);
}
Tensor Tensor::Ones(std::vector<int64_t> shape, DataType dtype){
    if(shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    return ops::Ones(shape,dtype);
}
Tensor Tensor::Fill(std::initializer_list<int64_t> shape, float value, DataType dtype){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    std::vector<int64_t> shape_(shape);
    return ops::Fill(shape_,dtype,value);
}
Tensor Tensor::Fill(std::vector<int64_t> shape, float value, DataType dtype){
    if(shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    return ops::Fill(shape,dtype,value);
}
Tensor Tensor::Random(std::initializer_list<int64_t> shape, float min, float max, DataType dtype){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    std::vector<int64_t> shape_(shape);
    return ops::Random(shape_,dtype,min,max);
}
Tensor Tensor::Random(std::vector<int64_t> shape, float min, float max, DataType dtype){
    if(shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    return ops::Random(shape,dtype,min,max);
}


template <typename T>
T Tensor::at(std::initializer_list<int64_t> idxs)
{
    if(this->device() != Device::CPU) throw std::runtime_error("Tensor device must be CPU");
    if(idxs.size() != this->m_meta.shape.size()) throw std::runtime_error("Tensor index size must be equal to tensor shape size");
    std::vector<int64_t> idxs_(idxs);
    for(int i =0;i<idxs_.size();i++){
        if(idxs_[i] >= this->shape(i) || idxs_[i] < 0)
            throw std::runtime_error("index out of range");
    }
    // 计算线性索引（使用 strides）
    size_t index = 0;
    size_t stride = 1;
    for(int i = this->m_meta.shape.size() - 1; i >= 0; --i){
        index += idxs_[i] * stride;
        stride *= this->m_meta.shape[i];
    }
    switch (this->m_meta.dtype) {
        case DataType::INT8:    return static_cast<T>(static_cast<int8_t*>(this->data())[index]);
        case DataType::INT16:   return static_cast<T>(static_cast<int16_t*>(this->data())[index]);
        case DataType::INT32:   return static_cast<T>(static_cast<int32_t*>(this->data())[index]);
        case DataType::INT64:   return static_cast<T>(static_cast<int64_t*>(this->data())[index]);
        case DataType::FLOAT16: return static_cast<T>(static_cast<float16*>(this->data())[index]);
        case DataType::BFLOAT16:return static_cast<T>(static_cast<bfloat16*>(this->data())[index]);
        case DataType::FLOAT32: return static_cast<T>(static_cast<float32*>(this->data())[index]);
        case DataType::FLOAT64: return static_cast<T>(static_cast<float64*>(this->data())[index]);
        default:
            throw std::runtime_error("Unsupported dtype in Tensor::at");
    }
}

template <typename T>
T Tensor::operator[](std::initializer_list<int64_t> idxs){
    if(this->device() != Device::CPU) throw std::runtime_error("Tensor device must be CPU");
    if(idxs.size() != this->m_meta.shape.size()) throw std::runtime_error("Tensor index size must be equal to tensor shape size");
    std::vector<int64_t> idxs_(idxs);
    for(int i =0;i<idxs_.size();i++){
        if(idxs_[i] >= this->shape(i) || idxs_[i] < 0)
            throw std::runtime_error("index out of range");
    }
    // 计算线性索引（使用 strides）
    size_t index = 0;
    size_t stride = 1;
    for(int i = this->m_meta.shape.size() - 1; i >= 0; --i){
        index += idxs_[i] * stride;
        stride *= this->m_meta.shape[i];
    }
    switch (this->m_meta.dtype) {
        case DataType::INT8:    return static_cast<T>(static_cast<int8_t*>(this->data())[index]);
        case DataType::INT16:   return static_cast<T>(static_cast<int16_t*>(this->data())[index]);
        case DataType::INT32:   return static_cast<T>(static_cast<int32_t*>(this->data())[index]);
        case DataType::INT64:   return static_cast<T>(static_cast<int64_t*>(this->data())[index]);
        case DataType::FLOAT16: return static_cast<T>(static_cast<float16*>(this->data())[index]);
        case DataType::BFLOAT16:return static_cast<T>(static_cast<bfloat16*>(this->data())[index]);
        case DataType::FLOAT32: return static_cast<T>(static_cast<float32*>(this->data())[index]);
        case DataType::FLOAT64: return static_cast<T>(static_cast<float64*>(this->data())[index]);
        default:
            throw std::runtime_error("Unsupported dtype in Tensor::at");
    }
}

Tensor& Tensor::squeeze(int dim){
    if (dim > this->m_meta.shape.size())  throw std::runtime_error("squeeze dim out of range");
    if(dim < 0){
        // 将shape中是1的维度去除，数据存储不修改
        for(int i = this->m_meta.shape.size()-1; i >= 0; i--){
            if(this->m_meta.shape[i] == 1){
                this->m_meta.shape.erase(this->m_meta.shape.begin()+i);
            }
        }
    }
    // 将shape中是1的维度去除，数据存储不修改
    if (this->m_meta.shape[dim] == 1)  this->m_meta.shape.erase(this->m_meta.shape.begin()+dim);
    return *this;
}
Tensor& Tensor::unsqueeze(size_t dim){
    if(dim > this->m_meta.shape.size()) throw std::runtime_error("unsqueeze dim out of range");
    this->m_meta.shape.insert(this->m_meta.shape.begin()+dim,1);
    return *this;
}


Tensor Tensor::operator==(const Tensor& other) const{
    return ops::Equal(*this,other);
}
Tensor Tensor::operator==(const float val) const{
    Tensor t = ops::Fill(this->m_meta.shape, DataType::FLOAT32, val);
    return ops::Equal(*this,t);
}

Tensor Tensor::operator!=(const Tensor& other) const{
    return ops::NotEqual(*this,other);
}
Tensor Tensor::operator!=(const float val) const{
    Tensor t = ops::Fill(this->m_meta.shape, DataType::FLOAT32, val);
    return ops::NotEqual(*this,t);
}

Tensor Tensor::operator>(const Tensor& other) const{
    return ops::Greater(*this,other);
}
Tensor Tensor::operator>(const float val) const{
    Tensor t = ops::Fill(this->m_meta.shape, DataType::FLOAT32, val);
    return ops::Greater(*this,t);
}

Tensor Tensor::operator>=(const Tensor& other) const{
    return ops::GreaterEqual(*this,other);
}

Tensor Tensor::operator>=(const float val) const{
    Tensor t = ops::Fill(this->m_meta.shape, DataType::FLOAT32, val);
    return ops::GreaterEqual(*this,t);
}

Tensor Tensor::operator<(const Tensor& other) const{
    return ops::Less(*this,other);
}
Tensor Tensor::operator<(const float val) const{
    Tensor t = ops::Fill(this->m_meta.shape, DataType::FLOAT32, val);
    return ops::Less(*this,t);
}

Tensor Tensor::operator<=(const Tensor& other) const{
    return ops::LessEqual(*this,other);
}
Tensor Tensor::operator<=(const float val) const{
    Tensor t = ops::Fill(this->m_meta.shape, DataType::FLOAT32, val);
    return ops::LessEqual(*this,t);
}

template int8_t Tensor::at<int8_t>(std::initializer_list<int64_t> idxs);
template int16_t Tensor::at<int16_t>(std::initializer_list<int64_t> idxs);
template int32_t Tensor::at<int32_t>(std::initializer_list<int64_t> idxs);
template int64_t Tensor::at<int64_t>(std::initializer_list<int64_t> idxs);
template float16 Tensor::at<float16>(std::initializer_list<int64_t> idxs);
template bfloat16 Tensor::at<bfloat16>(std::initializer_list<int64_t> idxs);
template float32 Tensor::at<float32>(std::initializer_list<int64_t> idxs);
template float64 Tensor::at<float64>(std::initializer_list<int64_t> idxs);

template int8_t Tensor::operator[]<int8_t>(std::initializer_list<int64_t> idxs);
template int16_t Tensor::operator[]<int16_t>(std::initializer_list<int64_t> idxs);
template int32_t Tensor::operator[]<int32_t>(std::initializer_list<int64_t> idxs);
template int64_t Tensor::operator[]<int64_t>(std::initializer_list<int64_t> idxs);
template float16 Tensor::operator[]<float16>(std::initializer_list<int64_t> idxs);
template bfloat16 Tensor::operator[]<bfloat16>(std::initializer_list<int64_t> idxs);
template float32 Tensor::operator[]<float32>(std::initializer_list<int64_t> idxs);
template float64 Tensor::operator[]<float64>(std::initializer_list<int64_t> idxs);

template Tensor::Tensor(std::vector<int8_t>  &vec, std::initializer_list<int64_t> shape);
template Tensor::Tensor(std::vector<int16_t> &vec, std::initializer_list<int64_t> shape);
template Tensor::Tensor(std::vector<int32_t> &vec, std::initializer_list<int64_t> shape);
template Tensor::Tensor(std::vector<int64_t> &vec, std::initializer_list<int64_t> shape);
template Tensor::Tensor(std::vector<float16> &vec, std::initializer_list<int64_t> shape);
template Tensor::Tensor(std::vector<bfloat16> &vec, std::initializer_list<int64_t> shape);
template Tensor::Tensor(std::vector<float32>  &vec, std::initializer_list<int64_t> shape);
template Tensor::Tensor(std::vector<float64>  &vec, std::initializer_list<int64_t> shape);

template Tensor::Tensor(std::vector<int8_t>  &vec, std::vector<int64_t> shape);
template Tensor::Tensor(std::vector<int16_t> &vec, std::vector<int64_t> shape);
template Tensor::Tensor(std::vector<int32_t> &vec, std::vector<int64_t> shape);
template Tensor::Tensor(std::vector<int64_t> &vec, std::vector<int64_t> shape);
template Tensor::Tensor(std::vector<float16> &vec, std::vector<int64_t> shape);
template Tensor::Tensor(std::vector<bfloat16> &vec, std::vector<int64_t> shape);
template Tensor::Tensor(std::vector<float32>  &vec, std::vector<int64_t> shape);
template Tensor::Tensor(std::vector<float64>  &vec, std::vector<int64_t> shape);