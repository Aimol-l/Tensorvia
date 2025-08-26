#include "ops.h"
#include "factory.h"
#include "tensor.h"
#include <utility>
#include <algorithm>

Tensor::Tensor(){
    m_numel = 0;
#ifdef BACKEND_CPU
    m_device = Device::CPU;
#endif
#ifdef BACKEND_CUDA
    m_device = Device::CUDA;
#endif
#ifdef BACKEND_SYCL
    m_device = Device::SYCL;
#endif
#ifdef BACKEND_VULKAN
    m_device = Device::VULKAN;
#endif
    m_shape = {};
    m_dtype = DataType::FLOAT32;
    m_impl = nullptr;
}
Tensor::Tensor(std::vector<int> shape,DataType dtype = DataType::FLOAT32){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    // 使用默认数据类型 float32 ,使用编译选择的设备
#ifdef BACKEND_CPU
    m_device = Device::CPU;
#endif
#ifdef BACKEND_CUDA
    m_device = Device::CUDA;
#endif
#ifdef BACKEND_SYCL
    m_device = Device::SYCL;
#endif
#ifdef BACKEND_VULKAN
    m_device = Device::VULKAN;
#endif
    m_shape = shape;
    m_dtype = dtype;
    m_numel = calc_numel(shape);
    m_impl = create_tensor_impl(shape,m_dtype, m_device);
}
Tensor::Tensor(std::initializer_list<int> shape,DataType dtype = DataType::FLOAT32){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");

    std::vector<int> shape_(shape);
    // 使用默认数据类型 float32 ,使用编译选择的设备
#ifdef BACKEND_CPU
    m_device = Device::CPU;
#endif
#ifdef BACKEND_CUDA
    m_device = Device::CUDA;
#endif
#ifdef BACKEND_SYCL
    m_device = Device::SYCL;
#endif
#ifdef BACKEND_VULKAN
    m_device = Device::VULKAN;
#endif
    m_dtype = dtype;
    m_shape = shape_;
    m_numel = calc_numel(shape_);
    m_impl = create_tensor_impl(shape_,m_dtype, m_device);
}
Tensor::Tensor(void *ptr, std::initializer_list<int> shape, DataType dtype, Device device){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");

    std::vector<int> shape_(shape);
    // 使用默认数据类型 float32 ,使用编译选择的设备
#ifdef BACKEND_CPU
    m_device = Device::CPU;
#endif
#ifdef BACKEND_CUDA
    m_device = Device::CUDA;
#endif
#ifdef BACKEND_SYCL
    m_device = Device::SYCL;
#endif
#ifdef BACKEND_VULKAN
    m_device = Device::VULKAN;
#endif
    m_dtype = DataType::FLOAT32;
    m_shape = shape_;
    m_numel = calc_numel(shape_);
    m_impl = create_tensor_impl(ptr,shape_,m_dtype, m_device);
}
Tensor::Tensor(std::vector<int> shape, DataType dtype, Device device){
    if(shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    m_device = device;
    m_dtype = dtype;
    m_shape = shape;
    m_numel = calc_numel(shape);
    m_impl = create_tensor_impl(shape, dtype, device);
}
Tensor::Tensor(std::initializer_list<int> shape, DataType dtype, Device device){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    m_device = device;
    m_dtype = dtype;
    m_shape = shape;

    m_numel = calc_numel(shape);
    std::vector<int> shape_(shape);
    m_impl = create_tensor_impl(shape_, dtype, device);
}
template <typename T>
Tensor::Tensor(std::vector<T> &vec, std::initializer_list<int> shape){
    // create a tensor from a vector
    if(vec.empty()) throw std::runtime_error("Cannot create a tensor from an empty vector");
    // 判断shape是否正确
#ifdef BACKEND_CPU
    m_device = Device::CPU;
#endif
#ifdef BACKEND_CUDA
    m_device = Device::CUDA;
#endif
#ifdef BACKEND_SYCL
    m_device = Device::SYCL;
#endif
#ifdef BACKEND_VULKAN
    m_device = Device::VULKAN;
#endif
    // 根据T的类型设置m_dtype
    if constexpr (std::is_same_v<T, float32>) {
        m_dtype = DataType::FLOAT32;
    } else if constexpr (std::is_same_v<T, float64>) {
        m_dtype = DataType::FLOAT64;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        m_dtype = DataType::INT64;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        m_dtype = DataType::INT32;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        m_dtype = DataType::INT16;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        m_dtype = DataType::INT8;
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        m_dtype = DataType::BFLOAT16;
    }else if constexpr (std::is_same_v<T, float16>) {
        m_dtype = DataType::FLOAT16;
    }
    m_shape = shape;
    m_numel = calc_numel(shape);
    m_impl = create_tensor_impl(vec.data(),m_shape,m_dtype, m_device);
}

Tensor Tensor::clone() const{
    Tensor temp(this->m_shape,this->m_dtype,this->m_device);
    if (this->m_impl && temp.m_impl)    
        this->m_impl->copy_to(temp.m_impl->data()); // 同设备复制
    return temp;
}

// 拷贝构造
Tensor::Tensor(const Tensor& other){
    m_device = other.device();
    m_dtype =  other.dtype();
    m_shape =  other.shape();
    m_numel =  calc_numel(other.shape());
    this->m_impl =  other.m_impl ? other.m_impl->clone() : nullptr;
}
// 移动构造
Tensor::Tensor(Tensor &&other) noexcept{
    m_device = std::move(other.m_device);
    m_dtype = std::move(other.m_dtype);
    m_shape = std::move(other.m_shape);
    m_numel = std::exchange(other.m_numel, 0);
    m_impl = std::move(other.m_impl);
}

    // 拷贝赋值运算符
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        m_device = other.m_device;
        m_dtype = other.m_dtype;
        m_shape = other.m_shape;
        m_numel = other.m_numel;
        m_impl = other.m_impl ? other.m_impl->clone() : nullptr;
    }
    return *this;
}

// 移动赋值运算符
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        m_device = std::move(other.m_device);
        m_dtype = std::move(other.m_dtype);
        m_shape = std::move(other.m_shape);
        m_numel = std::exchange(other.m_numel, 0);
        m_impl = std::move(other.m_impl);
    }
    return *this;
}
Device Tensor::device()const{return m_device;}

std::vector<int> Tensor::shape() const{return m_shape;}
void Tensor::reshape(std::vector<int> &newshape){
    int old_total = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<int>());
    int new_total = std::accumulate(newshape.begin(), newshape.end(), 1, std::multiplies<int>());
    if(old_total != new_total){
        std::string info = std::format("new elements count must be equal to old elements count. {} != {}",old_total,new_total);
        throw std::runtime_error(info);
    }
    m_shape.clear();
    m_shape.assign(newshape.begin(), newshape.end());
    m_impl->reshape(newshape);
}
void Tensor::reshape(std::initializer_list<int> newshape){
    int old_total = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<int>());
    int new_total = std::accumulate(newshape.begin(), newshape.end(), 1, std::multiplies<int>());
    if(old_total != new_total){
        throw std::runtime_error("new elements count must be equal to old elements count");
    }
    m_shape.clear();
    m_shape.assign(newshape.begin(), newshape.end());
    m_impl->reshape(newshape);
}
size_t Tensor::dims()const{
    return m_shape.size();
}
int Tensor::shape(int i)
{
    if(i<0) return m_shape[m_shape.size()+i];
    return m_shape[i];
}
int Tensor::shape(int i) const{
    if(i<0) return m_shape[m_shape.size()+i];
    return m_shape[i];
}
DataType Tensor::dtype() const { return m_dtype; }

Tensor Tensor::to_type(DataType dst){
   return ops::Typecast(*this,dst);
}

// cuda|sycl|vulkan -> cpu
void Tensor::to_host(){
    if(m_device == Device::CPU) return;
    auto cpu_impl =  create_tensor_impl(m_shape, m_dtype, Device::CPU);
    copy_device_to_host(m_impl,cpu_impl,m_dtype); // 
    m_impl = cpu_impl;
    m_device = Device::CPU;
}

// cpu --> cuda|sycl|vulkan
void Tensor::to_device(uint32_t id){
    if(m_device != Device::CPU) return;
#ifdef BACKEND_CPU
    return ;
#endif
#ifdef BACKEND_SYCL
    auto device_impl = create_tensor_impl(m_shape, m_dtype, Device::SYCL);
    copy_host_to_device(m_impl,device_impl,m_dtype);
    m_impl = std::move(device_impl);
    m_device = Device::SYCL;
#endif
#ifdef BACKEND_CUDA
    auto device_impl = create_tensor_impl(m_shape, m_dtype, Device::CUDA);
    copy_host_to_device(m_impl,device_impl,m_dtype);
    m_impl = std::move(device_impl);
    m_device = Device::CUDA;
#endif
#ifdef BACKEND_VULKAN
    auto device_impl = create_tensor_impl(m_shape, m_dtype, Device::VULKAN);
    copy_host_to_device(m_impl,device_impl,m_dtype);
    m_impl = std::move(device_impl);
    m_device = Device::VULKAN;
#endif
}
void *Tensor::data(){
    return m_impl->data();
}
const void* Tensor::data() const{
    return m_impl->data();
}
Tensor Tensor::empty_like(Tensor& tensor) const{
    return Tensor(tensor.shape(),this->m_dtype,this->m_device);
}

std::shared_ptr<TensorImpl> Tensor::get_impl() const{
    return m_impl;
}
Tensor Tensor::slice(const std::vector<std::pair<int, int>> &ranges) const{
    return ops::Slice(*this,ranges);
}

Tensor Tensor::add(const Tensor& other) const {
    return ops::Add(*this,other);
}

Tensor Tensor::operator+(const Tensor& other) const{
    return ops::Add(*this,other);
}

Tensor Tensor::sub(const Tensor& other) const {
    return ops::Sub(*this,other);
}

Tensor Tensor::operator-(const Tensor& other) const{
    return ops::Sub(*this,other);
}

Tensor Tensor::dot(const Tensor& other) const {
    return ops::Dot(*this,other);
}

Tensor Tensor::operator*(const Tensor& other) const{
    return ops::Dot(*this,other);
}

Tensor operator+(double a, const Tensor& other) {
    
    return ops::Add(other, a);
}

Tensor Tensor::operator+(double a) const{
    
    return ops::Add(*this, a);
}

Tensor operator-(double a, const Tensor& other) {
    return ops::Sub(other, a);
}

Tensor Tensor::operator-(double a) const{
    return ops::Sub(*this, a);
}

Tensor operator*(double a, const Tensor& other) {
    return ops::Dot(other, a);
}

Tensor Tensor::operator*(double a) const{
    return ops::Dot(*this, a);
}

Tensor Tensor::div(const Tensor& other) const {
    return ops::Div(*this,other);
}

Tensor Tensor::operator/(const Tensor& other) const{
    return ops::Div(*this,other);
}

Tensor Tensor::operator/(double a) const {
    return ops::Div(*this,a);
}

Tensor Tensor::matmul(const Tensor& other) const {
    return ops::Mul(*this,other);
}
Tensor Tensor::Zeros(std::initializer_list<int> shape, DataType dtype){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    std::vector<int> shape_(shape);
    return ops::Zeros(shape_,dtype);
}
Tensor Tensor::Zeros(std::vector<int> shape, DataType dtype){
    if(shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    return ops::Zeros(shape,dtype);
}
Tensor Tensor::Ones(std::initializer_list<int> shape, DataType dtype){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    std::vector<int> shape_(shape);
    return ops::Ones(shape_,dtype);
}
Tensor Tensor::Ones(std::vector<int> shape, DataType dtype){
    if(shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    return ops::Ones(shape,dtype);
}
Tensor Tensor::Fill(std::initializer_list<int> shape, float value, DataType dtype){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    std::vector<int> shape_(shape);
    return ops::Fill(shape_,dtype,value);
}
Tensor Tensor::Fill(std::vector<int> shape, float value, DataType dtype){
    if(shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    return ops::Fill(shape,dtype,value);
}
Tensor Tensor::Random(std::initializer_list<int> shape, float min, float max, DataType dtype){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    std::vector<int> shape_(shape);
    return ops::Random(shape_,dtype,min,max);
}
Tensor Tensor::Random(std::vector<int> shape, float min, float max, DataType dtype){
    if(shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    return ops::Random(shape,dtype,min,max);
}


template <typename T>
T Tensor::at(std::initializer_list<int> idxs)
{
    if(this->device() != Device::CPU) throw std::runtime_error("Tensor device must be CPU");
    if(idxs.size() != m_shape.size()) throw std::runtime_error("Tensor index size must be equal to tensor shape size");
    std::vector<int> idxs_(idxs);
    for(int i =0;i<idxs_.size();i++){
        if(idxs_[i] >= this->shape(i) || idxs_[i] < 0)
            throw std::runtime_error("index out of range");
    }
    // 计算线性索引（使用 strides）
    size_t index = 0;
    size_t stride = 1;
    for(int i = m_shape.size() - 1; i >= 0; --i){
        index += idxs_[i] * stride;
        stride *= m_shape[i];
    }
    switch (this->m_dtype) {
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
T Tensor::operator[](std::initializer_list<int> idxs){
    if(this->device() != Device::CPU) throw std::runtime_error("Tensor device must be CPU");
    if(idxs.size() != m_shape.size()) throw std::runtime_error("Tensor index size must be equal to tensor shape size");
    std::vector<int> idxs_(idxs);
    for(int i =0;i<idxs_.size();i++){
        if(idxs_[i] >= this->shape(i) || idxs_[i] < 0)
            throw std::runtime_error("index out of range");
    }
    // 计算线性索引（使用 strides）
    size_t index = 0;
    size_t stride = 1;
    for(int i = m_shape.size() - 1; i >= 0; --i){
        index += idxs_[i] * stride;
        stride *= m_shape[i];
    }
    switch (this->m_dtype) {
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
    if (dim > m_shape.size())  throw std::runtime_error("squeeze dim out of range");
    if(dim < 0){
        // 将shape中是1的维度去除，数据存储不修改
        for(int i = m_shape.size()-1; i >= 0; i--){
            if(m_shape[i] == 1){
                m_shape.erase(m_shape.begin()+i);
            }
        }
    }
    // 将shape中是1的维度去除，数据存储不修改
    if (m_shape[dim] == 1)  m_shape.erase(m_shape.begin()+dim);
    return *this;
}
Tensor& Tensor::unsqueeze(size_t dim){
    if(dim > m_shape.size()) throw std::runtime_error("unsqueeze dim out of range");
    m_shape.insert(m_shape.begin()+dim,1);
    return *this;
}

float Tensor::sum(){
    return ops::Sum(*this);
}

float Tensor::mean(){
    return ops::Mean(*this);
}

float Tensor::max(){
    return ops::Max(*this);
}

float Tensor::min(){
    return ops::Min(*this);
}

Tensor Tensor::sum(int axis){
    return ops::Sum(*this,axis);
}

Tensor Tensor::mean(int axis){
    return ops::Mean(*this,axis);
}

Tensor Tensor::max(int axis){
    return ops::Max(*this,axis);
}

Tensor Tensor::min(int axis){
    return ops::Min(*this,axis);
}

Tensor Tensor::argmax(int axis){
    return ops::Argmax(*this,axis);
}
bool Tensor::all(float val){
    return ops::All(*this,val);
}
bool Tensor::any(float val){
    return ops::Any(*this,val);
}
size_t Tensor::nonzero(){
    return ops::Nonzero(*this);
}
Tensor Tensor::argmin(int axis)
{
    return ops::Argmin(*this,axis);
}
Tensor Tensor::equal(const Tensor &other) const {
    return ops::Equal(*this,other);
}
Tensor Tensor::operator==(const Tensor& other) const{
    return ops::Equal(*this,other);
}
Tensor Tensor::operator==(const float val) const{
    Tensor t = ops::Fill(this->m_shape, DataType::FLOAT32, val);
    return ops::Equal(*this,t);
}
Tensor Tensor::not_equal(const Tensor& other) const{
    return ops::NotEqual(*this,other);
}
Tensor Tensor::operator!=(const Tensor& other) const{
    return ops::NotEqual(*this,other);
}
Tensor Tensor::operator!=(const float val) const{
    Tensor t = ops::Fill(this->m_shape, DataType::FLOAT32, val);
    return ops::NotEqual(*this,t);
}
Tensor Tensor::greater(const Tensor& other) const{
    return ops::Greater(*this,other);
}
Tensor Tensor::operator>(const Tensor& other) const{
    return ops::Greater(*this,other);
}
Tensor Tensor::operator>(const float val) const{
    Tensor t = ops::Fill(this->m_shape, DataType::FLOAT32, val);
    return ops::Greater(*this,t);
}
Tensor Tensor::greater_equal(const Tensor& other) const{
    return ops::GreaterEqual(*this,other);
}
Tensor Tensor::operator>=(const Tensor& other) const{
    return ops::GreaterEqual(*this,other);
}

Tensor Tensor::operator>=(const float val) const{
    Tensor t = ops::Fill(this->m_shape, DataType::FLOAT32, val);
    return ops::GreaterEqual(*this,t);
}

Tensor Tensor::less(const Tensor& other) const{
    return ops::Less(*this,other);
}
Tensor Tensor::operator<(const Tensor& other) const{
    return ops::Less(*this,other);
}
Tensor Tensor::operator<(const float val) const{
    Tensor t = ops::Fill(this->m_shape, DataType::FLOAT32, val);
    return ops::Less(*this,t);
}
Tensor Tensor::less_equal(const Tensor& other) const{
    return ops::LessEqual(*this,other);
}
Tensor Tensor::operator<=(const Tensor& other) const{
    return ops::LessEqual(*this,other);
}
Tensor Tensor::operator<=(const float val) const{
    Tensor t = ops::Fill(this->m_shape, DataType::FLOAT32, val);
    return ops::LessEqual(*this,t);
}

template int8_t Tensor::at<int8_t>(std::initializer_list<int> idxs);
template int16_t Tensor::at<int16_t>(std::initializer_list<int> idxs);
template int32_t Tensor::at<int32_t>(std::initializer_list<int> idxs);
template int64_t Tensor::at<int64_t>(std::initializer_list<int> idxs);
template float16 Tensor::at<float16>(std::initializer_list<int> idxs);
template bfloat16 Tensor::at<bfloat16>(std::initializer_list<int> idxs);
template float32 Tensor::at<float32>(std::initializer_list<int> idxs);
template float64 Tensor::at<float64>(std::initializer_list<int> idxs);

template int8_t Tensor::operator[]<int8_t>(std::initializer_list<int> idxs);
template int16_t Tensor::operator[]<int16_t>(std::initializer_list<int> idxs);
template int32_t Tensor::operator[]<int32_t>(std::initializer_list<int> idxs);
template int64_t Tensor::operator[]<int64_t>(std::initializer_list<int> idxs);
template float16 Tensor::operator[]<float16>(std::initializer_list<int> idxs);
template bfloat16 Tensor::operator[]<bfloat16>(std::initializer_list<int> idxs);
template float32 Tensor::operator[]<float32>(std::initializer_list<int> idxs);
template float64 Tensor::operator[]<float64>(std::initializer_list<int> idxs);

template Tensor::Tensor(std::vector<int8_t>  &vec, std::initializer_list<int> shape);
template Tensor::Tensor(std::vector<int16_t> &vec, std::initializer_list<int> shape);
template Tensor::Tensor(std::vector<int32_t> &vec, std::initializer_list<int> shape);
template Tensor::Tensor(std::vector<int64_t> &vec, std::initializer_list<int> shape);
template Tensor::Tensor(std::vector<float16> &vec, std::initializer_list<int> shape);
template Tensor::Tensor(std::vector<bfloat16> &vec, std::initializer_list<int> shape);
template Tensor::Tensor(std::vector<float32>  &vec, std::initializer_list<int> shape);
template Tensor::Tensor(std::vector<float64>  &vec, std::initializer_list<int> shape);