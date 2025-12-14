#include "ops.h"
#include "factory.h"
#include "tensor.h"
#include <utility>
#include <print>
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
    this->m_meta.calculate_strides();
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
    this->m_meta.calculate_strides();
    this->m_meta.numel = calc_numel(shape);
    this->m_impl = create_tensor_impl(this->m_meta.numel,this->m_meta.dtype, this->m_meta.device);
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
    this->m_meta.calculate_strides();
    this->m_meta.numel = calc_numel(shape_);
    m_impl = create_tensor_impl(this->m_meta.numel,this->m_meta.dtype, this->m_meta.device);
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
    this->m_meta.calculate_strides();
    this->m_meta.numel = calc_numel(shape_);
    m_impl = create_tensor_impl(ptr,this->m_meta.numel,this->m_meta.dtype, this->m_meta.device);
}
Tensor::Tensor(void* ptr, std::vector<int64_t> shape,DataType dtype,Device device) {
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    if(ptr == nullptr) throw std::runtime_error("ptr can not nullptr");
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
    this->m_meta.calculate_strides();
    this->m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(ptr,this->m_meta.numel,this->m_meta.dtype, this->m_meta.device);
}

Tensor::Tensor(std::vector<int64_t> shape, DataType dtype, Device device){
    if(shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    this->m_meta.device = device;
    this->m_meta.dtype = dtype;
    this->m_meta.shape = shape;
    this->m_meta.calculate_strides();
    this->m_meta.numel = calc_numel(shape);
    this->m_impl = create_tensor_impl(this->m_meta.numel, dtype, device);
}
Tensor::Tensor(std::initializer_list<int64_t> shape, DataType dtype, Device device){
    if(shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    this->m_meta.device = device;
    this->m_meta.dtype = dtype;
    this->m_meta.shape = shape;
    this->m_meta.calculate_strides();
    this->m_meta.numel = calc_numel(shape);
    this->m_impl = create_tensor_impl(this->m_meta.numel, dtype, device);
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
    this->m_meta.calculate_strides();
    this->m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(vec.data(),this->m_meta.numel,this->m_meta.dtype, this->m_meta.device);
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
    this->m_meta.calculate_strides();
    this->m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(vec.data(),this->m_meta.numel,this->m_meta.dtype, this->m_meta.device);
}
Tensor Tensor::t(){
    std::vector<int64_t> new_shape = {m_meta.shape[1], m_meta.shape[0]};
    std::vector<int64_t> new_strides = {m_meta.strides[1], m_meta.strides[0]};
    return _make_view(new_shape, new_strides,0);
}
Tensor Tensor::permute(std::initializer_list<int64_t> dims) const {
    std::vector<int64_t> dims_vec(dims);
    return this->permute(dims_vec);
}
Tensor Tensor::slice(const std::vector<std::pair<int64_t, int64_t>>& ranges) const {
    size_t ndim = m_meta.shape.size();
    if (ranges.size() > ndim) {
        throw std::invalid_argument("Too many slice ranges");
    }

    std::vector<int64_t> new_shape = m_meta.shape;
    std::vector<int64_t> new_strides = m_meta.strides; // 👈 strides 不变！
    size_t new_offset = m_meta.offset;
    // new_offset = CWH+WH+H+x
    for (size_t i = 0; i < ranges.size(); ++i) {
        int64_t dim_size = m_meta.shape[i];
        auto [start_raw, end_raw] = ranges[i];
        // 标准化负索引
        int64_t start = start_raw < 0 ? start_raw + dim_size : start_raw;
        int64_t end   = end_raw   < 0 ? end_raw   + dim_size : end_raw;
        // Clamp to valid range (like Python)
        start = std::clamp(start, INT64_C(0), dim_size);
        end   = std::clamp(end,   start,       dim_size);
        new_shape[i] = end - start;
        new_offset += static_cast<size_t>(start) * m_meta.strides[i]; // 👈 关键：加 offset
    }
    return _make_view(new_shape, new_strides, new_offset);
}
Tensor Tensor::permute(const std::vector<int64_t>& dims) const {
    // 1. 检查维度数量是否匹配
    if (dims.size() != m_meta.shape.size()) {
        throw std::invalid_argument(
            "permute: number of dims (" + std::to_string(dims.size()) +
            ") doesn't match tensor dim (" + std::to_string(m_meta.shape.size()) + ")"
        );
    }
    // 2. 检查是否有重复或越界
    std::vector<bool> seen(dims.size(), false);
    for (int64_t dim : dims) {
        int64_t ndim = static_cast<int64_t>(m_meta.shape.size());
        int64_t normalized = dim >= 0 ? dim : dim + ndim; // 支持负索引
        if (normalized < 0 || normalized >= ndim) {
            throw std::invalid_argument("Dimension out of range");
        }
        if (seen[normalized]) {
            throw std::invalid_argument("Duplicate dimension in permute");
        }
        seen[normalized] = true;
    }
    // 3. 构建新的 shape 和 strides
    std::vector<int64_t> new_shape;
    std::vector<int64_t> new_strides;
    new_shape.reserve(dims.size());
    new_strides.reserve(dims.size());
    for (int64_t dim : dims) {
        int64_t normalized = dim >= 0 ? dim : dim + static_cast<int64_t>(m_meta.shape.size());
        new_shape.push_back(m_meta.shape[normalized]);
        new_strides.push_back(m_meta.strides[normalized]);
    }
    // 4. 创建视图（共享 Storage）
    return _make_view(new_shape, new_strides,0);
}
Tensor Tensor::clone() const{
    Tensor temp(this->m_meta.shape,this->m_meta.dtype,this->m_meta.device);
    // 如果是连续内存，可以直接复制
    if(this->is_contiguous()){
        this->m_impl->copy_to(temp.m_impl);
        return temp;
    }
    // 如果不是连续内存，需要跨步复制
    auto new_impl = m_impl->clone_as_contiguous(m_meta);
    temp.m_impl = std::move(new_impl);
    return temp;
}

void* Tensor::data(){
    return m_impl->data();
}
const void* Tensor::data() const{
    return m_impl->data();
}
Tensor Tensor::contiguous()const{
    if(this->is_contiguous()) return *this;
    Tensor contig_tensor(m_meta.shape, m_meta.dtype, m_meta.device);
    auto new_impl = m_impl->clone_as_contiguous(m_meta);
    contig_tensor.m_impl = std::move(new_impl);
    return contig_tensor;
}

bool Tensor::is_contiguous() const {
    if (m_meta.shape.empty()) {
        return true; // 标量或空张量视为连续
    }
    // 计算期望的 row-major strides（C 风格）
    std::vector<int64_t> expected_strides(m_meta.shape.size());
    expected_strides.back() = 1;
    for (int64_t i = static_cast<int64_t>(m_meta.shape.size()) - 2; i >= 0; --i) {
        expected_strides[i] = expected_strides[i + 1] * m_meta.shape[i + 1];
    }
    return m_meta.strides == expected_strides;
}

Tensor& Tensor::to_contiguous(){
    if(this->is_contiguous()) return *this;
    auto new_impl = m_impl->clone_as_contiguous(m_meta);
    this->m_impl = std::move(new_impl);
    return *this;
}

Tensor Tensor::view(std::initializer_list<int64_t> new_shape_list){
    std::vector<int64_t> new_shape(new_shape_list);
    return this->view(new_shape);
}
Tensor Tensor::view(std::vector<int64_t> new_shape){
    // 1. 检查是否连续（PyTorch/NumPy 类似）
    if (!this->is_contiguous()) {
        throw std::invalid_argument(
            "view: tensor is not contiguous. Call .contiguous() first."
        );
    }
    // 2. 处理 -1（自动推导维度）
    int64_t infer_dim = -1;
    int64_t new_numel = 1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (infer_dim != -1) {
                throw std::invalid_argument("Only one dimension can be -1");
            }
            infer_dim = static_cast<int64_t>(i);
            continue;
        }
        if (new_shape[i] < 0) {
            throw std::invalid_argument("Negative dimension size not allowed (except -1)");
        }
        new_numel *= new_shape[i];
    }
    // 3. 推导 -1 的值
    if (infer_dim != -1) {
        if (new_numel == 0 || this->numel() % new_numel != 0) {
            throw std::invalid_argument(
                "view: invalid inferred size for -1 (numel mismatch)"
            );
        }
        new_shape[infer_dim] = static_cast<int64_t>(this->numel() / new_numel);
        new_numel = this->numel();
    }
    // 4. 检查元素总数是否匹配
    if (new_numel != static_cast<int64_t>(this->numel())) {
        throw std::invalid_argument(
            "view: new shape numel does not match original numel"
        );
    }
    std::vector<int64_t> new_strides(new_shape.size());
    if (!new_shape.empty()) {
        new_strides.back() = 1;
        for (int64_t i = static_cast<int64_t>(new_shape.size()) - 2; i >= 0; --i) {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }
    }
    return this->_make_view(new_shape, new_strides,0);
}

Tensor Tensor::_make_view(std::vector<int64_t> shape,std::vector<int64_t> strides,int64_t offset) const {
    Metadata meta;
    meta.numel = calc_numel(shape);
    meta.dtype = this->m_meta.dtype;
    meta.device = this->m_meta.device;
    meta.offset = offset;
    meta.shape = std::move(shape);
    meta.strides = std::move(strides);
    return Tensor(this->m_impl, std::move(meta));
}

// 拷贝构造
Tensor::Tensor(const Tensor& other){
    this->m_meta.device = other.device();
    this->m_meta.dtype =  other.dtype();
    this->m_meta.shape =  other.shape();
    this->m_meta.calculate_strides();
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
        this->m_meta.device = other.device();
        this->m_meta.dtype =  other.dtype();
        this->m_meta.shape =  other.shape();
        this->m_meta.calculate_strides();
        this->m_meta.numel =  calc_numel(other.shape());
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
    int old_total = calc_numel(this->m_meta.shape);
    int new_total = calc_numel(newshape);
    if(old_total != new_total){
        std::string info = std::format("new elements count must be equal to old elements count. {} != {}",old_total,new_total);
        throw std::runtime_error(info);
    }
    this->m_meta.shape.clear();
    this->m_meta.shape.assign(newshape.begin(), newshape.end());
    this->m_meta.calculate_strides();
}
void Tensor::reshape(std::initializer_list<int64_t> newshape){
    int old_total = calc_numel(this->m_meta.shape);
    int new_total = calc_numel(newshape);
    if(old_total != new_total){
        throw std::runtime_error("new elements count must be equal to old elements count");
    }
    this->m_meta.shape.clear();
    this->m_meta.shape.assign(newshape.begin(), newshape.end());
    this->m_meta.calculate_strides();
}
size_t Tensor::dims()const{
    return this->m_meta.shape.size();
}
int64_t Tensor::shape(int i){
    if(i<0) return this->m_meta.shape[this->m_meta.shape.size()+i];
    return this->m_meta.shape[i];
}
int64_t Tensor::shape(int i) const{
    if(i<0) return this->m_meta.shape[this->m_meta.shape.size()+i];
    return this->m_meta.shape[i];
}
DataType Tensor::dtype() const { return this->m_meta.dtype; }

Tensor Tensor::to_type_(DataType dst){
   return ops::Typecast(static_cast<const Tensor>(*this),dst);
}
void Tensor::to_type(DataType dst){
    ops::Typecast(*this,dst);
}

// cuda|sycl|vulkan -> cpu
void Tensor::to_host(){
    if(this->m_meta.device == Device::CPU) 
        return;
    auto cpu_impl =  create_tensor_impl(this->m_meta.numel, this->m_meta.dtype, Device::CPU);
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
    auto device_impl = create_tensor_impl(this->m_meta.numel, this->m_meta.dtype, Device::SYCL);
    copy_host_to_device(m_impl,device_impl,this->m_meta.dtype);
    m_impl = std::move(device_impl);
    this->m_meta.device = Device::SYCL;
#endif
#ifdef BACKEND_CUDA
    auto device_impl = create_tensor_impl(this->m_meta.numel, this->m_meta.dtype, Device::CUDA);
    copy_host_to_device(m_impl,device_impl,this->m_meta.dtype);
    m_impl = std::move(device_impl);
    this->m_meta.device = Device::CUDA;
#endif
#ifdef BACKEND_VULKAN
    auto device_impl = create_tensor_impl(this->m_meta.numel, this->m_meta.dtype, Device::VULKAN);
    copy_host_to_device(m_impl,device_impl,this->m_meta.dtype);
    m_impl = std::move(device_impl);
    this->m_meta.device = Device::VULKAN;
#endif
}

int64_t Tensor:: strides(int i) const{
    // 判断合法性
    // if(i >= this->m_meta.strides.size() || i < -this->m_meta.strides.size())
        // throw std::runtime_error("index out of range");
    // if(i <0) return this->m_meta.strides[this->m_meta.strides.size()+i];
    return this->m_meta.strides[i];
}

std::vector<int64_t> Tensor::strides() const{
    return this->m_meta.strides;
}


Tensor Tensor::empty_like(Tensor& tensor) const{
    return Tensor(tensor.shape(),this->m_meta.dtype,this->m_meta.device);
}

void Tensor::set_impl(std::shared_ptr<TensorImpl> impl,DataType new_dtype){
    m_impl = impl;
    m_meta.dtype = new_dtype;
}
std::shared_ptr<TensorImpl> Tensor::get_impl() const{
    return m_impl;
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
T Tensor::at(std::initializer_list<int64_t> idxs){
    if(this->device() != Device::CPU) 
        throw std::runtime_error("Tensor device must be CPU");
    if(idxs.size() != this->m_meta.shape.size()) throw std::runtime_error("Tensor index size must be equal to tensor shape size");
    std::vector<int64_t> idxs_(idxs);
    for(int i =0;i<idxs_.size();i++){
        if(idxs_[i] >= this->shape(i) || idxs_[i] < 0)
            throw std::runtime_error("index out of range");
    }
    // 计算线性索引（使用 strides）
    size_t index = this->m_meta.offset;
    for (int i = 0; i < idxs_.size(); i++) {
        index += idxs_[i] * this->m_meta.strides[i];
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
    size_t index = this->m_meta.offset;
    for (int i = 0; i < idxs_.size(); i++) {
        index += idxs_[i] * this->m_meta.strides[i];
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
    this->m_meta.calculate_strides();
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