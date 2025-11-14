#include "backend/cpu/ops/activate.h"
#include "backend/cpu/ops/arithmetic.h"
#include "backend/cpu/ops/concat.h"
#include "backend/cpu/ops/initializer.h"
#include "backend/cpu/ops/logical.h"
#include "backend/cpu/ops/mul.h"
#include "backend/cpu/ops/println.h"
#include "backend/cpu/ops/reduce.h"
#include "backend/cpu/ops/slice.h"
#include "backend/cpu/ops/transpose.h"
#include "backend/cpu/ops/typecast.h"

#ifdef BACKEND_SYCL
    #include "backend/sycl/ops/activate.h"
    #include "backend/sycl/ops/arithmetic.h"
    #include "backend/sycl/ops/concat.h"
    #include "backend/sycl/ops/initializer.h"
    #include "backend/sycl/ops/logical.h"
    #include "backend/sycl/ops/mul.h"
    #include "backend/sycl/ops/reduce.h"
    #include "backend/sycl/ops/slice.h"
    #include "backend/sycl/ops/transpose.h"
    #include "backend/sycl/ops/typecast.h"
#endif
#ifdef BACKEND_CUDA
    #include "backend/cuda/ops/activate.h"
    #include "backend/cuda/ops/arithmetic.h"
    #include "backend/cuda/ops/concat.h"
    #include "backend/cuda/ops/initializer.h"
    #include "backend/cuda/ops/logical.h"
    #include "backend/cuda/ops/mul.h"
    #include "backend/cuda/ops/reduce.h"
    #include "backend/cuda/ops/slice.h"
    #include "backend/cuda/ops/transpose.h"
    #include "backend/cuda/ops/typecast.h"
#endif
#ifdef BACKEND_VULKAN
    #include "backend/vulkan/ops/activate.h"
    #include "backend/vulkan/ops/arithmetic.h"
    #include "backend/vulkan/ops/concat.h"
    #include "backend/vulkan/ops/initializer.h"
    #include "backend/vulkan/ops/logical.h"
    #include "backend/vulkan/ops/mul.h"
    #include "backend/vulkan/ops/reduce.h"
    #include "backend/vulkan/ops/slice.h"
    #include "backend/vulkan/ops/transpose.h"
    #include "backend/vulkan/ops/typecast.h"
#endif
//*************************************************
namespace ops {
    OPS_API void println(Tensor & a){
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        Device dev = a.device();
        if(a.device() == Device::CPU){
            PrintlnImpl<Device::CPU>::execute(a);
        }else{
            Tensor temp = a.clone(); // clone()默认会contiguous
            temp.to_host();
            PrintlnImpl<Device::CPU>::execute(temp);
        }
        std::cout<<std::format("Tensor dtype: {} | Tensor device: {}",dtype_to_string(a.dtype()), device_to_string(dev))<<std::endl;
    }
    OPS_API void println(Tensor && a) {
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        Device dev = a.device();
        if (a.device() == Device::CPU) {
            PrintlnImpl<Device::CPU>::execute(a);
        } else {
            a.to_host(); // 修改 a 是安全的，因为它是一个临时对象
            a.to_contiguous();
            PrintlnImpl<Device::CPU>::execute(a);
        }
        std::cout<<std::format("Tensor dtype: {} | Tensor device: {}",dtype_to_string(a.dtype()), device_to_string(dev))<<std::endl;
    }
    Tensor Ones(const std::vector<int64_t>& shape, DataType dtype){
        // 合法性判断
        if(shape.empty()) 
            throw std::runtime_error("ops::Ones: shape is empty");
        // 分发
        #ifdef BACKEND_CPU
            return OnesImpl<Device::CPU>::execute(shape,dtype);
        #endif
        #ifdef BACKEND_SYCL
            return OnesImpl<Device::SYCL>::execute(shape,dtype);
        #endif
        #ifdef BACKEND_CUDA
            return OnesImpl<Device::CUDA>::execute(shape,dtype);
        #endif
        #ifdef BACKEND_VULKAN
            return OnesImpl<Device::VULKAN>::execute(shape,dtype);
        #endif
    }
    Tensor Zeros(const std::vector<int64_t>& shape, DataType dtype){
        // 合法性判断
        if(shape.empty()) 
            throw std::runtime_error("ops::Zeros: shape is empty");
        // 分发
        #ifdef BACKEND_CPU
            return ZerosImpl<Device::CPU>::execute(shape,dtype);
        #endif
        #ifdef BACKEND_SYCL
            return ZerosImpl<Device::SYCL>::execute(shape,dtype);
        #endif
        #ifdef BACKEND_CUDA
            return ZerosImpl<Device::CUDA>::execute(shape,dtype);
        #endif
        #ifdef BACKEND_VULKAN
            return ZerosImpl<Device::VULKAN>::execute(shape,dtype);
        #endif
    }
    Tensor Fill(const std::vector<int64_t>& shape, DataType dtype,float val){
        // 合法性判断
        if(shape.empty()) 
            throw std::runtime_error("ops::Fill: shape is empty");
        // 分发
        #ifdef BACKEND_CPU
            return FillImpl<Device::CPU>::execute(shape,dtype,val);
        #endif
        #ifdef BACKEND_SYCL
            return FillImpl<Device::SYCL>::execute(shape,dtype,val);
        #endif
        #ifdef BACKEND_CUDA
            return FillImpl<Device::CUDA>::execute(shape,dtype,val);
        #endif
        #ifdef BACKEND_VULKAN
            return FillImpl<Device::VULKAN>::execute(shape,dtype,val);
        #endif
    }
    Tensor Random(const std::vector<int64_t>& shape, DataType dtype,float min,float max){
        // 合法性判断
        if(shape.empty()) 
            throw std::runtime_error("ops::Random: shape is empty");
        // 分发
        #ifdef BACKEND_CPU
            return RandomImpl<Device::CPU>::execute(shape,dtype,min,max);
        #endif
        #ifdef BACKEND_SYCL
            return RandomImpl<Device::SYCL>::execute(shape,dtype,min,max);
        #endif
        #ifdef BACKEND_CUDA
            return RandomImpl<Device::CUDA>::execute(shape,dtype,min,max);
        #endif
        #ifdef BACKEND_VULKAN
            return RandomImpl<Device::VULKAN>::execute(shape,dtype,min,max);
        #endif
    }
    
    Tensor Slice(const Tensor& t, const std::vector<std::pair<int64_t, int64_t>>& ranges){
        // 合法性判断
        if(ranges.empty())  
            throw std::runtime_error("Slice ranges cannot be empty");
        // ranges的整数对数量要合法
        if(ranges.size() != t.shape().size())   
            throw std::runtime_error("Slice ranges size must be equal to tensor shape size");
        // 每个整数对要合法: 左边必须小于等于右边,左边需要>=0,右边需要小于max_
        for(int64_t i = 0; i < ranges.size(); i++){
            if(ranges[i].first > ranges[i].second)
                throw std::runtime_error("first must be less than second");
            if(ranges[i].first < 0 || ranges[i].first > ranges[i].second) 
                throw std::runtime_error("first must be >= 0 and <= second");
            if(ranges[i].second > t.shape(i)) 
                throw std::runtime_error(std::format("second must be less than {}",t.shape(i)));
        }
        // 后端分发
        if(t.device() == Device::CPU){
            return SliceImpl<Device::CPU>::execute(t,ranges);
        }
        #ifdef BACKEND_CPU
            return SliceImpl<Device::CPU>::execute(t,ranges);
        #endif
        #ifdef BACKEND_SYCL
            return SliceImpl<Device::SYCL>::execute(t,ranges);
        #endif
        #ifdef BACKEND_CUDA
            return SliceImpl<Device::CUDA>::execute(t,ranges);
        #endif
        #ifdef BACKEND_VULKAN
            return SliceImpl<Device::VULKAN>::execute(t,ranges);
        #endif
    }
    void Add(Tensor& a,float b){
        if(a.numel() == 0 || b == 0.0f) return;
        // 后端实现分发
        if(a.device() == Device::CPU){
            return AddImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_SYCL
            return AddImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return AddImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return AddImpl<Device::VULKAN>::execute(a,b);
        #endif
    }
    void Sub(Tensor& a,float b){
        if(a.numel() == 0 || b == 0.0f) return;
        // 后端实现分发
        if(a.device() == Device::CPU){
            SubImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_SYCL
            SubImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            SubImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            SubImpl<Device::VULKAN>::execute(a,b);
        #endif
    }
    void Dot(Tensor& a,float b){
        if(a.numel() == 0 || b == 1.0f) return;
        // 后端实现分发
        if(a.device() == Device::CPU){
            DotImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_SYCL
            DotImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            DotImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            DotImpl<Device::VULKAN>::execute(a,b);
        #endif
    }
    void Div(Tensor& a,float b){
        // 检查
        if(a.numel() == 0 || b == 1.0f) return;
        if(b == 0.0f) throw std::runtime_error("div: Cann't division by zero");
        // 后端实现分发
        if(a.device() == Device::CPU){
            DivImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_SYCL
            DivImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            DivImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            DivImpl<Device::VULKAN>::execute(a,b);
        #endif
    }
    
    Tensor Add(const Tensor& a, float b){
        if(a.numel()==0 || b == 0.0f) return a.clone();

        // 后端实现分发
        if(a.device() == Device::CPU){
            return AddImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_CPU
            return AddImpl<Device::CPU>::execute(a,b);
        #endif
        #ifdef BACKEND_SYCL
            return AddImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return AddImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return AddImpl<Device::VULKAN>::execute(a,b);
        #endif
    }
    Tensor Sub(const Tensor& a, float b){
        if(a.numel()==0 || b == 0.0f) return a.clone();
        // 后端实现分发
        #ifdef BACKEND_CPU
            return SubImpl<Device::CPU>::execute(a,b);
        #endif
        #ifdef BACKEND_SYCL
            return SubImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return SubImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return SubImpl<Device::VULKAN>::execute(a,b);
        #endif
    }
    Tensor Dot(const Tensor& a, float b){
        if(a.numel()==0 || b == 1.0f) return a.clone();
        // 后端实现分发
        #ifdef BACKEND_CPU
            return DotImpl<Device::CPU>::execute(a,b);
        #endif
        #ifdef BACKEND_SYCL
            return DotImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return DotImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return DotImpl<Device::VULKAN>::execute(a,b);
        #endif
    }
    Tensor Div(const Tensor& a, float b){
        if(a.numel()==0 || b == 1.0f) 
            return a.clone();
        if(b == 0.0f) 
            throw std::runtime_error("Division by zero");

        // 后端实现分发
        #ifdef BACKEND_CPU
            return DivImpl<Device::CPU>::execute(a,b);
        #endif
        #ifdef BACKEND_SYCL
            return DivImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return DivImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return DivImpl<Device::VULKAN>::execute(a,b);
        #endif
    }
    Tensor Add(const Tensor& a,const Tensor& b){
        if(a.device() != b.device()) 
            throw std::runtime_error("Tensor Device mismatch!");
        if(a.shape().size() != b.shape().size()) 
            throw std::runtime_error("Tensor dims mismatch!");
        
        for(int i=0;i<a.shape().size();i++){
            if(a.shape(i) != b.shape(i)) throw std::runtime_error("Tensor shape mismatch!");
        }
        // 后端实现分发
        if(a.device() == Device::CPU){
            return AddImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_CPU
            return AddImpl<Device::CPU>::execute(a,b);
        #endif
        #ifdef BACKEND_SYCL
            return AddImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return AddImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return AddImpl<Device::VULKAN>::execute(a,b);
        #endif
    }
    Tensor Sub(const Tensor& a,const Tensor& b){
        // 设备合法性检测
        if(a.device() != b.device()) throw std::runtime_error("Tensor Device mismatch!");
        // 维度合法性检测 
        if(a.shape().size() != b.shape().size()) throw std::runtime_error("Tensor dim mismatch!");
        for(int i=0;i<a.shape().size();i++){
            if(a.shape(i) != b.shape(i)) throw std::runtime_error("Tensor shape mismatch!");
        }
        // 广播
        // todo....
        // 后端实现分发
        if(a.device() == Device::CPU){
            return SubImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_CPU
            return SubImpl<Device::CPU>::execute(a,b);
        #endif
        #ifdef BACKEND_SYCL
            return SubImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return SubImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return SubImpl<Device::VULKAN>::execute(a,b);
        #endif
    }
    Tensor Dot(const Tensor& a,const Tensor& b){
        // 设备合法性检测
        if(a.device() != b.device()) throw std::runtime_error("Tensor Device mismatch!");
        // 维度合法性检测 
        if(a.shape().size() != b.shape().size()) throw std::runtime_error("Tensor dim mismatch!");
        for(int i=0;i<a.shape().size();i++){
            if(a.shape(i) != b.shape(i)) throw std::runtime_error("Tensor shape mismatch!");
        }
        // 后端实现分发
        if(a.device() == Device::CPU){
            return DotImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_CPU
            return DotImpl<Device::CPU>::execute(a,b);
        #endif
        #ifdef BACKEND_SYCL
            return DotImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return DotImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return DotImpl<Device::VULKAN>::execute(a,b);
        #endif
    }
    Tensor Div(const Tensor& a,const Tensor& b){
        if(a.device()!=b.device()) throw std::runtime_error("Device mismatch!");
        for(int i=0;i<a.shape().size();i++){
            if(a.shape(i) != b.shape(i)) throw std::runtime_error("Tensor shape mismatch!");
        }
        // 后端实现分发
        if(a.device() == Device::CPU){
            return DivImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_CPU
            return DivImpl<Device::CPU>::execute(a,b);
        #endif
        #ifdef BACKEND_SYCL
            return DivImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return DivImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return DivImpl<Device::VULKAN>::execute(a,b);
        #endif
    }
    
    void  Add(const Tensor& a,const Tensor& b,Tensor& dst){
        if(a.device() != b.device()) 
            throw std::runtime_error("Tensor Device mismatch!");
        if(a.shape().size() != b.shape().size()) 
            throw std::runtime_error("Tensor dims mismatch!");
        
        for(int i=0;i<a.shape().size();i++){
            if(a.shape(i) != b.shape(i)) throw std::runtime_error("Tensor shape mismatch!");
        }
        // 后端实现分发
        if(a.device() == Device::CPU){
            AddImpl<Device::CPU>::execute(a,b,dst);
        }
        #ifdef BACKEND_CPU
            AddImpl<Device::CPU>::execute(a,b,dst);
        #endif
        #ifdef BACKEND_SYCL
            AddImpl<Device::SYCL>::execute(a,b,dst);
        #endif
        #ifdef BACKEND_CUDA
            AddImpl<Device::CUDA>::execute(a,b,dst);
        #endif
        #ifdef BACKEND_VULKAN
            AddImpl<Device::VULKAN>::execute(a,b,dst);
        #endif
    }
    
    Tensor Mul(const Tensor& a,const Tensor& b){
        // 合法性判断
        if(a.device()!=b.device()) 
            throw std::runtime_error("Tensors a and b device mismatch!");
        if(a.shape().size() != b.shape().size()) 
            throw  std::runtime_error(std::format("Tensors a and b must have the same rank,but got {} and {}",a.shape().size(),b.shape().size()));
        if(a.shape().size() != 2 && a.shape().size() != 3)
            throw  std::runtime_error(std::format("Tensors a and b dims must be 2 or 3,but got {}",a.shape().size()));
        if(a.shape().size() == 3){ // [3,20,30] @ [3,30,10]
            if(a.shape(0) != b.shape(0)) 
                throw  std::runtime_error("Tensors a and b must have the same batch size");
            if(a.shape(2) != b.shape(1))
                throw  std::runtime_error(std::format("Tensors common dim must match,but {} and {}",a.shape(2) , b.shape(1)));
        }else if(a.shape().size() == 2){
            if (a.shape(1) != b.shape(0))
                throw std::runtime_error(std::format("Tensors common dim must match,but {} and {}",a.shape(0) , b.shape(0)));
        }
        // 后端实现分发
        if(a.device() == Device::CPU){
            return MulImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_CPU
            return MulImpl<Device::CPU>::execute(a,b);
        #endif
        #ifdef BACKEND_SYCL
            return MulImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return MulImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return MulImpl<Device::VULKAN>::execute(a,b);
        #endif
    }
    Tensor Abs(const Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        // 后端实现分发
        if(a.device() == Device::CPU){
            return AbsImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_CPU
            return AbsImpl<Device::CPU>::execute(a);
        #endif
        #ifdef BACKEND_SYCL
            return AbsImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            return AbsImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            return AbsImpl<Device::VULKAN>::execute(a);
        #endif
    }
    Tensor Sin(const Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        // 后端实现分发
        if(a.device() == Device::CPU){
            return SinImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_CPU
            return SinImpl<Device::CPU>::execute(a);
        #endif
        #ifdef BACKEND_SYCL
            return SinImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            return SinImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            return SinImpl<Device::VULKAN>::execute(a);
        #endif
    }
    Tensor Cos(const Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        
        // 后端实现分发
        if(a.device() == Device::CPU){
            return CosImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_CPU
            return CosImpl<Device::CPU>::execute(a);
        #endif
        #ifdef BACKEND_SYCL
            return CosImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            return CosImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            return CosImpl<Device::VULKAN>::execute(a);
        #endif
    }
    Tensor Tan(const Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        
        // 后端实现分发
        if(a.device() == Device::CPU){
            return TanImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_CPU
            return TanImpl<Device::CPU>::execute(a);
        #endif
        #ifdef BACKEND_SYCL
            return TanImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            return TanImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            return TanImpl<Device::VULKAN>::execute(a);
        #endif
    }
    Tensor Exp(const Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        
        // 后端实现分发
        if(a.device() == Device::CPU){
            return ExpImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_CPU
            return ExpImpl<Device::CPU>::execute(a);
        #endif
        #ifdef BACKEND_SYCL
            return ExpImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            return ExpImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            return ExpImpl<Device::VULKAN>::execute(a);
        #endif
    }
    Tensor Relu(const Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");

        // 后端实现分发
        if(a.device() == Device::CPU){
            return ReluImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_CPU
            return ReluImpl<Device::CPU>::execute(a);
        #endif
        #ifdef BACKEND_SYCL
            return ReluImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            return ReluImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            return ReluImpl<Device::VULKAN>::execute(a);
        #endif
    }
    Tensor Silu(const Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
       // 后端实现分发
        if(a.device() == Device::CPU){
            return SiluImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_CPU
            return SiluImpl<Device::CPU>::execute(a);
        #endif
        #ifdef BACKEND_SYCL
            return SiluImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            return SiluImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            return SiluImpl<Device::VULKAN>::execute(a);
        #endif
    }
    Tensor Tanh(const Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
         // 后端实现分发
        if(a.device() == Device::CPU){
            return TanhImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_CPU
            return TanhImpl<Device::CPU>::execute(a);
        #endif
        #ifdef BACKEND_SYCL
            return TanhImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            return TanhImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            return TanhImpl<Device::VULKAN>::execute(a);
        #endif
    }
    Tensor Sqrt(const Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");

        // 暂时不支持虚数，所以最好全都要是非负数
        if (ops::Any((a < 0),1))
            throw std::runtime_error("tensor must be non-negative");

        // 后端实现分发
        if(a.device() == Device::CPU){
            return SqrtImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_CPU
            return SqrtImpl<Device::CPU>::execute(a);
        #endif
        #ifdef BACKEND_SYCL
            return SqrtImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            return SqrtImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            return SqrtImpl<Device::VULKAN>::execute(a);
        #endif
    }
    Tensor Sigmoid(const Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        // 后端实现分发
        if(a.device() == Device::CPU){
            return SigmoidImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_CPU
            return SigmoidImpl<Device::CPU>::execute(a);
        #endif
        #ifdef BACKEND_SYCL
            return SigmoidImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            return SigmoidImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            return SigmoidImpl<Device::VULKAN>::execute(a);
        #endif
    }
    Tensor Pow(const Tensor& a,float val){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        
        // 后端实现分发
        if(a.device() == Device::CPU){
            return PowImpl<Device::CPU>::execute(a,val);
        }
        #ifdef BACKEND_CPU
            return PowImpl<Device::CPU>::execute(a,val);
        #endif
        #ifdef BACKEND_SYCL
            return PowImpl<Device::SYCL>::execute(a,val);
        #endif
        #ifdef BACKEND_CUDA
            return PowImpl<Device::CUDA>::execute(a,val);
        #endif
        #ifdef BACKEND_VULKAN
            return PowImpl<Device::VULKAN>::execute(a,val);
        #endif
    }
    Tensor Log(const Tensor& a,float val){
        if (val < 0 || val == 1)
            throw std::runtime_error("The base cannot be less than or equal to 0, and it cannot be 1");

        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        
        // log 的 定义域必须大于0
        if (ops::Any((a < 0),1))
            throw std::runtime_error("log: tensor must be non-negative");
            
        // 后端实现分发
        if(a.device() == Device::CPU){
            return LogImpl<Device::CPU>::execute(a,val);
        }
        #ifdef BACKEND_CPU
            return LogImpl<Device::CPU>::execute(a,val);
        #endif
        #ifdef BACKEND_SYCL
            return LogImpl<Device::SYCL>::execute(a,val);
        #endif
        #ifdef BACKEND_CUDA
            return LogImpl<Device::CUDA>::execute(a,val);
        #endif
        #ifdef BACKEND_VULKAN
            return LogImpl<Device::VULKAN>::execute(a,val);
        #endif
    }
    
    Tensor Softmax(const Tensor& a,int axis){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        if(std::abs(axis) >= a.shape().size()) 
            throw std::runtime_error("axis out of range");
        // 后端实现分发
        if(a.device() == Device::CPU){
            return SoftmaxImpl<Device::CPU>::execute(a,axis);
        }
        #ifdef BACKEND_CPU
            return SoftmaxImpl<Device::CPU>::execute(a,axis);
        #endif
        #ifdef BACKEND_SYCL
            return SoftmaxImpl<Device::SYCL>::execute(a,axis);
        #endif
        #ifdef BACKEND_CUDA
            return SoftmaxImpl<Device::CUDA>::execute(a,axis);
        #endif
        #ifdef BACKEND_VULKAN
            return SoftmaxImpl<Device::VULKAN>::execute(a,axis);
        #endif
    }
    Tensor Clamp(const Tensor& a,float min,float max){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        if(min > max) 
            throw std::runtime_error("min must be less than max");    // 后端实现分发
        if(a.device() == Device::CPU){
            return ClampImpl<Device::CPU>::execute(a,min,max);
        }
        #ifdef BACKEND_CPU
            return ClampImpl<Device::CPU>::execute(a,min,max);
        #endif
        #ifdef BACKEND_SYCL
            return ClampImpl<Device::SYCL>::execute(a,min,max);
        #endif
        #ifdef BACKEND_CUDA
            return ClampImpl<Device::CUDA>::execute(a,min,max);
        #endif
        #ifdef BACKEND_VULKAN
            return ClampImpl<Device::VULKAN>::execute(a,min,max);
        #endif
    }
    void Abs(Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        // 后端实现分发
        if(a.device() == Device::CPU){
            AbsImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_SYCL
            AbsImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            AbsImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            AbsImpl<Device::VULKAN>::execute(a);
        #endif
    }
    void Sin(Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        if(a.dtype() < DataType::BFLOAT16){
            throw std::runtime_error(std::format("{} of tensor a is not supported",dtype_to_string(a.dtype())));
        }
        // 后端实现分发
        if(a.device() == Device::CPU){
            SinImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_SYCL
            SinImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            SinImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            SinImpl<Device::VULKAN>::execute(a);
        #endif
    }
    void Cos(Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        if(a.dtype() < DataType::BFLOAT16){
            throw std::runtime_error(std::format("{} of tensor a is not supported",dtype_to_string(a.dtype())));
        }
        // 后端实现分发
        if(a.device() == Device::CPU){
            CosImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_SYCL
            CosImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            CosImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            CosImpl<Device::VULKAN>::execute(a);
        #endif
    }
    void Tan(Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        if(a.dtype() < DataType::BFLOAT16){
            throw std::runtime_error(std::format("{} of tensor a is not supported",dtype_to_string(a.dtype())));
        }
        // 后端实现分发
        if(a.device() == Device::CPU){
            TanImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_SYCL
            TanImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            TanImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            TanImpl<Device::VULKAN>::execute(a);
        #endif
    }
    void Relu(Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        // 后端实现分发
        if(a.device() == Device::CPU){
            ReluImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_SYCL
            ReluImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            ReluImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            ReluImpl<Device::VULKAN>::execute(a);
        #endif
    }
    void Silu(Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        if(a.dtype() < DataType::BFLOAT16){
            throw std::runtime_error(std::format("{} of tensor a is not supported",dtype_to_string(a.dtype())));
        }
        // 后端实现分发
        if(a.device() == Device::CPU){
            SiluImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_SYCL
            SiluImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            SiluImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            SiluImpl<Device::VULKAN>::execute(a);
        #endif
    }
    void Tanh(Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        if(a.dtype() < DataType::BFLOAT16){
            throw std::runtime_error(std::format("{} of tensor a is not supported",dtype_to_string(a.dtype())));
        }
        // 后端实现分发
        if(a.device() == Device::CPU){
            TanhImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_SYCL
            TanhImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            TanhImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            TanhImpl<Device::VULKAN>::execute(a);
        #endif
    }
    void Sigmoid(Tensor& a){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        if(a.dtype() < DataType::BFLOAT16){
            throw std::runtime_error(std::format("{} of tensor a is not supported",dtype_to_string(a.dtype())));
        }
        // 后端实现分发
        if(a.device() == Device::CPU){
            SigmoidImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_SYCL
            SigmoidImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            SigmoidImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            SigmoidImpl<Device::VULKAN>::execute(a);
        #endif
    }
    
    void Clamp(Tensor& a,float min,float max){
        if(a.data() == nullptr || a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        if(min > max) 
            throw std::runtime_error("min must be less than max");
        // 后端实现分发
        if(a.device() == Device::CPU){
            ClampImpl<Device::CPU>::execute(a,min,max);
        }
        #ifdef BACKEND_SYCL
            ClampImpl<Device::SYCL>::execute(a,min,max);
        #endif
        #ifdef BACKEND_CUDA
            ClampImpl<Device::CUDA>::execute(a,min,max);
        #endif
        #ifdef BACKEND_VULKAN
            ClampImpl<Device::VULKAN>::execute(a,min,max);
        #endif
    }
    float Sum(const Tensor& a){
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");

        // 后端实现分发
        if(a.device() == Device::CPU){
            return SumImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_CPU
            return SumImpl<Device::CPU>::execute(a);
        #endif
        #ifdef BACKEND_SYCL
            return SumImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            return SumImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            return SumImpl<Device::VULKAN>::execute(a);
        #endif
    }
    float Min(const Tensor& a){
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        // 后端实现分发
        if(a.device() == Device::CPU){
            return MinImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_CPU
            return MinImpl<Device::CPU>::execute(a);
        #endif
        #ifdef BACKEND_SYCL
            return MinImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            return MinImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            return MinImpl<Device::VULKAN>::execute(a);
        #endif
    }
    float Max(const Tensor& a){
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        // 后端实现分发
        if(a.device() == Device::CPU){
            return MaxImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_CPU
            return MaxImpl<Device::CPU>::execute(a);
        #endif
        #ifdef BACKEND_SYCL
            return MaxImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            return MaxImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            return MaxImpl<Device::VULKAN>::execute(a);
        #endif
    }
    float Mean(const Tensor& a){
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        // 后端实现分发
        if(a.device() == Device::CPU){
            return MeanImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_CPU
            return MeanImpl<Device::CPU>::execute(a);
        #endif
        #ifdef BACKEND_SYCL
            return MeanImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            return MeanImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            return MeanImpl<Device::VULKAN>::execute(a);
        #endif
    }
    Tensor Sum(const Tensor& a,int axis){
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        axis = axis < 0 ? axis + a.shape().size() : axis;
        // 判断axis 合法性
        if(axis >= a.shape().size())
            throw std::runtime_error("axis is out of range");

        if(a.device() == Device::CPU){
            return SumImpl<Device::CPU>::execute(a,axis);
        }
        #ifdef BACKEND_CPU
            return SumImpl<Device::CPU>::execute(a,axis);
        #endif
        #ifdef BACKEND_SYCL
            return SumImpl<Device::SYCL>::execute(a,axis);
        #endif
        #ifdef BACKEND_CUDA
            return SumImpl<Device::CUDA>::execute(a,axis);
        #endif
        #ifdef BACKEND_VULKAN
            return SumImpl<Device::VULKAN>::execute(a,axis);
        #endif
    }
    Tensor Min(const Tensor& a,int axis){
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        axis = axis < 0 ? axis + a.shape().size() : axis;
        // 判断axis 合法性
        if(axis >= a.shape().size())
            throw std::runtime_error("axis is out of range");
        if(a.device() == Device::CPU){
            return MinImpl<Device::CPU>::execute(a,axis);
        }
        #ifdef BACKEND_CPU
            return MinImpl<Device::CPU>::execute(a,axis);
        #endif
        #ifdef BACKEND_SYCL
            return MinImpl<Device::SYCL>::execute(a,axis);
        #endif
        #ifdef BACKEND_CUDA
            return MinImpl<Device::CUDA>::execute(a,axis);
        #endif
        #ifdef BACKEND_VULKAN
            return MinImpl<Device::VULKAN>::execute(a,axis);
        #endif
    }
    Tensor Max(const Tensor& a,int axis){
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        axis = axis < 0 ? axis + a.shape().size() : axis;
        // 判断axis 合法性
        if(axis >= a.shape().size())
            throw std::runtime_error("axis is out of range");
        if(a.device() == Device::CPU){
            return MaxImpl<Device::CPU>::execute(a,axis);
        }
        #ifdef BACKEND_CPU
            return MaxImpl<Device::CPU>::execute(a,axis);
        #endif
        #ifdef BACKEND_SYCL
            return MaxImpl<Device::SYCL>::execute(a,axis);
        #endif
        #ifdef BACKEND_CUDA
            return MaxImpl<Device::CUDA>::execute(a,axis);
        #endif
        #ifdef BACKEND_VULKAN
            return MaxImpl<Device::VULKAN>::execute(a,axis);
        #endif
    }
    Tensor Mean(const Tensor& a,int axis){
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        axis = axis < 0 ? axis + a.shape().size() : axis;
        // 判断axis 合法性
        if(axis >= a.shape().size())
            throw std::runtime_error("axis is out of range");

        if(a.device() == Device::CPU){
            return MeanImpl<Device::CPU>::execute(a,axis);
        }
        #ifdef BACKEND_CPU
            return MeanImpl<Device::CPU>::execute(a,axis);
        #endif
        #ifdef BACKEND_SYCL
            return MeanImpl<Device::SYCL>::execute(a,axis);
        #endif
        #ifdef BACKEND_CUDA
            return MeanImpl<Device::CUDA>::execute(a,axis);
        #endif
        #ifdef BACKEND_VULKAN
            return MeanImpl<Device::VULKAN>::execute(a,axis);
        #endif
    }
    Tensor Typecast(const Tensor& a,DataType dst_type){
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        if (!is_cast_valid(a.dtype(), dst_type))
            throw std::runtime_error(std::format("Invalid typecast from {} to {}", dtype_to_string(a.dtype()), dtype_to_string(dst_type)));
        if (a.dtype() == dst_type)
            return a;
        // 后端实现分发
        if(a.device() == Device::CPU){
            return TypecastImpl<Device::CPU>::execute(a,dst_type);
        }
        #ifdef BACKEND_CPU
            return TypecastImpl<Device::CPU>::execute(a,dst_type);
        #endif
        #ifdef BACKEND_SYCL
            return TypecastImpl<Device::SYCL>::execute(a,dst_type);
        #endif
        #ifdef BACKEND_CUDA
            return TypecastImpl<Device::CUDA>::execute(a,dst_type);
        #endif
        #ifdef BACKEND_VULKAN
            return TypecastImpl<Device::VULKAN>::execute(a,dst_type);
        #endif
    }
    Tensor Concat(const std::vector<Tensor>& tensors, int dim){
        // 合法性判断
        if(tensors.empty()) throw std::invalid_argument("tensors is empty");
        if(tensors.size() == 1) return tensors[0];

        DataType dtype = tensors[0].dtype();
        Device device = tensors[0].device();
        const auto& first_shape = tensors[0].shape();
        for (const auto& t : tensors) {
            if (t.dtype() != dtype) throw std::invalid_argument("tensors must have same dtype");
            if (t.shape().size() != first_shape.size()) throw std::invalid_argument("tensors must have same number of dimensions");
            for (int i = 0; i < first_shape.size(); ++i) {
                if (i != dim && t.shape()[i] != first_shape[i]) 
                    throw std::invalid_argument("tensors must have same shape except concatenation dimension");
            }
        }
        // 后端实现分发
        if(device == Device::CPU){
            return ConcatImpl<Device::CPU>::execute(tensors,dim);
        }
        #ifdef BACKEND_CPU
            return ConcatImpl<Device::CPU>::execute(tensors,dim);
        #endif
        #ifdef BACKEND_SYCL
            return ConcatImpl<Device::SYCL>::execute(tensors,dim);
        #endif
        #ifdef BACKEND_CUDA
            return ConcatImpl<Device::CUDA>::execute(tensors,dim);
        #endif
        #ifdef BACKEND_VULKAN
            return ConcatImpl<Device::VULKAN>::execute(tensors,dim);
        #endif
    }
    void Transpose(Tensor& a){
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        if(a.shape().size()!=2) 
            throw std::runtime_error("transpose only support 2D tensor");
        
        // 后端实现分发
        if(a.device() == Device::CPU){
            TransposeImpl<Device::CPU>::execute(a);
        }
        #ifdef BACKEND_SYCL
            TransposeImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            TransposeImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            TransposeImpl<Device::VULKAN>::execute(a);
        #endif
    }
    void Transpose(const Tensor& a,Tensor& dst,std::initializer_list<int64_t> axes){
        if(a.shape().empty()) 
            throw std::runtime_error("Input tensor must be not null.");
        if(a.shape().size() != axes.size()) 
            throw std::runtime_error("transpose shape and axes size not match");
        if(a.numel() != dst.numel()){
            throw std::runtime_error("dst numel not enough");
        }
        // 后端实现分发
        if(a.device() == Device::CPU){
            TransposeImpl<Device::CPU>::execute(a,dst,axes);
        }
        #ifdef BACKEND_CPU
            TransposeImpl<Device::CPU>::execute(a,dst,axes);
        #endif
        #ifdef BACKEND_SYCL
            TransposeImpl<Device::SYCL>::execute(a,dst,axes);
        #endif
        #ifdef BACKEND_CUDA
            TransposeImpl<Device::CUDA>::execute(a,dst,axes);
        #endif
        #ifdef BACKEND_VULKAN
            TransposeImpl<Device::VULKAN>::execute(a,dst,axes);
        #endif
    }

    Tensor Transpose(const Tensor& a,std::initializer_list<int64_t> axes){
        if(a.shape().empty()) 
            throw std::runtime_error("Input tensor must be not null.");
        if(a.shape().size() != axes.size()) 
            throw std::runtime_error("transpose shape and axes size not match");

        // 后端实现分发
        if(a.device() == Device::CPU){
            return TransposeImpl<Device::CPU>::execute(a,axes);
        }
        #ifdef BACKEND_CPU
            return TransposeImpl<Device::CPU>::execute(a,axes);
        #endif
        #ifdef BACKEND_SYCL
            return TransposeImpl<Device::SYCL>::execute(a,axes);
        #endif
        #ifdef BACKEND_CUDA
            return TransposeImpl<Device::CUDA>::execute(a,axes);
        #endif
        #ifdef BACKEND_VULKAN
            return TransposeImpl<Device::VULKAN>::execute(a,axes);
        #endif
    }
    Tensor Equal(const Tensor& a,const Tensor& b) {
        if(a.device()!=b.device()) 
            throw std::runtime_error("Tensors a and b device mismatch!");
        if(a.shape().size() != b.shape().size()) 
            throw  std::runtime_error(std::format("Tensors a and b must have the same rank,but got {} and {}",a.shape().size(),b.shape().size()));
        // 后端实现分发
        if(a.device() == Device::CPU){
            return EqualImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_CPU
            return EqualImpl<Device::CPU>::execute(a,b);
        #endif
        #ifdef BACKEND_SYCL
            return EqualImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return EqualImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return EqualImpl<Device::VULKAN>::execute(a,b);
        #endif
    }

    Tensor NotEqual(const Tensor& a,const Tensor& b){
        if(a.device()!=b.device()) 
            throw std::runtime_error("Tensors a and b device mismatch!");
        if(a.shape().size() != b.shape().size()) 
            throw  std::runtime_error(std::format("Tensors a and b must have the same rank,but got {} and {}",a.shape().size(),b.shape().size()));

        if(a.device() == Device::CPU){
            return NotEqualImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_CPU
            return NotEqualImpl<Device::CPU>::execute(a,b);
        #endif
        #ifdef BACKEND_SYCL
            return NotEqualImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return NotEqualImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return NotEqualImpl<Device::VULKAN>::execute(a,b);
        #endif
     }
    Tensor Greater(const Tensor& a,const Tensor& b){
        if(a.device()!=b.device()) 
            throw std::runtime_error("Tensors a and b device mismatch!");
        if(a.shape().size() != b.shape().size()) 
            throw  std::runtime_error(std::format("Tensors a and b must have the same rank,but got {} and {}",a.shape().size(),b.shape().size()));

        if(a.device() == Device::CPU){
            return GreaterImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_CPU
            return GreaterImpl<Device::CPU>::execute(a,b);
        #endif
        #ifdef BACKEND_SYCL
            return GreaterImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return GreaterImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return GreaterImpl<Device::VULKAN>::execute(a,b);
        #endif
     }
    Tensor Less(const Tensor& a,const Tensor& b){
        if(a.device()!=b.device()) 
            throw std::runtime_error("Tensors a and b device mismatch!");
        if(a.shape().size() != b.shape().size()) 
            throw  std::runtime_error(std::format("Tensors a and b must have the same rank,but got {} and {}",a.shape().size(),b.shape().size()));

        if(a.device() == Device::CPU){
            return LessImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_CPU
            return LessImpl<Device::CPU>::execute(a,b);
        #endif
        #ifdef BACKEND_SYCL
            return LessImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return LessImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return LessImpl<Device::VULKAN>::execute(a,b);
        #endif
     }
    Tensor GreaterEqual(const Tensor& a,const Tensor& b){
        if(a.device()!=b.device()) 
            throw std::runtime_error("Tensors a and b device mismatch!");
        if(a.shape().size() != b.shape().size()) 
            throw  std::runtime_error(std::format("Tensors a and b must have the same rank,but got {} and {}",a.shape().size(),b.shape().size()));

        if(a.device() == Device::CPU){
            return GreaterEqualImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_CPU
            return GreaterEqualImpl<Device::CPU>::execute(a,b);
        #endif
        #ifdef BACKEND_SYCL
            return GreaterEqualImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return GreaterEqualImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return GreaterEqualImpl<Device::VULKAN>::execute(a,b);
        #endif
     }
    Tensor LessEqual(const Tensor& a,const Tensor& b){
        if(a.device()!=b.device()) 
            throw std::runtime_error("Tensors a and b device mismatch!");
        if(a.shape().size() != b.shape().size()) 
            throw  std::runtime_error(std::format("Tensors a and b must have the same rank,but got {} and {}",a.shape().size(),b.shape().size()));

        if(a.device() == Device::CPU){
            return LessEqualImpl<Device::CPU>::execute(a,b);
        }
        #ifdef BACKEND_CPU
            return LessEqualImpl<Device::CPU>::execute(a,b);
        #endif
        #ifdef BACKEND_SYCL
            return LessEqualImpl<Device::SYCL>::execute(a,b);
        #endif
        #ifdef BACKEND_CUDA
            return LessEqualImpl<Device::CUDA>::execute(a,b);
        #endif
        #ifdef BACKEND_VULKAN
            return LessEqualImpl<Device::VULKAN>::execute(a,b);
        #endif
    }
    bool All(const Tensor &a,float val = 0.0f){
        // 判断张量 t 中是否所有元素都非零（true）。
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");

         #ifdef BACKEND_CPU
            return AllImpl<Device::CPU>::execute(a,val);
        #endif
        #ifdef BACKEND_SYCL
            return AllImpl<Device::SYCL>::execute(a,val);
        #endif
        #ifdef BACKEND_CUDA
            return AllImpl<Device::CUDA>::execute(a,val);
        #endif
        #ifdef BACKEND_VULKAN
            return AllImpl<Device::VULKAN>::execute(a,val);
        #endif
    }
    bool Any(const Tensor &a,float val = 0.0f){
        // 判断张量 t 中是否至少有一个元素非零（true）。
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");

         #ifdef BACKEND_CPU
            return AnyImpl<Device::CPU>::execute(a,val);
        #endif
        #ifdef BACKEND_SYCL
            return AnyImpl<Device::SYCL>::execute(a,val);
        #endif
        #ifdef BACKEND_CUDA
            return AnyImpl<Device::CUDA>::execute(a,val);
        #endif
        #ifdef BACKEND_VULKAN
            return AnyImpl<Device::VULKAN>::execute(a,val);
        #endif
    }
    size_t Nonzero(const Tensor &a){
        // 统计张量 t 中非零元素的数量。
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");

         #ifdef BACKEND_CPU
            return NonZeroImpl<Device::CPU>::execute(a);
        #endif
        #ifdef BACKEND_SYCL
            return NonZeroImpl<Device::SYCL>::execute(a);
        #endif
        #ifdef BACKEND_CUDA
            return NonZeroImpl<Device::CUDA>::execute(a);
        #endif
        #ifdef BACKEND_VULKAN
            return NonZeroImpl<Device::VULKAN>::execute(a);
        #endif
    }
   
    Tensor Argmax(const Tensor &a, int axis){
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        // axis 可以为负数
        axis = axis < 0 ? axis + a.shape().size() : axis;
        
        // 判断axis 合法性
        if(axis >= a.shape().size())
            throw std::runtime_error("axis is out of range");

        #ifdef BACKEND_CPU
            return ArgMaxImpl<Device::CPU>::execute(a,axis);
        #endif
        #ifdef BACKEND_SYCL
            return ArgMaxImpl<Device::SYCL>::execute(a,axis);
        #endif
        #ifdef BACKEND_CUDA
            return ArgMaxImpl<Device::CUDA>::execute(a,axis);
        #endif
        #ifdef BACKEND_VULKAN
            return ArgMaxImpl<Device::VULKAN>::execute(a,axis);
        #endif
    }

    Tensor Argmin(const Tensor &a, int axis){
        if(a.data() == nullptr|| a.numel() == 0) 
            throw std::runtime_error("tensor a is null");
        // axis 可以为负数
        axis = axis < 0 ? axis + a.shape().size() : axis;
        
        // 判断axis 合法性
        if(axis >= a.shape().size())
            throw std::runtime_error("axis is out of range");

        #ifdef BACKEND_CPU
            return ArgMinImpl<Device::CPU>::execute(a,axis);
        #endif
        #ifdef BACKEND_SYCL
            return ArgMinImpl<Device::SYCL>::execute(a,axis);
        #endif
        #ifdef BACKEND_CUDA
            return ArgMinImpl<Device::CUDA>::execute(a,axis);
        #endif
        #ifdef BACKEND_VULKAN
            return ArgMinImpl<Device::VULKAN>::execute(a,axis);
        #endif
    }
}