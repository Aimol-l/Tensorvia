// 由 tools/gen_facade.py 从 tools/facade_exports.def 生成，请勿手改。
//
// mcpp 路径的模块包装：全局模块片段 include 头文件事实源，export 出公开
// API 面。没有任何逻辑代码；清单与头文件的一致性由 --check 守卫，名字
// 的存在性由 tests/facade_api.cpp 编译守卫。
module;
#include "interface/tensorvia.hpp"

export module tensorvia;

export using ::float32;
export using ::float64;
export using ::float16;
export using ::bfloat16;
export using ::kMaxTensorDims;
export using ::DimArray;
export using ::compute_type_helper;
export using ::compute_type_t;
export using ::OpType;
export using ::Device;
export using ::DataType;
export using ::op_to_string;
export using ::device_to_string;
export using ::dtype_to_string;
export using ::make_pipeline_key;
export using ::reduction_type;
export using ::float_reduction_type;
export using ::calc_dtype_size;
export using ::CppDTypeVariant;
export using ::ConstCppDTypeVariant;
export using ::make_variant;
export using ::data_as_const_variant;
export using ::dispatch_dtype;
export using ::valid_casts;
export using ::compute_type;
export using ::is_cast_valid;
export using ::calc_numel;
export using ::Metadata;
export using ::TensorImpl;
export using ::Tensor;
export using ::ContextImpl;
export using ::CPUContext;
export using ::TensorImplFactory;
export using ::BackendInitializer;
export using ::register_tensor_impl;
export using ::register_backend_initializer;
export using ::create_tensor_impl;
export using ::copy_device_to_host;
export using ::copy_host_to_device;

export namespace tensorvia {
using ::tensorvia::version;
}

export namespace ops {
using ::ops::println;
using ::ops::export_csv;
using ::ops::Ones;
using ::ops::Zeros;
using ::ops::Fill;
using ::ops::Random;
using ::ops::Randn;
using ::ops::Arange;
using ::ops::Linspace;
using ::ops::Eye;
using ::ops::Slice;
using ::ops::Add;
using ::ops::Sub;
using ::ops::Mul;
using ::ops::Div;
using ::ops::MatMul;
using ::ops::Abs;
using ::ops::Relu;
using ::ops::Sin;
using ::ops::Cos;
using ::ops::Tan;
using ::ops::Silu;
using ::ops::Tanh;
using ::ops::Sigmoid;
using ::ops::Pow;
using ::ops::Log;
using ::ops::Clamp;
using ::ops::Exp;
using ::ops::Sqrt;
using ::ops::Softmax;
using ::ops::LogSoftmax;
using ::ops::Erf;
using ::ops::LayerNorm;
using ::ops::RmsNorm;
using ::ops::Sum;
using ::ops::Min;
using ::ops::Max;
using ::ops::Mean;
using ::ops::Prod;
using ::ops::Var;
using ::ops::Std;
using ::ops::Norm;
using ::ops::Cumsum;
using ::ops::MseLoss;
using ::ops::NllLoss;
using ::ops::CrossEntropyLoss;
using ::ops::Typecast;
using ::ops::Concat;
using ::ops::Transpose;
using ::ops::Equal;
using ::ops::NotEqual;
using ::ops::Greater;
using ::ops::Less;
using ::ops::GreaterEqual;
using ::ops::LessEqual;
using ::ops::Where;
using ::ops::All;
using ::ops::Any;
using ::ops::Nonzero;
using ::ops::Argmax;
using ::ops::Argmin;
using ::ops::IndexSelect;
using ::ops::Gather;
using ::ops::Temp;
}
