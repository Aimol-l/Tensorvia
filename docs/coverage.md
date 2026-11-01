# 算子 × 后端覆盖矩阵

> 由 `tools/coverage_matrix.py` 生成，请勿手改。漂移守门：`python3 -I tools/coverage_matrix.py --check`（CI 同款）。
> 人工标注唯一来源：`tools/coverage_overrides.toml`（组合实现与显式拒绝不从代码推断）。

**native** = 后端自己的实现（`XxxImpl<Device::X>` 全特化，或 `src/core/ops.cpp` 中标注的 CPU 直实现）；**emulated** = 非原生但可用（由其他算子拼成，或在宿主生成后一次性搬运到设备）；**unsupported** = 显式拒绝（设备上调用抛错）。

矩阵共 77 个算子。

| 算子 | CPU | CUDA | SYCL | VULKAN | 备注 |
|------|-----|------|------|--------|------|
| `Abs` | native | native | native | native |  |
| `Add` | native | native | native | native |  |
| `Addmm` | emulated | emulated | emulated | emulated | MatMul + Mul/Add 缩放与广播相加 |
| `All` | native | native | native | native |  |
| `Any` | native | native | native | native |  |
| `Arange` | native | emulated | emulated | emulated | 宿主生成后 to_device 搬运 |
| `Argmax` | native | native | native | native |  |
| `Argmin` | native | native | native | native |  |
| `Bmm` | emulated | emulated | emulated | emulated | 形状校验后转 MatMul |
| `Chunk` | emulated | emulated | emulated | emulated | 块长换算后转 Split |
| `Clamp` | native | native | native | native |  |
| `Concat` | native | native | native | native |  |
| `Conv2d` | native | native | native | native |  |
| `Cos` | native | native | native | native |  |
| `CountNonzero` | emulated | emulated | emulated | emulated | Typecast 归一 + Nonzero 取计数 |
| `CrossEntropyLoss` | emulated | unsupported | unsupported | unsupported | CPU: LogSoftmax + NllLoss 组合；CUDA,SYCL,VULKAN: 部件 NllLoss 设备端 require_cpu 拒绝，传递性不支持 |
| `Cumsum` | native | native | native | native |  |
| `Div` | native | native | native | native |  |
| `Equal` | native | native | native | native |  |
| `Erf` | native | unsupported | unsupported | unsupported | require_cpu 显式拒绝 |
| `Exp` | native | native | native | native |  |
| `Eye` | native | emulated | emulated | emulated | 宿主生成后 to_device 搬运 |
| `Fill` | native | native | native | native |  |
| `Flip` | emulated | emulated | emulated | emulated | 逐元素 narrow 逆序视图 + Concat |
| `Gather` | native | native | native | native |  |
| `Greater` | native | native | native | native |  |
| `GreaterEqual` | native | native | native | native |  |
| `IndexSelect` | native | native | native | native |  |
| `LayerNorm` | native | emulated | emulated | emulated | Sub/Mean/Mul/Div/Sqrt/Add 组合（layer_norm_composed） |
| `Less` | native | native | native | native |  |
| `LessEqual` | native | native | native | native |  |
| `Linear` | emulated | emulated | emulated | emulated | MatMul(input, weight^T)，bias 形式再加 Add |
| `Linspace` | native | emulated | emulated | emulated | 宿主生成后 to_device 搬运 |
| `Log` | native | native | native | native |  |
| `LogSoftmax` | native | emulated | emulated | emulated | Max/Exp/Sum/Sub/Log 组合（log_softmax_composed） |
| `MaskedFill` | native | unsupported | unsupported | unsupported | CPU: 宿主侧按 mask 逐元素写入；CUDA,SYCL,VULKAN: require_cpu 显式拒绝 |
| `MaskedSelect` | native | unsupported | unsupported | unsupported | CPU: 宿主侧按 mask 逐元素挑选后拷出；CUDA,SYCL,VULKAN: require_cpu 显式拒绝 |
| `MatMul` | native | native | native | native |  |
| `Max` | native | native | native | native |  |
| `Mean` | native | native | native | native |  |
| `Min` | native | native | native | native |  |
| `MseLoss` | native | unsupported | unsupported | unsupported | require_cpu 显式拒绝 |
| `Mul` | native | native | native | native |  |
| `NllLoss` | native | unsupported | unsupported | unsupported | require_cpu 显式拒绝 |
| `Nonzero` | native | native | native | native |  |
| `Norm` | native | emulated | emulated | emulated | Abs+Sum 或 Mul+Sum+Sqrt 组合（norm_composed） |
| `NotEqual` | native | native | native | native |  |
| `Ones` | native | native | native | native |  |
| `Pad` | emulated | emulated | emulated | emulated | narrow 裁剪 + 常量块 Concat |
| `Pow` | native | native | native | native |  |
| `Prod` | native | unsupported | unsupported | unsupported | require_cpu 显式拒绝 |
| `Randn` | native | emulated | emulated | emulated | 宿主生成后 to_device 搬运 |
| `Random` | native | native | native | native |  |
| `Relu` | native | native | native | native |  |
| `Repeat` | emulated | emulated | emulated | emulated | reshape + expand 纯视图 |
| `RmsNorm` | native | emulated | emulated | emulated | Mean/Mul/Div/Sqrt 组合（rms_norm_composed） |
| `Roll` | emulated | emulated | emulated | emulated | 两段 narrow 视图 + Concat |
| `Sigmoid` | native | native | native | native |  |
| `Silu` | native | native | native | native |  |
| `Sin` | native | native | native | native |  |
| `Slice` | native | native | native | native |  |
| `Softmax` | native | native | native | native |  |
| `Sort` | native | unsupported | unsupported | unsupported | execute_on 无设备特化，抛 not implemented |
| `Split` | emulated | emulated | emulated | emulated | narrow 视图切片，纯视图无内核 |
| `Sqrt` | native | native | native | native |  |
| `Stack` | emulated | emulated | emulated | emulated | unsqueeze 视图 + Concat |
| `Std` | native | emulated | emulated | emulated | Sqrt(var_composed) |
| `Sub` | native | native | native | native |  |
| `Sum` | native | native | native | native |  |
| `Tan` | native | native | native | native |  |
| `Tanh` | native | native | native | native |  |
| `Topk` | native | unsupported | unsupported | unsupported | execute_on 无设备特化，抛 not implemented |
| `Transpose` | native | native | native | native |  |
| `Typecast` | native | native | native | native |  |
| `Var` | native | emulated | emulated | emulated | Mean/Sub/Mul 组合（var_composed） |
| `Where` | native | native | native | native |  |
| `Zeros` | native | native | native | native |  |

## 统计

| 后端 | native | emulated | unsupported | 支持率 |
|------|--------|----------|-------------|--------|
| CPU | 65 | 12 | 0 | 100.0% |
| CUDA | 47 | 21 | 9 | 88.3% |
| SYCL | 47 | 21 | 9 | 88.3% |
| VULKAN | 47 | 21 | 9 | 88.3% |

支持率 = (native + emulated) / 算子总数（77）。

## 宿主侧 API（不经过后端派发，不在矩阵中）

`DeviceGuard`、`SortResult`、`TopkResult`、`current_device`、`device_count`、`empty_cache`、`export_csv`、`println`、`set_device`、`synchronize`。
