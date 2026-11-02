# Specification: Tensorvia 推理张量契约与执行模型

## Metadata
- **Version**: 1.0.0
- **Status**: Realized
- **Author**: Claude Code
- **Created**: 2026-10-30
- **Last Updated**: 2026-11-01

## Overview

**现状**：Tensorvia 是 C++23 跨后端张量库，已具备 Tensor 句柄 + 引用语义 + 零拷贝视图（`t/permute/slice/view`）、`DeviceId{kind,index}` 与线程私有 current device、惰性后端初始化与按实例注册 factory、编译期后端派发（未实现算子显式报错、不静默回退）、CPU/Vulkan/CUDA/SYCL 四个后端、CUDA caching allocator 与每线程 current stream、Vulkan VMA、CPU per-thread pool，以及 gtest/bench/examples 与 mcpp/CMake 两条构建路径。

**目标**：定义下一阶段的公开契约与执行模型，让库在**推理/数值计算**场景下对齐 PyTorch 的设计哲学。范围是 P0 语义契约、P1 推理算子覆盖、P2 异步执行与数据通路、P3 统一 strided 迭代层。BOOL 采用方案 C：独立 dtype、1-byte 存储、不做 bit-packing。本文是契约级 spec，不规定具体类名或函数签名；实现细节由后续 plan 决定。

**非目标**：不做 Autograd、`nn.Module`、optimizer、complex dtype、量化、完整 PyTorch Dispatcher、单次构建编入多个设备后端；权重加载/序列化/互操作（safetensors/GGUF/ONNX/DLPack/Python）另立 spec，不在本范围。

### 实施状态

| FR              | 状态                       | 说明                                                                                                           |
|-----------------|----------------------------|----------------------------------------------------------------------------------------------------------------|
| FR-1 形状契约   | 已实现（四 lane 已验证）   | 0-dim、空张量（`numel==0`）与逐算子语义已落地，见“FR-1 落地说明”；SYCL lane 仅剩两条既有环境失败               |
| FR-2 dtype 契约 | 已实现（四 lane 已验证）   | 见下文“FR-2 落地说明”；CPU / CUDA / Vulkan lane 全绿，SYCL lane 见下方 lane 覆盖说明                           |
| FR-3 归约契约   | 已实现（四 lane 已验证）   | 全/单轴/多轴归约、0-dim 结果、keepdim、负轴、`dtype` 限定、`all/any` 0-dim BOOL 已落地；`out=` 未做，见“FR-3 落地说明” |
| FR-4 视图契约   | 已实现（四 lane 已验证）   | `narrow/select/expand/squeeze/unsqueeze/reshape` 均为 const 视图，offset 与 strides 正确；见“FR-4 落地说明” |
| FR-5 索引契约   | 已实现（四 lane 已验证）   | `set<T>`/`Nonzero` 索引张量/`MaskedSelect`/`MaskedFill`；masked 两算子暂为 CPU-only，见“FR-5 落地说明” |
| FR-6 可变性契约 | 已实现（四 lane 已验证）   | 统一的 `_` 后缀原地族 + `copy_`，写穿共享存储，见“FR-6 落地说明” |
| FR-7 工厂契约   | 已实现（四 lane 已验证）   | `TensorOptions`、`DeviceId` 工厂重载、可配置默认 device/dtype，见“FR-7 落地说明” |
| FR-8 错误契约   | 已实现（四 lane 已验证）   | 错误信息含算子名与期望/实际；未实现的后端组合显式抛错，见“FR-8 落地说明” |
| FR-9 矩阵乘     | 已实现（四 lane 已验证）   | N-D batched matmul + batch 广播、`bmm/addmm/linear`，核心层折叠为 `[batch,rows,cols]`，见“FR-9 落地说明” |
| FR-10 推理算子   | 已实现（四 lane 已验证）   | 归一化/softmax 族在设备上用组合实现；`Erf/Cumsum/Where/IndexSelect/Gather` 设备端显式报错，见“FR-10 落地说明” |
| FR-11 搬运算子   | 已实现（四 lane 已验证）   | `split/chunk/stack/pad/repeat/flip/roll` 由视图 + `Concat`/`Fill` 拼成；`sort/topk`、`masked_fill/select` 仍 CPU-only，见“FR-11 落地说明” |
| FR-12 后端一致性 | 已实现（四 lane 已验证）   | 跨后端一致性用例集 + 视图 offset/整块拷贝两条规则固化，见“FR-12 落地说明” |
| FR-13..FR-17    | 已实现（四 lane 已验证）   | 简化异步模型，见“P2 范围决策”；CUDA/SYCL/Vulkan 均按要求入队返回，见“FR-13..FR-17 落地说明”                    |
| FR-18..FR-20    | 已实现（四 lane 已验证）   | `ops::iter` 统一 strided/broadcast 迭代层 + 物化策略 + 跨步归约，见“FR-18..FR-20 落地说明”                     |

**lane 覆盖**（2026-10-31）：CPU lane 全绿；CUDA lane（RTX 4070 Ti SUPER，`cuda13.3+{sm_89}`）与 Vulkan lane（NVIDIA 驱动 Vulkan 1.4，`vulkan1.4`）的 `DeviceOps.*` 在真实设备上跑通，含设备侧全归约 + `item<T>()`、keepdim、负轴；`MultiDevice.*` 因只有一张卡按设计 SKIP。SYCL lane 能编译并跑完 CPU 路径，`DeviceOps.*` 实际在 SYCL 设备上跑通（含本切片的 `DeviceOps.EmptyTensors`）；仅 `DeviceId.SynchronizeCoversAllIndices` 与 `MultiDevice.CrossOperandRejected` 因本机 SYCL 运行时分配失败（`sycl Ones: device failure`）而失败，已用本改动之前的 HEAD 复现，属既有环境问题、与本契约无关。 P0 收尾（同日）复跑：CPU 70 passed、CUDA 74 passed、Vulkan 70 passed、SYCL 72 passed + 同上 2 条既有环境失败；新增用例集中在 `DeviceOps.MultiAxisReduction/ReductionDtype/AllAnyTruthiness/NonzeroOnDevice` 与 `View.*`、`Indexing.*`、`Masked.*`、`Tensor.InplaceFamily`、`Options.*`。 P1 收尾（同日）复跑：CPU 90 passed、CUDA 94 passed、Vulkan 90 passed、SYCL 92 passed + 同上 2 条既有环境失败；`MultiDevice.*` 在单卡机器上按设计 SKIP。P1 的 21 个用例集中在 `tests/p1.cpp`（`SortAndTopkOnCpu` 与 `UnsupportedOpsFailExplicitlyOnDevices` 在设备 lane 上按设计 SKIP 掉 CPU 部分断言，同时用 `EXPECT_THROW` 覆盖显式报错）。P3 收尾（2026-11-01）复跑：CPU / CUDA / Vulkan lane 全绿，SYCL lane 仅剩上述 2 条既有环境失败；新增用例集中在 `tests/p3.cpp`（10 个），其中 `WhereOnViewsMatchesContiguous` 因 `where` 只有 CPU 实现而在设备 lane 上按设计 SKIP。

**FR-2 落地说明**（2026-10-31）：

- 提升规则集中在 `compute_type` / `scalar_promote_type`：BOOL 是最弱整型；两个半精度相遇为 FLOAT32；整型遇浮点取浮点；弱标量下浮点张量保持自身 dtype、整型/BOOL 张量取线程私有的 `default_dtype()`。
- `default_dtype()`/`set_default_dtype()` 供线程私有的默认浮点 dtype，工厂与弱标量提升共用；只接受浮点 dtype。
- 归约 dtype：`Sum/Prod` 走 `reduction_type`（整型与 BOOL → INT64，f16/bf16 → FLOAT32，f64 → FLOAT64），Vulkan 的 int8/int16 sum shader 输出已改为 `int64_t`；`Mean` 走 `float_reduction_type`（整型 → FLOAT32，仅 FLOAT64 保持）；`Min/Max` 保持输入 dtype；`argmax/argmin` 恒为 INT64。
- BOOL：比较/谓词输出 BOOL；数值算术在核心层统一拒绝；`Sum` 先 cast 到 INT64、`Min/Max` 经 INT8 往返、`All/Any/Nonzero/Argmax/Argmin` 经 INT8；`Mean/Prod/Var/Std/Norm/Cumsum` 拒绝。Vulkan 无 BOOL shader，在 `submitCompute` 统一映射到同为 1 byte 的 INT8，入口处先把 BOOL 操作数提升为 INT8。
- 比较结果 dtype 归一：部分后端内核（如 CUDA）仍把比较结果写成 INT8，核心层用 `as_bool` 在校验后统一 cast 回 BOOL，四 lane 的 `(a > b).dtype()` 因此一致为 BOOL。
- cast 规范化：数值 → BOOL 非零写 1；BOOL → 数值按 0/1 输出，绝不泄出原始字节；Fill/工厂写入 BOOL 时同样规范化。外部写入的非 0/1 字节在读取时按非零为真。
- 测试：`DtypeContract.*` 六个用例（9×9 提升矩阵、张量算术提升、BOOL 算术拒绝、BOOL cast/规范化、默认 dtype、归约 dtype），CPU lane 全绿。

**FR-3 落地说明**（2026-10-31）：

- `Sum/Min/Max/Mean/Prod/Var/Std/Norm` 一律返回 `Tensor`，与 PyTorch C++ 一致：全归约等价于“归约掉所有轴”，结果是 **0-dim（rank-0）张量**，形状规则与单轴归约拆掉该轴完全一致；值留在原设备上，不隐式同步。
- 取回宿主标量只能显式调用 `Tensor::item<T>()`：仅接受 `numel()==1`，按张量的 dtype 读出再转换为 `T`；设备张量上这是一次同步拷贝（`copy_to_host` 在所有后端都等待其流/队列），即宿主/设备边界的唯一显式入口。offset 视图或非连续视图会先在本地句柄上物化，避免读到 buffer 基址。
- 全归约的实现不改任何后端 kernel：核心层把输入零拷贝展平成 1-D 句柄，复用已有的 axis-0 归约；`keepdim` 是分配后的元数据 reshape（`with_reduced_shape`），后端 `new_shape` 为空时的 `{1}` 兜底因此不会再泄漏到公开形状。
- 核心层统一做轴规范化：负轴从末尾计数，越界轴（含 0-dim 张量上的任何轴）抛出带算子名的错误。
- 归约 dtype 沿用 FR-2：全归约与单轴归约走同一规则；CPU 的 `Mean/Var/Std/Norm` 轴核已改为按 `float_reduction_type` 输出（FLOAT64 输入保持双精度累加）。
- rank-0 成为一等形状：`Tensor` 构造、`Zeros/Ones/Fill/Random/Randn`、`at<T>({})`、广播（0-dim 参与时为标量语义）都按 `numel==1` 处理；`t()` 明确只接受 2-D 输入。
- 测试：`ReduceContract.*` 四个用例（全归约 0-dim 与广播、keepdim/负轴/越界轴、`item<T>()` 守卫与视图取值、rank-0 构造）覆盖核心层；设备侧在 `DeviceOps.ReductionAndSoftmax` 里断言全归约 0-dim、结果留在原设备、`item<T>()`、keepdim 与负轴，CPU / CUDA / Vulkan lane 全绿；原有 `DeviceOps.*`、`Reduce.*`、`Numeric.*`、`DtypeContract.*` 期望同步更新。
- 多轴归约（`std::initializer_list<int64_t>` / `std::vector<int64_t>` 重载）：轴从大到小逐轴折叠，每一步都 `keepdim` 保持其余轴的索引有效，最后 `reshape` 到目标形状；空轴列表等价于全归约（`keepdim=true` 时把每个轴压成长度 1）。`Var/Std/Norm` 不能用这种折叠实现（二阶矩不可折叠、范数不按元素折叠），多轴形式显式报错并提示逐轴调用。
- `dtype=` 限定：先按算子自身的 dtype 规则归约，再把结果 cast 到指定 dtype；`BOOL` 作为归约输出 dtype 被显式拒绝（`Sum/Prod/Mean/Var/Std/Norm` 上无意义）。
- `All/Any` 返回 0-dim `BOOL` 张量：真值判定为“非零即真”（`NotEqual(x, 0)` → `INT8` → `Min/Max` 折叠 → cast 回 `BOOL`），全程在输入设备上，`item<bool>()` 仍是唯一同步点；空张量 `All=true`、`Any=false`。
- `Nonzero` 返回索引张量 `(count, dims)` INT64（1-D 输入为 `(count)`，空输入为 `(0, dims)`）；旧计数语义改名为 `CountNonzero`。索引由 CPU 侧循环生成，设备张量先回拷一次。
- 未做（本切片外）：`out=` 输出参数、`MseLoss/NllLoss/CrossEntropyLoss` 返回张量、后端 `{1}` 兜底与已无调用的“整体归约返回宿主标量”后端实现（`XxxImpl::execute(const Tensor&)`）的清理。

**FR-4 落地说明**（2026-10-31）：

- 视图族统一为 `const` 成员并返回新张量：`view/reshape/squeeze/unsqueeze/narrow/select/expand/expand_as`。旧的 `void reshape(...)`、`Tensor& squeeze(...)/unsqueeze(...)` 原地改形状的语义已移除。
- `view` 现在保留视图的 storage offset（此前会把它清零），因此 `slice → view` 组合读到的仍是切片首元素，而不是 buffer 基址。
- `reshape` 的分支是 PyTorch 的规则：连续（含带 offset 的连续视图）返回零拷贝视图，非连续先 `contiguous()` 再返回视图——即“能零拷贝就零拷贝，否则复制”，不报错。
- `narrow(dim,start,length)` 与 `select(dim,index)` 支持负 `dim`/负 `start`/负 `index`，越界抛带算子名的错误；`expand` 只允许长度 1 的轴扩展或新增前导轴（stride 0），其余尺寸不匹配报错。
- 写穿：`set<T>(indices, value)` 与原地算子通过同一份 offset+strides 定位，视图上的写会落到共享存储。
- 后端 `Transpose(a, dst, axes)` 的 `out=` 路径改为 `dst = dst.reshape(...)` 承接返回值，语义不变（要求 dst 连续）。

**FR-5 落地说明**（2026-10-31）：

- 元素读写：`at<T>()` 读（已有）、新增 `set<T>()` 写；二者都按 `data()` 已含 offset + strides 定位，越界抛 `index out of range`；`set` 仅支持 CPU（与 `at` 一致），`BOOL` 写入规范化为 0/1。
- 切片赋值：`Slice/select/narrow` 返回视图，视图上调用原地族（如 `fill_`、`copy_`、`MaskedFill`）即等价于切片赋值；没有单独的 slice-assign API。
- `Nonzero` 返回索引张量（见 FR-3 落地说明），`CountNonzero` 保留计数。
- `MaskedSelect(input, mask)` → 1-D 张量，按行主序收集选中元素；`MaskedFill(input, mask, value)` → 与输入同形状/同 dtype 的新张量。mask 可为 `BOOL` 或任意有符号整型（非零为真，兼容 INT8 旧谓词）；mask 必须广播到输入形状，广播会**扩大**结果时报错。两者当前只在 CPU 后端实现，设备张量调用会得到 FR-8 的显式错误。

**FR-6 落地说明**（2026-10-31）：

- out-of-place 仍是默认（`Add/Sub/Mul/Div/Abs/Relu/Clamp/...` 全部返回新张量）；原地操作统一用 PyTorch 的尾随下划线：`fill_/zero_/copy_/add_/sub_/mul_/div_/abs_/relu_/clamp_`，全部返回 `Tensor&` 以便链式调用。
- dtype/device 转换：`to_type()` 返回新张量、`to_type_()` 原地；`to_device()` 是句柄的搬迁（保持既有语义），不产生新 dtype。
- 别名与写穿：拷贝 `Tensor` 是 O(1) 句柄共享，任何原地写立刻对同一存储的其它句柄可见；`clone()` 是唯一的显式深拷贝。非连续视图上的原地写被拒绝（`require_contiguous`），避免写到视图之外；Vulkan 上带 storage offset 的视图同样被拒绝（内核只拿得到整个 buffer，写入会落到别人的元素上），详见 FR-12。
- 整型张量上的浮点标量原地算术被拒绝（会产生未定义的截断），需先 `to_type_()`。

**FR-7 落地说明**（2026-10-31）：

- `TensorOptions{dtype, device, device_index}`：`TensorOptions{}` 从 `default_dtype()`/`default_device()` 起步；设备可只带 `Device`（索引交由线程 current device 解析）或直接带 `DeviceId`；`with_dtype/with_device` 支持链式改写。
- 工厂两套重载并存：既有的 `(shape, DataType, Device)` 形式保持不变，新增 `(shape, TensorOptions)` 与 `(shape, DataType, DeviceId)`；`Tensor` 构造、`Zeros/Ones/Fill/Random` 及 `ops::*` 工厂都覆盖。
- 默认值：`default_dtype()`/`set_default_dtype()`（仅浮点）、`default_device()`/`set_default_device()`（CPU 只接受 0 号；设备索引越界或负值立即报错）。
- 明确不做 layout：推理场景只支持行主序 strided，`TensorOptions` 不预留 layout 字段。

**FR-8 落地说明**（2026-10-31）：

- 契约违例的错误信息包含算子名与关键上下文：轴越界报 `"sum: axis is out of range for a 3D tensor"`，dtype 违例报 `"mean: BOOL is not supported; cast the tensor to a numeric dtype first"`，设备违例报 `"prod: only the CPU backend implements this op (tensor is on CUDA)"`。
- 跨设备操作数、设备类型不匹配、mask 广播过大、重复/越界轴、`BOOL` 算术与 `BOOL` 归约输出都有独立错误路径。
- 未实现的后端组合由 `execute_on` 的编译期分派统一抛出 `"this operation is not implemented on the <BACKEND> backend"`，没有静默回退到 CPU。


**FR-1 落地说明**（2026-10-31）：

- 空张量是一等形状：`numel==0` 合法，维度长度只需 ≥ 0（负维度仍抛错，`calc_numel` 的三份重载统一按 `dim < 0` 判断）。`data()` 永远不为空——CPU 池、VMA、CUDA caching allocator、SYCL seam 都拿到 1 字节占位分配，因此 clone/拷贝/析构路径不需要额外分支。
- 行主序步长按定义计算：空维会把外侧步长压成 0（`{2,0,3}` → strides `{0,3,1}`），`is_contiguous()` 仍为真。
- **核心层短路，不跑 kernel**：凡是输出必然为空的算子都在 `src/core/ops.cpp` 直接用同设备构造空结果（`empty_on`），不进入后端 launch。这样既避开 CUDA 零 grid、SYCL 零 range、Vulkan 零长度 buffer 的后端差异，也绕开 CPU ufunc 的 `broadcast_drive` 在 `inner==0` 时除零。需要写出恒等值/NaN 时用 `filled_on`（复用四个后端都已实现的 Fill kernel），结果仍留在输入设备上，不隐式搬回宿主。
- 逐类语义：
  - 逐元素（一元/标量/二元/比较/where/typecast）：空入空出，dtype 与形状规则不变；广播时 0 与 1 得 0，0 与非 1 维按 `broadcast_shapes` 报错；in-place 版本对空输入直接返回，但参数校验（如 `Clamp` 的 `min > max`、整型输入拒绝）仍在短路之前。
  - 结构算子（slice/concat/transpose/permute/t/view/reshape）：空入空出，形状规则不变；`Concat` 的空输入不贡献元素。
  - 全归约：`Sum=0`、`Prod=1`（`reduction_type`），`Norm=0`、`Mean/Var/Std=NaN`（`float_reduction_type`，0/0 的确定取值），`Min/Max` 抛错（无恒等元）。
  - 单轴归约：归约轴长度为 0 → 用恒等值/NaN 填充结果；非归约轴为 0 → 空结果；`Min/Max/Argmax/Argmin` 在空归约轴上抛错，`Argmax/Argmin` 在空输出上返回空 INT64（形状照常）。
  - `Cumsum` 空入空出；`All(empty)=true`、`Any(empty)=false`、`Nonzero(empty)=0`。
  - `MatMul`：输出本来没有元素时为空；内积维 K==0 时输出全 0（与 PyTorch 一致），2-D 与批处理 3-D 均如此。
  - `Softmax/LogSoftmax/LayerNorm/RmsNorm`：空入空出，整型输入仍按规则提升到 FLOAT32；损失函数保留原有的显式报错。
- 测试：`Empty.*` 四个用例覆盖核心层（构造/元数据/拷贝/打印、逐元素与广播含 `0 vs n>1` 报错、全归约与单轴归约的恒等值/NaN/抛错、结构算子/MatMul/concat/索引/softmax/归一化）；`DeviceOps.EmptyTensors` 在设备 lane 上断言分配、归约恒等值、NaN、空 BOOL 比较、K==0 matmul 与空输入 concat 都在原设备上完成。2026-10-31 复跑：CPU 55 passed、CUDA 59 passed、Vulkan 59 passed、SYCL 57 passed + 2 条既有环境失败（`DeviceId.SynchronizeCoversAllIndices`、`MultiDevice.CrossOperandRejected`）。

**FR-9 落地说明**（2026-10-31）：

- N-D 折叠在核心层完成，不给四个后端各写一套 N-D 内核：`matmul_dims` 从尾部对齐推 batch 广播形状，并按 PyTorch 处理 1-D 操作数（左操作数补前导 1、右操作数补尾随 1，结果去掉被提升的维——`vec @ vec` 是 0-dim，`vec @ matrix` 是 1-D）；`matmul_operand` 把操作数右对齐 `expand` 到 `[batch, rows, cols]` 再 `contiguous`，交给后端已有的批处理内核（`MatMulImpl` 签名本就带 `batch`）。
- `Bmm` 是 3-D 校验包装；`Addmm(input, mat1, mat2, beta, alpha) = beta*input + alpha*(mat1@mat2)`，`beta == 0` 时完全不读 `input`（与 PyTorch 一致）；`Linear(x, W[, b]) = x @ W^T (+ b)`，对任意前导 batch 维成立。
- 边界：输出本无元素 → 空；收缩维 `K == 0` → 全 0；0-dim 操作数、内积维不匹配、batch 维不可广播、跨设备操作数都是带算子名的错误。
- 现有 2-D/3-D 调用的语义与测试期望不变。
- 测试：`DeviceOps.MatMul2DMatchesReference / MatMulPromotesVectorOperands / MatMulBatchesAndBroadcasts / MatMulAcceptsNonContiguousOperand / MatMulEmptyAndZeroInner / MatMulRejectsBadShapes / BmmRequiresMatching3D / AddmmScalesAndAccumulates / LinearMatchesBatchedAffine`。

**FR-10 落地说明**（2026-10-31）：

- 在设备上可用、且**不新增设备内核**的一族（核心层用已有设备算子组合，全程留在输入设备、无隐式回拷）：`LogSoftmax`、`LayerNorm`、`RmsNorm`、`Var`、`Std`、`Norm`。CPU 仍走专用内核，两条路径在测试容差内一致。
  - `log_softmax = x − max(x) − log(Σ exp(x − max(x)))`（`Log` 以 e 为底）；`var = mean((x − mean)²)`；`std = sqrt(var)`；`norm(p=1) = Σ|x|`、`norm(p=2) = sqrt(Σx²)`（`p` 只接受 1/2，FR-3 既有约束）。
  - `layer_norm`/`rms_norm` 在 `[outer, norm_size]` 视图上组合，weight/bias 广播到尾部块；`eps <= 0`、weight/bias 的 numel 或 dtype 不匹配在组合之前报错。
- 生成器：`Arange/Linspace/Eye/Randn` 新增 `TensorOptions` 重载，值在宿主按目标 dtype 生成后一次性 `to_device` 到目标设备；RNG 只有宿主一份实现，因此同一 seed 在各后端取值相同（`Randn` 的 `std < 0` 与 BOOL dtype 报错）。
- 明确不支持、走 FR-8 显式错误路径（不静默回退 CPU）：`Erf`、`Cumsum`、`Where`、`IndexSelect`、`Gather` 只有 CPU 内核，设备张量调用抛 `"<op>: only the CPU backend implements this op (tensor is on <BACKEND>)"`；`MaskedSelect/MaskedFill` 同理（见 FR-5）。
- 测试：`DeviceOps.NormAndSoftmaxFamilyMatchesCpu`（log_softmax/var/std/norm 单轴与全归约/keepdim/layer_norm/rms_norm 与 CPU 逐值对比）、`DeviceOps.GeneratorsLandOnTheDevice`、`DeviceOps.UnsupportedOpsFailExplicitlyOnDevices`（断言错误信息点名目标后端）。

**FR-11 落地说明**（2026-10-31）：

- 纯视图、无内核：`Split`（`narrow` 切片，返回共享 storage 的视图；首块覆盖 buffer 的头部视图是"连续但只覆盖部分 buffer"的情形，见 FR-12）、`Chunk`（块长 `ceil(size/chunks)`，块数可能少于 `chunks`；空轴返回 `chunks` 个空视图）、`Stack`（`unsqueeze` + `Concat`）。
- 由已有原语拼成，因此在四个后端都可用：`Pad`（按 `F.pad` 的从尾轴向前布局，负数先裁再拼常量块，`Fill` + `Concat`）、`Repeat`（NumPy tile：`reshape` 成 `(1,size)` 对 → `expand` → `reshape`）、`Flip`（逆序 `narrow` + 一次 `Concat`，尺寸 ≤1 的轴跳过）、`Roll`（按 dim 切两段再 `Concat`，`shift` 先对长度取模）。
- 仍为 CPU-only：`Sort`/`Topk`（返回 `{values, indices}`，`indices` 为 INT64；`Topk` 的 `k` 越过该维长度报错）、`MaskedSelect`/`MaskedFill`；设备调用走 FR-8 显式错误。
- 测试：`DeviceOps.SplitReturnsViews / ChunkUsesCeilBlockSize / StackConcatenatesAlongNewAxis / PadAddsConstantBorder / RepeatTilesLikeNumpy / FlipAndRollReorderData / SortAndTopkOnCpu`。

**FR-12 落地说明**（2026-10-31）：

- 一致性用例集：同一输入的 CPU 参考结果与设备结果逐值对比（shape、dtype、device、值四项），覆盖 FR-9/FR-10/FR-11 的算子；不支持的组合另有用例断言抛错且错误信息点名后端。新用例集中在 `tests/p1.cpp`，在 CPU/CUDA/Vulkan/SYCL 四个 lane 上执行。
- 视图寻址规则固化（本轮修掉的静默错误根源）：内核有两种寻址方式——CPU/CUDA/SYCL 用 `Tensor::data()`（基址 + offset），Vulkan 绑定 impl 的整个 buffer（描述符没有 offset）。因此：
  - CUDA unary/scalar 原地/typecast/matmul 的指针从 impl 基址改为 `Tensor::data()`；CUDA/SYCL 的二元与比较路径本就用 `Bcast{off_a, off_b, off_o}`，未改。
  - Vulkan 上带 offset 的连续视图在 `linearize`/`Concat`/`matmul_operand` 处先 `clone` 成独立 buffer，再进内核。
  - 写入型算子（原地族、`fill_/zero_/typecast/out=` 目的端）要求目的端"连续且（Vulkan 上）无 offset"，否则 `require_contiguous` 报错——拒绝而不是写坏同一 storage 的其它元素。
- 整块拷贝规则：`to_host/clone/copy_/item<T>()` 的后端传输只对"`offset == 0` 且 numel 覆盖整个 impl buffer"的张量走整块拷贝；`split/narrow` 得到的头部视图虽连续且无 offset，但只覆盖 buffer 的一部分，必须物化成自己的 buffer（`materialize_for_transfer`），否则会溢出目的分配（本轮修掉的 CUDA lane SIGSEGV）。`copy_` 的目的端同样必须覆盖整个 buffer，否则报错提示先 `clone()`。
- 覆盖矩阵（✅ 原生/组合可用，✗ 显式报错）：`matmul/bmm/addmm/linear`、`split/chunk/stack/pad/repeat/flip/roll`、`log_softmax/layer_norm/rms_norm/var/std/norm`、`arange/linspace/eye/randn` 在 CPU/CUDA/Vulkan/SYCL 上均为 ✅（设备端多为组合实现）；`erf/cumsum/where/index_select/gather/sort/topk/masked_select/masked_fill/prod` 为 CPU ✅ / 设备 ✗。

**P2 范围决策**（2026-10-31）：异步模型按**简化版**收敛，不做公开的 Stream/Event 抽象。

- **流**：对外语义是“每条（线程, 设备实例）一条隐式流”——同一线程内的算子按提交顺序执行；不同线程的流之间**没有顺序保证**，可能并发，也可能不并发。流由 context 内部管理，首次使用时惰性创建；不对外暴露句柄，也不保留 `StreamGuard`、`set_current_stream()` 与 stream 池——P1 留下的这三者只服务于"临时换流"，简化模型里不存在这个动作，随 P2 删除。
- **各后端如何实现该语义**：CUDA 用线程局部映射到 `cudaStream_t`；SYCL 用线程局部映射到 `sycl::queue`（所有队列共享同一个 `sycl::context`，否则 USM 分配不能跨队列使用）；Vulkan 从所选队列族预建队列池（数量受 `queueCount` 限制），按线程分配。队列族只暴露 1 条计算队列的硬件上，Vulkan 退化为共享队列 + 提交互斥——这只降低并发度，不改变顺序保证与同步语义。
- **线程亲和**：张量归属创建它的线程；跨线程移交必须先在原线程 `synchronize()`，否则行为未定义。库不检测这种误用（不做每张量的线程 id 校验），靠契约与测试约束。这样“跨流依赖”问题被排除在契约之外。
- **分配器**：按流分池，块记住自己的流，只在该流上复用——同流按序执行保证复用安全，热路径不需要 event。
- **多卡**：每（线程, 卡）一条流；跨设备拷贝经宿主中转，天然串行。
- **不做什么**：不提供 `record_stream`、不提供公开 Event、不做跨流事件等待。只有出现“同一张量被多条流并发使用”的真实用例时，才另立 spec 引入这些抽象。
- **同步点**：`item<T>()`、`to_host()`、`ops::synchronize()`。`Tensor::synchronize()` 从未在公开 API 中实现，早期段落对它的引用是笔误；张量级等待只经由前两者发生。

**FR-13..FR-17 落地说明**（2026-10-31）：

- CUDA：线程局部映射 `设备实例 id → cudaStream_t`，首次用时建一条 non-blocking 流并登记到 context，供 `synchronize_all()` 与析构清空；算子入队后直接返回，P1 留在算子尾部的 `wait()` 全部删除（matmul/ufunc/typecast/concat/slice/transpose/activate/reduce/initializer/clone 共二十处）。缓存分配器按流分池：块记住释放它的流，只在该流的空闲表里复用；`empty_cache`/OOM 回收仍会阻塞，因为 `cudaFree` 隐含一次设备同步。
- SYCL：每条（线程, 设备实例）一条 in-order `sycl::queue`，首次使用时惰性创建并登记到 context；所有队列由**同一个 `sycl::context`** 构造——USM 分配归属 context，不共享 context 的队列之间不能互用同一块内存。算子在当前线程的队列上入队；`sycl::free` 交给**分配该块的那条队列**的 `host_task`，排在它的既有工作之后（提交失败时退化为“等待后直接释放”）。`ops::synchronize()` 等待所有队列的 `wait_and_throw()`；张量记住自己的队列句柄，对应 CUDA 的 `m_stream`。
- Vulkan：从所选队列族预建提交通道池（`min(queueCount, 8)` 条），每条通道自带 `VkQueue`、命令池、描述符池与在飞 deque——命令池与描述符池都不是线程安全的，“每通道一份”同时也解决了并发提交。每条（线程, 设备实例）首次提交时按轮转分配一条通道并固定下来，线程数超过通道数时按环复用（提交互斥保证正确，只降低并发度；不记录“已占用”，线程结束也不会把池耗光）。提交自带 fence、入队即返回；在飞提交上限 512，超限才等队首；完成的提交由后续操作顺带回收，命令缓冲与描述符集在回收时才释放。buffer 销毁延迟到**分配它的那条通道**的既有提交完成之后（该通道本就空闲时立即销毁）；`to_host()`/`item<T>()` 只等调用线程的通道，与 CUDA 的 `cudaStreamSynchronize` 一致。`ops::synchronize()` 等待所有通道。
- CPU：`synchronize`/`empty_cache` 保持 no-op。
- 删除项：`StreamGuard`、`set_current_stream()`、`stream_from_pool()`、`default_stream()` 与 `wait()` 均无生产调用者，随本契约移除。
- 测试：`tests/p2.cpp` 覆盖"入队不等待"（按后端实测提交开销自选矩阵规模，见 P2 测试步骤 1）、同步点可见性、反复分配/释放数值正确、跨线程入队后主线程 `synchronize()`；`tests/device.cpp` 覆盖每线程一条稳定流、`empty_cache` 不误伤活张量、同索引多 context 各自独立。

**FR-18..FR-20 落地说明**（2026-11-01）：

- **统一层只有一份**：`ops::iter`（`include/tensorvia/tensor_iterator.hpp`）是广播形状、逐操作数 element stride、视图 offset 与 dtype 提升的唯一实现。`Layout` 描述"广播后的形状 + 每个操作数怎么走 + 输出是否 broadcastless"；后端不再各自推导索引。`PreparedInputs` 是算子的唯一入口，它在构造时就决定了这次调用要不要物化，算子只拿到 `const Tensor&` 和 `Layout`。
- **物化策略集中在实现里的一处**：`should_materialize = !linearizable && trailing_linear_extent < 4 && numel >= 1024`。`linearizable` 判定所有操作数在广播后形状上已是紧凑行主序（可一次线性 kernel 覆盖全程，包括"广播但不跨步"的常见情形）；`trailing_linear_extent` 是"一次 kernel 调用能覆盖的元素数"，作为局部性代价的代理。也就是说：内层段太短（< 4 个元素，跨步读完全没有合并空间）且张量足够大时，复制一次换线性访问；小张量或内层段够长的视图直接跨步算。阈值是策略而不是契约，随 bench 调整。
- **算子侧的统一形态**：`linearizable(lay)` 为真时一次 kernel 打满 `lay.numel`；否则 `drive(lay, …)` 用里程计把遍历切成段——段内线性、段间按 stride 推进，每段交给同一个 kernel。CPU 的 `binary_op`×2、`logical_op`/`logical_op_scalar`、`unary_op`、`Where` 三个重载全部迁移到这个形态；`cpu_ops.hpp` 的 `broadcast_drive` 与 `where.cpp` 的 `where_broadcast` 随之删除（它们是旧的第二套广播实现）。
- **归约不再无条件物化**：CPU 上非连续、且被归约轴 stride == 1 的视图直接跨步归约——按 `ReduceMap` 枚举每个输出元素在输入里的起点，以 `(outer=1, inner=1, axis_size)` 调用既有的 axis kernel。因此累加 dtype、NaN 传播、首个 max 下标的语义与连续路径逐位相同，十一个归约 kernel 一行未改。被归约轴跨大步长（如转置后归约列）仍然物化：跨步读列是缓存最坏的访问模式，复制一次换连续读更划算。
- **设备端**：二进制与比较算子本来就把"逐操作数 stride + offset"作为 `Bcast` 参数交给 kernel，P3 的用例在 CUDA/Vulkan/SYCL 上直接验证了这条既有路径；设备端线性 kernel（如标量比较）保持一次性物化。
- **仍然是"策略性物化"，不是遗漏**：`Cumsum`（输出布局与输入视图不同，跨步写要再教一套索引）、全归约（`flatten_for_reduce`，标量输出且复制代价与归约同阶）、`sort/topk/gather/index_select` 等索引空间算子（本质是重排访问）、设备端 `matmul` 对操作数 `contiguous()`（GEMM 需要紧凑布局）。`Where` 设备侧未实现，公开入口仍是 CPU-only。
- **测试**：`tests/p3.cpp` 十个用例把"算子作用在视图上"的结果与**在 host 上按索引手算**的基线逐值对比，覆盖转置、permute、切片 offset、以及它们与广播的组合；归约用例同时踩跨步路径（`sum/mean/max/argmax/argmin` 沿 stride-1 轴）与物化路径（沿跨步轴）。基线不经过本库任何算子，所以锁的是语义而不是实现。
- **回归**：CPU lane 的 bench 与 P3 之前一致或更好（`add f32 16M` 3.12–3.47 ms / 58–64 GB/s，`mul` 3.01–3.26 ms，`sum-axis1` 0.72–0.77 ms，均落在跨次运行的抖动内）；`matmul f32 512` 三次运行 562–608 GFLOPS（≈±5%），该路径本期未改动，属机器抖动而非回退。NFR-1 的 5% 上限按"连续路径不因统一层变慢"理解并通过。

## Requirements

### Functional Requirements

状态标记：**✅ 已实现**（有自动化测试覆盖并在 lane 上验证）、**🟡 部分实现**（核心语义已落地，条目标注的子项未做）；未标记的条目尚无实现或验证记录。逐项明细见上方“实施状态”与各 FR 的落地说明。

#### P0 — 语义契约（先于算子扩展）

- FR-1 ✅ **形状与 BOOL 存储契约**：支持 0-dim 标量张量、空张量（numel 为 0）；新增独立 `BOOL` dtype，采用 **1-byte 存储**（`uint8_t`，值规范化为 0/1），**不做 bit-packing**；所有算子对 0-dim/空输入有明确定义（例如 `sum(empty)=0`、`mean(empty)` 的取值或报错行为，二选一并写入文档）。
- FR-2 ✅ **dtype 契约**：类型提升遵循 PyTorch 规则（含标量/弱类型语义、整数与浮点混合）；比较运算与谓词类算子（如 `isnan/isfinite`）返回 `BOOL`，`argmax/argmin` 返回 int64；归约结果 dtype 按 PyTorch 规则提升；支持默认 dtype 配置。`BOOL` 的数值算术默认拒绝，仅支持明确列出的逻辑/比较/where/masked/index/归约算子；bool↔数值 cast 明确定义（数值→BOOL 非零为真，BOOL→数值为 0/1）；非 0/1 的外部字节在读取时按非零为真、写入时规范化为 0/1。
- FR-3 ✅ **归约契约**（已落地：全/单轴/多轴归约、0-dim 结果、`keepdim`、负轴、`dtype=` 限定、`item<T>()`、空输入、`all/any` 返回 0-dim BOOL；未做：`out=`）：全归约返回 0-dim 张量而非宿主标量；`dim` 支持负轴、单轴或多轴；支持 `keepdim`；在实现成本可控时支持 `dtype`/`out`。宿主标量一律经 `Tensor::item<T>()` 显式取回，该调用是同步点。BOOL 归约遵循 PyTorch 语义：`sum` 提升为整数，`mean` 明确不支持或定义输出 dtype，`all/any` 返回 0-dim `BOOL`。
- FR-4 ✅ **视图契约**：任意视图组合（`slice → t/permute → narrow/select → expand`）保持正确的 storage offset 与 strides；`reshape` 在无法零拷贝时复制或显式报错；`squeeze/unsqueeze` 只作用于指定轴并返回视图。
- FR-5 ✅ **索引契约**：支持元素访问、切片赋值、BOOL 掩码选择/填充；迁移期内 `where/masked_*` 可兼容 INT8 非零谓词，但规范谓词类型是 BOOL；明确哪些索引返回视图、哪些返回拷贝；`Nonzero` 返回索引张量而非计数。
- FR-6 ✅ **可变性契约**：默认 out-of-place；in-place 显式且命名统一；dtype/device 转换默认返回新张量；别名与写穿规则写入文档。
- FR-7 ✅ **工厂契约**：shape+dtype+device（+layout）以统一的 TensorOptions 表达；工厂函数接受 `DeviceId`；默认 device/dtype 可配置。
- FR-8 ✅ **错误契约**：契约违规抛出包含算子名、期望/实际 dtype/shape/device 的错误；任何后端未实现的组合都必须显式报错，禁止静默回退。

#### P1 — 推理算子覆盖

- FR-9 ✅ **矩阵乘**：支持 N-D batched matmul + 广播；提供 `bmm/addmm/linear`；现有 2D/3D 用法保持兼容。
- FR-10 ✅ **推理关键算子的加速器覆盖**：`LayerNorm`、`RmsNorm`、`LogSoftmax`、`Erf`、`Var/Std/Norm`、`Cumsum`、`Where`、`IndexSelect`、`Gather`、`Arange/Linspace/Eye/Randn` 在已编入的设备后端上可用，或明确声明不支持。
- FR-11 ✅ **形状/数据搬运算子**：`split/chunk/stack/pad/repeat/flip/roll`、`sort/topk`、`masked_fill`；与现有 `Concat/Transpose/Slice` 组成完整的推理图积木。
- FR-12 ✅ **后端一致性**：同一算子在 CPU 与各设备后端上的 dtype/shape 语义一致；不支持的组合走 FR-8 的错误路径；建立跨后端一致性测试集。

#### P2 — 异步执行与数据通路

- FR-13 ✅ **异步执行**：设备算子入队后立即返回，不在每个 kernel 后做宿主同步；同步点仅限文档列出的数据依赖 API（`item<T>()`、`to_host()`、`ops::synchronize()`）。
- FR-14 ✅ **隐式流**：每条（线程, 设备实例）持有一条隐式流，同流内顺序执行；不提供公开的 Stream/Event 抽象，`StreamGuard`/`set_current_stream()`/stream 池随本契约删除；流生命周期与 context 绑定。
- FR-15 ✅ **分配器流安全**：缓存分配器按流分池，块只在创建它的流上复用（同流按序，故无需 event）；不提供 `record_stream`——跨流使用已由 FR-14 的线程亲和规则排除；`empty_cache` 不得释放仍被引用的块。
- FR-16 ✅ **宿主数据通路**：明确同步拷贝与 non_blocking 拷贝的完成语义（non_blocking 拷贝的结果在 `to_host()`/`synchronize()` 之后可见）；pinned host memory 不在 P2 范围，需要时另立 spec。
- FR-17 ✅ **设备控制**：`synchronize/empty_cache` 覆盖所有已启动实例与所有线程流；CPU 实现保持 no-op；多线程压测通过；跨线程移交张量必须显式 `synchronize()`（未同步的跨线程使用是未定义行为，库不做检测）。

#### P3 — 统一 strided 迭代层

- FR-18 ✅ **统一迭代器**（已落地：`ops::iter` 的 `Layout`/`PreparedInputs`/`linearizable`/`drive` 是唯一实现，非连续输入按 `should_materialize` 策略决定是否复制）：CPU 与设备后端共用一套 strided/broadcast 迭代语义；非连续输入不再无条件 materialize；需要物化时由统一策略决定。
- FR-19 ✅ **统一类型提升/广播规则**（已落地：提升规则在核心层 `compute_type`/`scalar_promote_type`，广播在 `iter::build`；后端只吃 typed pointer + 迭代范围）：只保留一份 dtype promotion 与 broadcast 规则实现；后端 kernel 只负责 typed pointer + 迭代范围。
- FR-20 ✅ **迁移与回归**（已落地：元素级/比较/一元/where 全部迁移，归约按轴 stride 选路；`tests/p3.cpp` 十个用例在四 lane 上验证）：现有算子迁移到统一层后行为不变；CPU/设备端一致性测试与 bench 无显著回退。

### Non-functional Requirements

- NFR-1 **性能**：设备端热路径无隐式宿主同步；非连续/广播算子相对连续路径的开销有量化上限；现有 benchmark 回退不超过 5%（以基线提交为准）。
- NFR-2 **兼容性**：除文档列出的破坏性变更（归约返回类型、比较 dtype、`to_device` 语义、0-dim/空张量）外，现有公开 API 保持源码兼容；提供迁移指南。
- NFR-3 **线程安全**：注册表、current device、current stream 的线程私有/加锁语义明确；不同流上的并发算子安全；保留多线程小张量 churn 测试。
- NFR-4 **可测试性**：每条 FR 至少一个自动化测试；覆盖 CPU 与已编入的设备 lane；边界条件纳入测试（0-dim、空、负轴、BOOL 掩码、非 0/1 外部字节、BOOL 算术拒绝、BOOL↔数值 cast、非连续视图、offset 视图、NaN/Inf、OOM）。
- NFR-5 **文档**：README、API 契约、异步语义、破坏性变更与迁移指南同步更新。
- NFR-6 **范围守护**：本 spec 不含 Autograd、`nn.Module`、optimizer、complex dtype、量化、完整 PyTorch Dispatcher、单次构建编入多个设备后端；出现需求先另立 spec。
- NFR-7 **风险与缓解**：
  - R1 语义迁移破坏现有使用方 → 分阶段弃用、迁移指南、在 P0 gate 前完成影响面清单。
  - R2 异步后生命周期/竞态问题 → 张量线程亲和 + 显式 `synchronize()` 移交、分配器按流分池、多线程压力测试、TSan/ASan 回归。
  - R3 后端行为分叉 → 跨后端一致性测试集 + 统一 TensorIterator（P3）。
  - R4 strided 迭代引入性能回退 → bench gate + 保留物化快路径 + 按算子灰度迁移。
  - R5 0-dim/空/bool 改动波及全部 kernel → 先定契约与测试矩阵，再逐算子迁移。
  - R6 范围蔓延 → P0–P3 分阶段 gate，非目标写入 README/issue 模板。

## 破坏性变更与迁移指南

P0 落地后源码级不兼容的点，以及替代写法。全部变更在四 lane 上验证。

| # | 变更 | 旧写法 | 新写法 | 说明 |
|---|------|--------|--------|------|
| 1 | 归约返回 `Tensor`（0-dim），不再返回宿主标量 | `float s = ops::Sum(a);` | `float s = ops::Sum(a).item<float>();` | 值与设备绑定；`item<T>()` 是唯一同步点 |
| 2 | `reshape` 返回值，不再原地改形状 | `a.reshape({2, 3});` | `a = a.reshape({2, 3});` 或 `Tensor v = a.reshape({2, 3});` | 连续张量上是零拷贝视图；非连续先复制 |
| 3 | `squeeze/unsqueeze` 返回值，不再原地 | `a.unsqueeze(0);` | `Tensor v = a.unsqueeze(0);` | 均为视图 |
| 4 | `Nonzero` 返回索引张量 | `size_t n = ops::Nonzero(a);` | `size_t n = ops::CountNonzero(a);` / `Tensor idx = ops::Nonzero(a);` | 索引形状 `(count, dims)` INT64 |
| 5 | `All/Any` 一元形式返回 0-dim BOOL 张量 | — | `ops::All(a).item<bool>()` | 旧的 `All(a, val)`（“全部等于 val”）保留为 `bool` |
| 6 | 比较算子结果 dtype 为 `BOOL` | 以 INT8 读取结果 | `at<bool>()` 或 `Typecast` 到数值 dtype | BOOL 是 1-byte 存储、值规范化为 0/1 |
| 7 | `view` 保留 storage offset | 切片后 `view` 读到 buffer 基址 | 读取切片首元素 | 这是修复，不是行为选择 |
| 8 | `BOOL` 数值算术、`BOOL` 作为归约输出 dtype 报错 | 静默按 0/1 计算 | 先 `Typecast` 到数值 dtype | FR-2/FR-3 |

迁移步骤建议：先替换归约的宿主标量读取（`item<T>()`），再处理 `reshape/squeeze/unsqueeze` 的原地调用点（编译错误会全部暴露），最后按 `Nonzero` 的用途拆分计数与索引。

## Test Steps

### 构建与基础验证

```sh
# mcpp 路径（测试默认走 dev profile，没有 --dev 开关）
mcpp build
mcpp test

# 设备 lane（按可用硬件）
mcpp test --features cuda --accel "cuda13.3+{sm_89}"
# Vulkan：本机 mcpp 自带的 loader 默认搜不到 NVIDIA 驱动，需补上运行时包里的库目录
# （版本号随 mcpp 解析结果变化）。缺这一步时设备枚举为 0，DeviceOps 会整体 SKIP。
LD_LIBRARY_PATH=$HOME/.mcpp/registry/data/xpkgs/compat-x-vulkan-runtime/2026.09.07/mcpp_generated/vulkan_runtime/lib \
    mcpp test --features vulkan --accel vulkan1.4
mcpp test --features sycl --accel sycl

# CMake 路径
cmake -B build -DBACKEND_CPU=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### P0 语义契约

1. 构造 0-dim 标量、空张量、BOOL 张量；验证 numel/shape/dtype/device、`is_contiguous()` 与 `element_size()==1`（确认不是 bit-packed）。
2. 运行 dtype promotion 矩阵（9 种 dtype × 9 种 dtype，含 BOOL 与标量混合）；与 PyTorch 对照表逐项断言。
3. BOOL 语义：比较/谓词算子返回 BOOL；BOOL 数值算术报错；BOOL↔数值 cast 正确；非 0/1 外部字节读取为真、写入规范化为 0/1；`sum/mean/all/any` 行为符合 FR-3。
4. 全归约：验证返回 0-dim 张量、dtype 正确、`keepdim`、负轴、多轴、空输入。
5. 视图链：`slice → t/permute → narrow/select → expand`；验证 shape/strides/storage_offset 与逐元素值；对非连续 `reshape` 验证“复制或报错”契约。
6. 索引：元素读写、切片赋值、BOOL 掩码选择/填充、`Nonzero` 索引张量；验证视图/拷贝语义。
7. 可变性：out-of-place 默认、in-place 命名、`to/type/device` 返回值；验证不修改源张量。
8. 工厂：`DeviceId` 直接建张量、默认 device/dtype 生效、无效组合报错。
9. 错误：未实现后端组合、dtype 不匹配、shape 不匹配、跨设备操作数；断言错误信息包含算子名与关键上下文。

### P1 推理算子

1. N-D matmul 与朴素参考实现对比（2D/3D/4D、batch broadcast、非连续输入）。
2. `bmm/addmm/linear` 与 matmul 组合的结果一致性。
3. 推理关键算子在 CPU 与设备后端的结果一致性（容差按 dtype）。
4. `split/chunk/stack/pad/repeat/flip/roll`、`sort/topk`、`masked_fill` 的边界用例（空维、单元素、负轴、越界索引、BOOL 掩码、INT8 legacy 谓词）。
5. 不支持组合必须报错，不得静默回退。

### P2 异步执行

1. 入队后立即返回：kernel 入队后测量宿主耗时（或查询流完成状态），断言未发生同步。矩阵规模按后端自选——每算子的 GPU 工作若不明显大于一次提交的宿主开销，GPU 会在排队期间跑完，`总耗时 ≈ 入队耗时`，断言失去意义（本机 Vulkan 一次提交约半毫秒，CUDA 十几微秒）。
2. 同步点：`item<T>()`/`to_host()`/`synchronize()` 之后结果可见；同流连续 op 结果正确。
3. 分配器按流复用：同流反复分配/释放同一尺寸，块被复用且数值正确；压力测试下无数据竞争。
4. non_blocking 拷贝：调用本身不阻塞宿主；`synchronize()` 之后目的缓冲区读到新值（不涉及 pinned 内存）。
5. 多线程 × 每线程一条流 churn；`synchronize/empty_cache` 后无泄漏、无悬垂。
6. 线程亲和：各工作线程只入队不读结果，主线程 `synchronize()` 后读取全部正确（未同步的跨线程使用属未定义行为，不设断言）。
7. 失败路径：OOM、无效设备、context 销毁后使用张量；断言明确报错。

### P3 统一迭代器

1. 同一算子在连续/非连续/广播/offset 视图上的结果与连续基线一致。
2. 对比迁移前后各 dtype 的结果与 bench；确认无非预期物化。
3. 跨后端一致性测试集全绿。

已固化为 `tests/p3.cpp` 的十个用例（每个都在 host 上按索引手算基线）：

1. 双操作数转置 `Aᵀ ⊙ Bᵀ`；permute 到 `{4,2,3}` 后与 `{3}` 广播相加；切片 offset 视图与自身/全 1 张量比较（offset 只能计入一次）；offset + 转置 + 广播三者叠加；不可广播形状在发射前报错。
2. 转置视图上的 `Exp`、转置视图之间的 `Equal/Greater`、视图 + 标量分支的 `Where`。
3. 转置视图的单轴归约：`Sum/Mean/Max/Argmax/Argmin` 走 stride-1 跨步路径、`Sum` 沿跨步轴走物化路径；permute 视图上的全归约（标量输出）。
4. 回归：CPU lane 全量用例 + bench 与 P3 之前对比（`add/mul/sum-axis1/exp/where` 均在抖动内），CUDA/Vulkan lane 全绿，SYCL lane 仅剩既有 2 条环境失败。

### 边界与回归

- 0-dim、空张量、单元素、负轴、bool 掩码、offset 视图、非连续视图、NaN/Inf、整数溢出、OOM、后端不可用、单卡/多卡。
- 现有 `tests/` 全量 + benchmark 基线对比。

## Acceptance Criteria

### Phase Gates

- **P0 Gate**：FR-1..FR-8 实现并有自动化测试；BOOL 作为 1-byte dtype 的契约测试通过（`element_size()==1`、0/1 规范化、算术拒绝、cast 规则、`all/any/sum/mean` 语义）；破坏性变更清单与迁移指南完成；所有现有 CPU 测试（含更新后的期望）通过；无静默回退。
- **P1 Gate**：FR-9..FR-12 实现；N-D matmul 与推理关键算子在 CPU 与至少一个设备后端上通过一致性测试；不支持组合显式报错。
- **P2 Gate**：FR-13..FR-17 按简化异步模型实现；设备热路径无隐式同步（以测试/计时证明）；张量线程亲和与跨线程 `synchronize()` 用例通过；分配器同流复用无数据竞争且 sanitizer 通过；Vulkan 提交后不再同步等待；不引入公开 Stream/Event 抽象。
- **P3 Gate** ✅：FR-18..FR-20 完成；非连续/广播算子不再无条件物化（物化由 `iter::should_materialize` 统一决定，跨步归约直接算）；跨后端一致性测试全绿（CPU/CUDA/Vulkan 全绿，SYCL 仅既有 2 条环境失败）；bench 回退不超过 NFR-1 上限。

### Overall

- 所有 FR 有测试，所有已编入 lane 通过；文档（README/API/异步语义/迁移指南）更新。
- 无 Autograd/`nn.Module`/optimizer/complex/量化/完整 Dispatcher 的实现混入本 spec。
- 代码审查确认：错误路径明确、无静默回退、线程/流安全假设有测试支撑。

## Change Log

| Date       | Version | Description                                                      | Author      |
|------------|---------|------------------------------------------------------------------|-------------|
| 2026-10-30 | 0.1.0   | Initial draft                                                    | Claude Code |
| 2026-10-31 | 0.2.0   | BOOL per option C                                                | Claude Code |
| 2026-10-31 | 0.3.0   | FR-2 dtype 契约落地；补 9×9 提升矩阵测试与实施状态               | Claude Code |
| 2026-10-31 | 0.4.0   | FR-3 归约契约（0-dim 结果 + `item<T>()` + keepdim/负轴）落地     | Claude Code |
| 2026-10-31 | 0.4.1   | 补 lane 覆盖：CPU/CUDA/Vulkan 设备侧验证通过，记录 SYCL 环境限制 | Claude Code |
| 2026-10-31 | 0.5.0   | FR-1 空张量契约落地：`numel==0` 全链路支持与逐算子语义           | Claude Code |
| 2026-10-31 | 0.5.1   | Functional Requirements 逐条标注实现状态（✅/🟡）                | Claude Code |
| 2026-10-31 | 0.6.0   | P0 收尾：FR-3 多轴/dtype/`all/any`、FR-4..FR-8 落地，补破坏性变更与迁移指南 | Claude Code |
| 2026-10-31 | 0.7.0   | P1 落地：N-D matmul 族、设备端归一化/softmax 组合实现、搬运算子、跨后端一致性用例集与视图 offset/整块拷贝规则 | Claude Code |
| 2026-10-31 | 0.8.0   | P2 范围收敛为简化异步模型：每（线程, 设备）一条隐式流、不公开 Stream/Event、分配器按流分池；重写 FR-13..FR-17、测试步骤与 P2 Gate | Claude Code |
| 2026-10-31 | 0.9.0   | P2 落地：CUDA/SYCL/Vulkan 算子入队即返回（删除各处 `wait()`）、CUDA 按流分池、SYCL 延迟释放、Vulkan fence + 退役表回收；删除 StreamGuard/流池；补 `tests/p2.cpp` | Claude Code |
| 2026-11-01 | 0.10.0  | 语义对齐：SYCL 改为每（线程, 设备实例）一条 `sycl::queue`（共享同一 `sycl::context`），Vulkan 改为预建提交通道池 + 按线程分配，两者都与 CUDA 的每线程流一致；`ops::synchronize()` 覆盖所有线程的队列/通道，`to_host()`/`item<T>()` 只等当前线程；修正早期段落里不存在的 `Tensor::synchronize()` 同步点 | Claude Code |
| 2026-11-01 | 1.0.0   | P3 落地（spec 收尾，Status → Realized）：`ops::iter` 统一 strided/broadcast 迭代层 + `should_materialize` 物化策略 + `drive` 分段驱动，元素级/比较/一元/where 全部迁移并删除旧的第二套广播实现；CPU 归约在被归约轴 stride==1 时跨步直算（复用既有 axis kernel）；补 `tests/p3.cpp` 十个语义用例，四 lane 验证 | Claude Code |
