# 同步语义：宿主什么时候会等待设备

> 本文由 Task 5 落地时据源码盘点写成；每条同步点都能在代码里指出位置。覆盖矩阵
> （`docs/coverage.md`，由 `tools/coverage_matrix.py` 生成）末尾有各后端异步状态的
> 摘要；自动化的判据是 `tests/async_contract.cpp` 的可观测计数器断言。

## 契约（FR-4.4）

**除下列显式点外，算子调用不得让宿主等待设备。** 判据不是计时，而是计数：每次
宿主真正等待设备，`tensorvia::sync::note_wait`（`include/_sync_counters.h`）记一次，
`tests/async_contract.cpp` 断言提交阶段计数为 0、显式点各计一次。

显式同步点只有四个：

| 显式点 | 入口 | 为什么必须同步 |
|--------|------|----------------|
| `to_host()` | `Tensor::to_host` → `copy_to_host` | 公开契约是"返回时数据在宿主可读"，调用方随即释放自己的 buffer |
| `item<T>()` | `Tensor::item` → `copy_to_host` | 同上：返回的是一个宿主标量 |
| `synchronize()` | `ops::synchronize` → `DeviceControl::synchronize` | 调用方显式要求"到此为止的工作都完成" |
| 跨设备拷贝 | `Tensor::to_device`（非 CPU → 非 CPU 经宿主中转） | 正确性要求：中转 buffer 在返回后即被释放 |

一个后端差异（调用方可见，故明示）：`to_device` 的 H2D 上传在 CUDA / SYCL 上是
**同步**的（拷贝完成后才返回，宿主暂存随即释放），在 Vulkan 上**只入队**（暂存
buffer 的销毁排到提交完成之后）。这是暂存生命周期的差异，不是契约差异——三种
情况下调用方都不需要额外同步就能使用目标张量。

## 各后端的同步点清单

### CUDA

| 位置 | 触发路径 | 机制 |
|------|----------|------|
| `copy_bytes`（runtime.cu） | `copy_to_host` / `copy_from_host` / `copy_to` / `clone` / 带初始数据的构造 | `cudaMemcpyAsync` + `cudaStreamSynchronize`（当前流） |
| `synchronize_current_stream`（runtime.cu） | `ops::synchronize` 的单流路径 | `cudaStreamSynchronize` |
| `synchronize_all`（runtime.cu） | `ops::synchronize` 的设备级路径 | 逐条已建流 `cudaStreamSynchronize` |
| `release_free_segments`（allocator.cuh） | `empty_cache()` 与 OOM 补救 | `cudaFree` 隐含一次设备同步 |

算子入队本身不同步：隐式流 + 按流分池，启动即返回。非连续视图的物化
（`clone_as_contiguous` → `launch_repack`）是一个内核，同样异步。

### SYCL

| 位置 | 触发路径 | 机制 |
|------|----------|------|
| `sycl_c_memcpy`（runtime.sycl） | 一切搬运（H2D / D2H / D2D） | `queue::memcpy(...).wait_and_throw()` |
| `sycl_c_wait`（runtime.sycl） | `ops::synchronize` | 逐条队列 `wait_and_throw`（设备级契约） |
| `device_runs_kernels`（common.h） | 设备可用性探测（一次性） | `queue.wait_and_throw` |

算子提交只往调用线程的 in-order 队列入队。非连续视图物化是内核，异步。

### Vulkan

| 位置 | 触发路径 | 机制 |
|------|----------|------|
| `downloadBuffer`（vulkan_context.cpp） | `to_host` / `item` | 提交拷贝命令后 `waitLocked`（等队首 fence） |
| `synchronize`（vulkan_context.cpp） | `ops::synchronize` | 逐条在飞通道 `waitLocked` |
| `submitAsync` 积压上限（vulkan_context.cpp） | 在飞提交达到 `kMaxInFlight`（512） | 等最老的一次提交——描述符池上限 2000 的前置守卫 |
| `uploadBuffer`（vulkan_context.cpp） | `to_device` / 跨设备 | **只入队**：暂存 buffer 的销毁排到提交完成之后 |

队列积压是唯一一条"非显式点但可能等待"的路径：连续提交超过 512 个算子而不排空
时，第 513 次提交会等最老的一次。`tests/async_contract.cpp` 的提交循环（每次 160 个
算子）刻意留在这个上限之内，因此断言为 0；超过时的等待记在 `Wait::Backlog` 槽位。

## 异步句柄的生命周期与退役

- **CUDA**：块的释放事件记录在分配时所在的那条流上；跨线程交接的数据需要调用方先
  `ops::synchronize()`（每线程一条流，流之间没有顺序关系）。
- **SYCL**：USM 分配归属 context；释放排在同一条 in-order 队列上既有工作之后
  （`sycl_c_free` 的 host_task 路径）。
- **Vulkan**：buffer 的销毁排到引用它的 fence 有信号之后（`destroyBuffer` → 在飞
  通道的 retired 列表），`completed_seq` 按序号推进。

## 计数器设施

`include/_sync_counters.h`：四个槽位（`Copy` / `Synchronize` / `Backlog` /
`OomFree`），relaxed 原子加。常开而不是"仅测试构建"：这些调用点本身就是微秒到
毫秒级的设备往返，一次原子加可以忽略；条件编译会让四条 lane 各少一份覆盖。

## 数据通路：连续 / 非连续的分派（FR-4.2）

统一迭代层（`include/tensorvia/tensor_iterator.hpp`）推导广播布局，`linearizable`
判定"所有操作数与输出同形、行主序连续"——它是分派的唯一依据，后端无关：

| 输入形态 | 判定 | 路径 |
|----------|------|------|
| 连续同形（含带 offset 的连续视图） | `linearizable` | **一维内核**：宿主把参数的 `ndim` 置 0，内核跳过每元素每操作数的 div/mod 解码 |
| 广播（有 stride-0 维） | `broadcastless = false` | strided 内核：重复读取比物化便宜 |
| 跨步视图 | 非 linearizable | strided 内核；或按 `should_materialize`（尾段 < 4 且总量 ≥ 1024）先物化一次再走一维 |

三个设备后端同构：CUDA（`binary_kernel` / `compare_kernel` 的 `ndim == 0` 分支）、
SYCL（同构两个内核）、Vulkan（80 份广播/比较着色器的同一分支）。CPU 后端本就是
这个分派（`linearizable` → 线性 kernel，否则 `iter::drive` 分段）。

一致性由 `tests/p1.cpp` 的 `ContiguousAndStridedPathsAgree` 锁定：同样数值的连续
与跨步输入、四种算术、比较、where、广播、带 offset 的连续视图，结果逐元素一致。
