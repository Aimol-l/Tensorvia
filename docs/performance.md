# 性能下限（FR-4.3 校准稿）

> 本文是 spec `specs/2026-11-01-multi-backend-maturity.md` FR-4.3 的校准记录：
> 每个百分比都能由 `benchmark/bench.cpp` 在参考机器上复现。基线原始数据在
> [`benchmark/baseline/2026-11-01-i5-14600kf-rtx4070ti.md`](../benchmark/baseline/2026-11-01-i5-14600kf-rtx4070ti.md)，
> 机器/驱动/编译器/profile 见该文首节。本文不手写 Bench 数字——所有"实测"列
> 都从基线表取。

## 参照（同机）

| 参照 | 数值 | 来源 |
|------|------|------|
| memcpy 带宽上限（与内核同并行度） | **CPU 61.9** / CUDA 589.5 / SYCL 35.7 / VULKAN 143.1 GB/s | bench `memcpy-mt f32 16M`（CPU：OpenMP 并行拷贝）与 `memcpy f32 16M`（设备：clone 本身就是并行拷贝） |
| （备查）CPU 单线程 clone | 19.9 GB/s | bench `memcpy f32 16M`。**不作为上限**：拿它当分母会让 OpenMP 逐元素内核"超过 100%"，百分比失去意义 |
| Eigen CPU SGEMM 1024 | 461 GFLOPS | `/usr/include/eigen3`，`-O3 -march=x86-64-v3 -fopenmp` |
| cuBLAS SGEMM 1024（经 PyTorch 2.11 CUDA） | 20731 GFLOPS | `torch.matmul` fp32 |
| PyTorch CPU conv2d 8x64x56x56（oneDNN/MKL） | 652 GFLOPS | `F.conv2d` |
| PyTorch CUDA conv2d 8x64x56x56（cuDNN） | 24208 GFLOPS | `F.conv2d` |

口径：访存受限类按**字节**（读+写）与同机并行 memcpy 比；计算类按 FLOPs 与同机
最佳实现比。吞吐一律取 bench 的 `total`（提交 + 同步），冷启动单独一列不计入。
注意 bench 报告的三列是各自分布的中位数，单次里 `submit + sync == total` 严格
成立，但三个中位数不必相加相等（统计性质，非度量错误）。

## 下限表（校准值）

### 访存受限类（占同机并行 memcpy 带宽的百分比）

| 子类 | CPU | CUDA | SYCL | VULKAN |
|------|-----|------|------|--------|
| 逐元素 / 激活 | 70% | 70% | 100% | 100% |
| 类型转换（cast） | 40% | 50% | 70% | 100% |
| 转置 | 50% | 15% | 8% | 8% |
| gather / index_select | 20% | 15% | 10% | 1% |

### 归约类（占同机并行 memcpy 带宽的百分比）

| 子类 | CPU | CUDA | SYCL | VULKAN |
|------|-----|------|------|--------|
| 轴归约 | 100% | 10% | 100% | 25% |
| 全归约 | 30% | 1% | 15% | 1% |

### 计算类（占同机最佳实现的百分比）

| 子类 | 参照 | CPU | CUDA | SYCL | VULKAN |
|------|------|-----|------|------|--------|
| matmul 1024 | Eigen 461 / cuBLAS 20731 / PyTorch-CUDA 20731 | 60% | 8% | 8% | 5% |
| conv2d 8x64x56x56 | PyTorch-CPU 652 / PyTorch-CUDA 24208 | 1% | 2% | 1% | 3% |

定值原则：取"现状"与"可达上限"之间、离现状留 30% 以上回归余量的值。现状离上限
很远的子类（conv2d 2~4%、Vulkan gather 8%）下限因此定得低——下限的作用是抓
回归，不是假装到达上限。

## 现状对照（Task 9 校准 / Task 10 验收时的基线）

| 子类（用例） | CPU | CUDA | SYCL | VULKAN |
|---|---|---|---|---|
| 逐元素/激活（add 16M） | 102% ✅ | 104% ✅ | 206% ✅ | 160% ✅ |
| 类型转换（cast 16M） | 60% ✅ | 75% ✅ | 99% ✅ | **2% ❌** |
| 转置 2D | 72% ✅ | 25% ✅ | 10% ✅ | 11% ✅ |
| gather | 32% ✅ | 84% ✅ | 16% ✅ | 8% ✅ |
| index_select | 120% ✅ | 20% ✅ | 16% ✅ | 1% ✅（贴线） |
| 轴归约 | 144% ✅ | 16% ✅ | 142% ✅ | 38% ✅ |
| 全归约（Sum/Prod） | 43% ✅ | **0% ❌** | 21% ✅ | **0% ❌** |
| matmul 1024 | 82% ✅ | 12% ✅ | 11% ✅ | 8% ✅ |
| conv2d | 2% ✅ | 3% ✅（贴线） | 3% ✅ | 4% ✅（贴线） |

## 不达标项（单列，含根因与归属）

1. **CUDA / VULKAN 全归约**（Sum 与 Prod 都是 0.0x% of memcpy，下限 1%）。根因：
   公共同款 API 把全归约展开成 `axis=0` 且 `outer=inner=1` 的轴归约，轴内核的
   `total = outer*inner = 1`，只剩一个线程串行走完整个轴（`sum f32 16M`：CUDA
   ~520ms / Vulkan ~560ms，CPU 2.7ms）。旧 bench 只测提交、看不到设备耗时，
   Task 6 的三段计时才暴露。修法：全归约走多 block 两趟归约（partial + fold），
   或让轴内核在 total 过小时也按轴切分网格。**未列入本计划的 Files，归属后续
   性能任务。**
2. **VULKAN 类型转换**（2% of memcpy，下限 100%；submit 72.7ms / sync 10.2ms）。
   submit 远大于 sync 说明耗时在宿主侧：`to_type` 在 Vulkan 上走了逐元素的宿主
   路径而不是一个 cast 着色器。修法：补 cast 的 Vulkan 着色器路径。**归属后续
   任务。**

贴线项（达标但余量 < 30%，回归会立刻踩线，值得回头看）：CUDA conv2d 3%、
VULKAN conv2d 4%、VULKAN index_select 1%。conv2d 全后端都只有参照的 2~4%——
当前是朴素直接卷积，无 im2col / 隐式 GEMM；下限按现状定到 1~3%，要提升需重写
内核（FR-3 之后的事）。

另记录（不在 FR-4.3 的点名类别里）：设备后端的生成器 `arange` / `randn`（宿主
生成 + 搬运，3~8 GB/s）；CPU 的跨步视图物化（`gather permute 128x64x64` ≈ 2.5
GB/s，逐元素 div/mod 寻址 + 堆分配）。

## 编译期基线

CPU 后端以 `-march=x86-64-v3`（AVX2/FMA/F16C）为**编译期硬基线**，无运行时
CPU 特性派发：`cpu_context.cpp` 的 `__builtin_cpu_supports` 只用于打印，不参与
选择。在更旧的 CPU 上运行会 SIGILL——这是既有的构建配置决策（mcpp 与 CMake 两条
路径统一），不是某个内核的选择。

## 复现方式

```sh
mcpp test bench                                        # CPU lane
mcpp test --features cuda --accel "cuda13.3+{sm_89}" bench
mcpp test --features sycl --accel sycl bench
LD_LIBRARY_PATH=... mcpp test --features vulkan --accel vulkan1.4 bench
```

同一 lane 连跑两次取中位数；亚毫秒用例在未钉核心的桌面机上运行间方差可超过 10%，
判读时以 `>= 1ms` 的用例为准（Task 10 复跑与 Task 9 采数对比：113 项中位差 3%）。
