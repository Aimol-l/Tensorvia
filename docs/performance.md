# 性能下限（FR-4.3 校准稿）

> 本文是 spec `specs/2026-11-01-multi-backend-maturity.md` FR-4.3 的校准记录：
> 每个百分比都能由 `benchmark/bench.cpp` 在参考机器上复现。基线原始数据在
> [`benchmark/baseline/2026-11-01-i5-14600kf-rtx4070ti.md`](../benchmark/baseline/2026-11-01-i5-14600kf-rtx4070ti.md)，
> 机器/驱动/编译器/profile 见该文首节。本文不手写 Bench 数字——所有"实测"列
> 都从基线表取。

## 参照（同机）

| 参照 | 数值 | 来源 |
|------|------|------|
| memcpy 带宽（各后端自己的同 dtype 拷贝） | CPU 19.5 / CUDA 594 / SYCL 37 / VULKAN 135 GB/s | bench `memcpy f32 16M` |
| Eigen CPU SGEMM 1024 | 461 GFLOPS | `/usr/include/eigen3`，`-O3 -march=x86-64-v3 -fopenmp` |
| cuBLAS SGEMM 1024（经 PyTorch 2.11 CUDA） | 20731 GFLOPS | `torch.matmul` fp32 |
| PyTorch CPU conv2d 8x64x56x56（oneDNN/MKL） | 652 GFLOPS | `F.conv2d` |
| PyTorch CUDA conv2d 8x64x56x56（cuDNN） | 24208 GFLOPS | `F.conv2d` |

口径：访存受限类按**字节**（读+写）与同机 memcpy 比；计算类按 FLOPs 与同机最佳
实现比。吞吐一律取 bench 的 `total`（提交 + 同步），冷启动单独一列不计入。

## 下限表（校准值）

### 访存受限类（占同机 memcpy 带宽的百分比）

| 子类 | CPU | CUDA | SYCL | VULKAN |
|------|-----|------|------|--------|
| 逐元素 / 激活 | 100% | 70% | 100% | 100% |
| 类型转换（cast） | 100% | 70% | 100% | 100% |
| 转置 | 100% | 15% | 8% | 10% |
| gather / index_select | 50% | 15% | 10% | 1% |

### 归约类（占同机 memcpy 带宽的百分比）

| 子类 | CPU | CUDA | SYCL | VULKAN |
|------|-----|------|------|--------|
| 轴归约 | 100% | 20% | 100% | 30% |
| 全归约 | 100% | 1% | 10% | 1% |

### 计算类（占同机最佳实现的百分比）

| 子类 | 参照 | CPU | CUDA | SYCL | VULKAN |
|------|------|-----|------|------|--------|
| matmul 1024 | Eigen 461 / cuBLAS 20731 / PyTorch-CUDA 20731 | 30% | 8% | 8% | 5% |
| conv2d 8x64x56x56 | PyTorch-CPU 652 / PyTorch-CUDA 24208 | 1% | 3% | 1% | 3% |

定值原则：取"现状"与"可达上限"之间、离现状留 30% 以上回归余量的值。现状离上限
很远的子类（conv2d 2~4%、Vulkan gather 8%）下限因此定得低——下限的作用是抓
回归，不是假装到达上限。

## 现状对照（Task 9 校准时的基线）

| 子类（用例） | CPU | CUDA | SYCL | VULKAN |
|---|---|---|---|---|
| 逐元素/激活（add 16M） | 323% ✅ | 101% ✅ | 214% ✅ | 185% ✅ |
| 类型转换（cast 16M） | 195% ✅ | 98% ✅ | 102% ✅ | **2% ❌** |
| 转置 2D | 231% ✅ | 24% ✅ | 11% ✅ | 16% ✅ |
| gather | 107% ✅ | 82% ✅ | 16% ✅ | 8% ✅ |
| index_select | 483% ✅ | 20% ✅ | 18% ✅ | **1% ✅（贴线）** |
| 轴归约 | 451% ✅ | 28% ✅ | 116% ✅ | 45% ✅ |
| 全归约 | 137% ✅ | **0% ❌** | 22% ✅ | **0% ❌** |
| matmul 1024 | 77% ✅ | 12% ✅ | 11% ✅ | 7% ✅ |
| conv2d | **2% ❌** | **3% ✅（贴线）** | **2% ❌** | **4% ✅（贴线）** |

## 不达标项（单列，含根因与归属）

1. **CUDA / VULKAN 全归约**（0.02% / 0.09% of memcpy）。根因：公共同款 API 把全归约
   展开成 `axis=0` 且 `outer=inner=1` 的轴归约，轴内核的 `total = outer*inner = 1`，
   只剩一个线程串行走完整个轴（`sum f32 16M`：CUDA ~520ms / Vulkan ~560ms，
   CPU 2.7ms）。旧 bench 只测提交、看不到设备耗时，Task 6 的三段计时才暴露。
   修法：全归约走多 block 两趟归约（partial + fold），或让轴内核在 total 过小时
   也按轴切分网格。**未列入本计划的 Files，归属后续性能任务。**
2. **VULKAN 类型转换**（2% of memcpy；submit 72.7ms / sync 10.2ms）。submit 远大于
   sync 说明耗时在宿主侧：`to_type` 在 Vulkan 上走了逐元素的宿主路径而不是一个
   cast 着色器。修法：补 cast 的 Vulkan 着色器路径。**归属后续任务。**
3. **conv2d 全后端**（2~4% of 参照）。当前是朴素直接卷积，无 im2col/隐式 GEMM。
   下限按现状定到 1~3%；要提升需重写内核（FR-3 之后的事）。**归属后续任务。**
4. **CPU 的跨步视图物化**（`gather permute 128x64x64` = 2.5 GB/s ≈ 13% of memcpy）：
   逐元素 div/mod 寻址 + 堆分配；不在 FR-4.3 的点名列表里，记录备查。

## 复现方式

```sh
mcpp test bench                                        # CPU lane
mcpp test --features cuda --accel "cuda13.3+{sm_89}" bench
mcpp test --features sycl --accel sycl bench
LD_LIBRARY_PATH=... mcpp test --features vulkan --accel vulkan1.4 bench
```

同一 lane 连跑两次取中位数；亚毫秒用例在未钉核心的桌面机上运行间方差可超过 10%，
判读时以 `>= 1ms` 的用例为准。
