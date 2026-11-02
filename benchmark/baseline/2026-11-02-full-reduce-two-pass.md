# 性能基线补遗：全归约两趟化（2026-11-02）

> 只重测受 `specs/2026-11-02-full-reduce-parallelism.md` 影响的用例。该改动只碰
> 全归约（`outer*inner == 1` 且 `axis >= 阈值`）的派发路径，轴归约、逐元素与其
> 他用例的代码路径不变，因此不重复采集全表；全量表仍以
> [`2026-11-01-i5-14600kf-rtx4070ti.md`](2026-11-01-i5-14600kf-rtx4070ti.md) 为准。
>
> 口径与首版一致：`total`（submit + sync）算吞吐，冷启动单列不计入。百分比按
> **同一次运行**的 memcpy 折算——它与首版基线表的分母有同机波动，不要跨表相减。

## 机器与构建

| 项 | 值 |
|----|----|
| CPU | Intel Core i5-14600KF（20 核，桌面机，未做核心钉定）|
| 内存 | 31 GB |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER（16 GB，sm_89）|
| 显卡驱动 | 615.78.08（CUDA 13.3）|
| Vulkan | 1.4.351（同一张 GPU）|
| SYCL 设备 | Intel OpenCL CPU（OpenCL 3.0）——本机无 Intel GPU |
| 编译器 | clang 23.1.4（mcpp toolchain llvm@22.1.8；CUDA 设备单元 nvcc）|
| 构建 profile | mcpp release（`-march=x86-64-v3`，OpenMP 全线程）|
| 日期 | 2026-11-02 |
| commit | 全归约两趟化（`864f3ed` 之后的工作区，未提交）|

## 受影响用例（total 口径）

| 用例 | CPU | CUDA | SYCL | VULKAN |
|------|-----|------|------|--------|
| memcpy f32 16M（同次运行的参照） | — | 604.39 GB/s | 34.25 GB/s | 138.62 GB/s |
| **sum f32 16M（全归约）** | 未重测 | **38.10 GB/s** | **38.79 GB/s** | **27.49 GB/s** |
| 占同次 memcpy | — | **6.3%** | **113%** | **19.8%** |
| sum-axis1 4096x4096（轴归约，未改路径） | — | 164.13 GB/s | 41.41 GB/s | 63.01 GB/s |

total 耗时（ms）：

| 用例 | CUDA | SYCL | VULKAN |
|------|------|------|--------|
| sum f32 16M | **1.761** | **1.730** | **2.442** |
| sum f32 16M（首版基线） | 521.782 | 521.810 | 526.871 |
| 加速比 | 296× | 302× | 216× |

FR-4.3 归约类下限：轴归约 100/10/100/25，全归约 30/1/15/1（CPU/CUDA/SYCL/VULKAN）。
全归约现状 6.3% / 113% / 19.8%，CUDA 与 VULKAN 均从 0% 提到下限之上，SYCL 从
20.8% 提到 113%——超过 100% 是因为归约只读一遍而 memcpy 读+写，分母口径偏保守。

## 采集说明与已知偏差

- **CUDA 与 VULKAN**：`bench <backend>`，未改动 bench。
- **SYCL**：本机 SYCL 设备取不到 `random_kernel<float>`（首版基线记录在案的既有
  环境问题），bench 在第一个用例就抛 `sycl Random: device failure`。本次临时把
  bench 的 `Tensor::Random` 换成"宿主生成后搬运"以绕过它，测完已 `git checkout`
  还原。受影响的只有数据来源，被测内核路径与调用形态完全一致。
- **CPU 未重测**：CPU 后端不在此次改动范围内，首版的 26.75 GB/s / 43% 仍有效。
- 单次采集，未做多次中位数；首版基线是同机多次运行的结果。跨表比较时以数量级
  与百分比为准，不要抠个位数。
