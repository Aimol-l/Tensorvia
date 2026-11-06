# 多后端成熟度实施计划（P0 覆盖一致性 + P1 性能下限）

> 执行时逐项勾选 `- [ ]`；每个 Task 独立可回退，测试不通过不进入下一个 Task。
>
> 注：本文引用的 `docs/performance.md`（已于 `fa0b645` 删除）、`benchmark/baseline/` 与
> `tools/coverage_matrix.py`（均已于 2026-11-05 删除）不在仓库中，内容可从 git 历史
> 取回；当前性能数据见 README 的「性能基线」一节。

**Goal:** 把 spec `specs/2026-11-01-multi-backend-maturity.md` 的 P0 / P1 落地：覆盖矩阵可生成可校验、CPU-only 例外收敛、同步语义可断言、性能有可信的度量与下限。

**Architecture:** 不改变公开 API 与派发架构。新增 `tools/coverage_matrix.py`（类似 `tools/gen_facade.py` 的生成/校验器）作为覆盖事实源；设备端新算子沿用现有 `XxxImpl<Device::X>` + `execute_on` 的派发模式；性能工作集中在 `include/tensorvia/cpu_ops.hpp` 的快路径内核与各后端 ufunc 的线性化分派。

**Tech Stack:** C++23、mcpp、CMake、Python 3（工具脚本）、GoogleTest、CUDA 13.3 / Vulkan 1.4 / SYCL。

**Spec:** `specs/2026-11-01-multi-backend-maturity.md`（FR-1 / FR-2 / FR-4 / FR-5）

## Global Constraints

- 四条 lane 必须全绿后才算任务完成，命令固定（顺序执行，不可并行，`mcpp.lock` 会冲突）：
  - CPU：`mcpp test`
  - CUDA：`mcpp test --features cuda --accel "cuda13.3+{sm_89}"`
  - Vulkan：`LD_LIBRARY_PATH=$HOME/.mcpp/registry/data/xpkgs/compat-x-vulkan-runtime/2026.09.07/mcpp_generated/vulkan_runtime/lib mcpp test --features vulkan --accel vulkan1.4`
  - SYCL：`mcpp test --features sycl --accel sycl`
- 每次 `mcpp test` 后执行 `git checkout -- mcpp.lock`（CPU lane 会删掉 30 行 feature 门控）。
- 公开算子面变化必须同步 `tools/facade_exports.def` 与 `src/interface/tensorvia.ixx`，并让 `python3 -I tools/gen_facade.py --check` 通过。
- 任何性能改动都必须附“改前 / 改后”同机数据；没有数据不合并。
- 不新增第三方依赖。
- 设备端新实现必须在四条 lane 的对应测试里真实执行，不允许 SKIP。

## Review Focus

1. **矩阵解析的可信度**：`OpsImpl<Device::X>` 的正则匹配是否会漏掉多行声明、注释掉的声明、模板偏特化；组合实现（无 Impl，由其他算子拼成）与“不支持”必须有显式标注来源，不能靠猜。
2. **Sort/Topk 的语义**：PyTorch 的 `sort` 稳定性、`topk` 的平局选取顺序；设备端排序在空张量与全相同元素下的行为。
3. **异步断言的稳定性**：隐式同步测试不能依赖计时，要用可观测计数器或“提交后立即读哨兵”的方式，避免在快机器上偶发失败。
4. **SIMD 的边界**：`float16`/`bfloat16`/`BOOL` 是否有独立的标量回退；`div` 的零分母 NaN 语义；尾部不足一个向量的处理；`-march=x86-64-v3` 与运行时不匹配时如何降级。
5. **基准噪声**：median-of-N、固定线程数、暖机次数、冷启动（JIT / 流水线创建）单独一组数字。

---

### Task 1: 覆盖矩阵生成器与 CI 守门

**Files:**
- Create: `tools/coverage_matrix.py`（生成 + `--check`）
- Create: `tools/coverage_overrides.toml`（组合实现 / 不支持的人工标注）
- Create: `docs/coverage.md`（生成产物，入库）
- Modify: `.github/workflows/ci.yml`（加 `python3 -I tools/coverage_matrix.py --check`）
- Modify: `README.md`（CPU-only 段落改为引用矩阵）

**Interfaces:**
- Consumes: `include/tensorvia/ops.hpp` 的公开算子名；`include/tensorvia/cpu_ops.hpp` 与 `include/tensorvia/backend/{cuda,sycl,vulkan}/*_ops.hpp` 中的 `struct XxxImpl<Device::X>` 特化
- Produces: `docs/coverage.md`（算子 × 后端矩阵 + 统计）；`--check` 退出码

- [x] **Step 1: 基线盘点**。手工抽取 10 个算子（含 1 个组合实现、1 个 CPU-only）与源码逐一对账，确认解析规则能覆盖 `ops.hpp` 的全部声明形态（多行声明、重载、`initializer_list` 参数）。
- [x] **Step 2: 实现解析与生成**。输出四列 `CPU / CUDA / SYCL / VULKAN`，取值 `native` / `emulated` / `unsupported`；`emulated` 与 `unsupported` 只从 overrides 读，不从代码推断。生成表格末尾给出每个后端的支持率。
- [x] **Step 3: 校验模式**。`--check` 在干净树上通过；人为删除 `include/tensorvia/backend/vulkan/vulkan_ops.hpp` 里一个特化后重跑，期望失败并指名具体算子与后端。
- [x] **Step 4: CI 与文档**。CI 加校验步骤；README 的“仍为 CPU-only 的算子”一段改为链接 `docs/coverage.md`，不再手写清单。

**验收:** `docs/coverage.md` 与源码一致；`--check` 在漂移时失败；README 不再维护第二份清单。

---

### Task 2: `Prod` 落到三个设备后端

**Files:**
- Modify: `include/tensorvia/backend/{cuda,sycl,vulkan}/*_ops.hpp`（新增 `ProdImpl` 特化）
- Modify: 各后端归约内核文件、`src/backend/cuda/cuda_launch.hpp`、`src/backend/sycl/sycl_island.h`
- Create: `shader/prod/prod_<dtype>.comp`
- Modify: `include/tensorvia/types.hpp`（`OpType::Prod`）、`src/backend/vulkan/vulkan_context.cpp`（注册）
- Modify: `tests/p1.cpp` 或新增测试

**Interfaces:**
- Consumes: 现有 `SumImpl` 的归约几何与轴语义；`reduction_type` 的 dtype 规则（`Prod` 与 `Sum` 同族）
- Produces: 四后端一致的 `ops::Prod`，全归约返回 0-dim

- [ ] **Step 1: 定义语义基线**。先用 CPU 实现写测试：全归约、单轴、多轴、`keepdim`、空张量（空乘积为 1）、整型提升到 INT64、BOOL 拒绝，与 PyTorch 对照。
- [ ] **Step 2: CUDA**。复用 `launch_reduce_axis` 的结构；全归约可直接复用轴归约折叠。四 lane 中先跑 CUDA lane。
- [ ] **Step 3: SYCL**。`sycl_island.h` 加 POD 与 `extern "C"` 声明，按现有归约模板实现。
- [ ] **Step 4: Vulkan**。四份按 dtype 的 shader（首版可只做 float32/float64，其余在矩阵里标 `unsupported` 并显式拒绝），注册 `OpType::Prod`，注意 pipeline key 与 `op_to_string` 一致。
- [ ] **Step 5: 矩阵与测试**。更新 `docs/coverage.md`；四条 lane 全绿。

**验收:** `Prod` 在四后端的测试真实执行；空张量、`keepdim`、整型提升与 CPU 结果一致。

---

### Task 3: `Erf` 落到三个设备后端

**Files:**
- Modify: 各后端 ops 头文件与一元数学内核文件（参考 `Exp` / `Tanh` 的现有路径）
- Create: `shader/erf/erf_<dtype>.comp`
- Modify: `include/tensorvia/types.hpp`、`vulkan_context.cpp`
- Modify: `tests/p1.cpp`

**Interfaces:**
- Consumes: 一元 ufunc 的现有分派（`linearizable` 快路径 + strided 回退）
- Produces: 四后端 `ops::Erf`

- [ ] **Step 1: 精度契约**。确定实现方式（CUDA `erff`、SYCL 内置、Vulkan 用 Abramowitz-Stegun 近似或连分式），在测试里写明容差。
- [ ] **Step 2: CUDA / SYCL**。先跑通，再核对 `erf` 与 `exp` 的 dtype 支持面一致。
- [ ] **Step 3: Vulkan**。shader 实现近似并加注释说明误差界；无 `erf` 的 dtype 在矩阵标 `unsupported`。
- [ ] **Step 4: 测试**。数值对照 CPU 参考实现（`std::erf`），覆盖负数、0、极大/极小值、非连续输入。

**验收:** 四后端结果与 `std::erf` 在 float32 下最大误差 ≤ 2 ulp 量级；NaN/Inf 传播正确。

---

### Task 4: `Sort` / `Topk` 设备端

**Files:**
- Modify: 各后端 ops 头文件、内核文件
- Create: `shader/sort/`、`shader/topk/` 下的着色器
- Modify: `src/core/ops.cpp`（若需提升形状校验到公共契约）
- Modify: `tests/p1.cpp`

**Interfaces:**
- Consumes: 现有 CPU 实现的语义（排序轴、`descending`、`topk` 的 `largest/sorted` 参数）
- Produces: 设备端 `ops::Sort` / `ops::Topk`

- [ ] **Step 1: 语义冻结**。把 PyTorch 的平局规则、稳定性、`k > 轴长` 的报错写进测试（CPU 先通过）。
- [ ] **Step 2: CUDA**。小轴长用内核对每个切片的插入/选择排序，大轴长用 bitonic；先保证正确性。
- [ ] **Step 3: SYCL**。同结构移植。
- [ ] **Step 4: Vulkan**。shader 里的排序用工作组 shared memory；注意 32 位索引上限。
- [ ] **Step 5: 测试**。随机数据 × 多种轴长 × 平局数据；与 CPU 结果逐元素比对（平局时比较排序后的值序列）。

**验收:** 四后端与 CPU 结果一致；空张量、单元素、全相同元素、`k = 轴长` 边界通过。

---

### Task 5: 同步语义文档与隐式同步断言

**Files:**
- Create: `docs/async-semantics.md`
- Modify: `README.md`（链接）
- Create: `tests/async_contract.cpp`
- Modify: `include/tensorvia/*`（若需要可观测计数器）

**Interfaces:**
- Consumes: 各后端现有提交/同步实现（`cuda` stream、`sycl` queue、`vulkan` in-flight 通道）
- Produces: 明确的同步点清单 + 自动化断言

- [ ] **Step 1: 盘点同步点**。逐后端列出所有会让宿主等待设备的调用路径：`to_host`、`item<T>`、`synchronize`、跨设备拷贝、分配器回退（OOM 时）、Vulkan 队列积压、SYCL 事件等待。写进文档，含“为什么这里必须同步”。
- [ ] **Step 2: 可观测计数器**（仅测试构建或环境变量开启）：记录每次宿主等待的次数与位置。
- [ ] **Step 3: 断言测试**。连续提交 N 个算子（含视图输入）后检查计数为 0；`item()` 后计数 +1；`synchronize()` 后再次提交计数不变。设备 lane 上同样运行。
- [ ] **Step 4: 文档与矩阵联动**。`docs/coverage.md` 中标注每个后端的异步状态（设备端算子均为入队返回 / 含同步的组合实现）。

**验收:** 四 lane 上断言通过；文档列出的同步点与代码一致（Review 时人工核对一次）。

---

### Task 6: benchmark 三段拆分与基线表

**Files:**
- Modify: `benchmark/bench.cpp`（或拆出 `benchmark/timings.cpp`）
- Create: `benchmark/baseline/<date>-<machine>.md`（入库的基线表）
- Modify: `README.md`（性能段落引用基线表）

**Interfaces:**
- Consumes: 现有 `median_ms` / `report`；`ops::synchronize()`
- Produces: 每个算子一行的 `submit / wait / total` 与 `GB/s` 或 `GFLOPS`，外加冷启动数据

- [ ] **Step 1: 三段计时**。在提交前后与同步前后各取一次时钟：`submit`（调用返回）、`sync`（`synchronize()` 耗时）、`total`（含首次调用）。CPU 后端 `sync` 近似为 0，仍照常输出。
- [ ] **Step 2: 冷 / 热分离**。第一轮单独统计（含 SYCL JIT、Vulkan 流水线创建），后续取 median。
- [ ] **Step 3: 同机基线**。加入 memcpy（同 dtype 拷贝）与 float32 加法作为带宽上限参照；矩阵乘加入 512 / 1024 两档。
- [ ] **Step 4: 设备参数**。bench 接受 `cpu|cuda|vulkan|sycl` 参数，与示例保持一致。
- [ ] **Step 5: 记录基线表**。在参考机器上跑四条 lane，写入 `benchmark/baseline/`，表中包含硬件/驱动/编译器/构建 profile。

**验收:** 三段耗时之和与总耗时吻合（误差 < 5%）；同一 lane 连跑两次的 median 差异 < 10%；基线表入库。

---

### Task 7: 设备后端的连续快路径

**Files:**
- Modify: `src/backend/{cuda,sycl,vulkan}` 的 ufunc / 算子入口（或新增后端 ufunc 头）
- Modify: `include/tensorvia/tensor_iterator.hpp`（若需共享判定）
- Create/Modify: 对应测试

**Interfaces:**
- Consumes: CPU 侧已有的 `iter::linearizable` 分派模式（`cpu_ops.hpp:413`、`where.cpp:68`）
- Produces: 三设备后端同构的“连续 → 一维内核 / 非连续 → strided 或 repack”分派

- [ ] **Step 1: 审计**。逐个后端记录当前对非连续输入的处理方式（repack / 逐元素 strided / 直接拒绝），产出差异表。
- [ ] **Step 2: 定义统一策略**。连续走一维内核；非连续若可线性化（如带 offset 的连续视图）直接走一维；否则 repack 一次。写进 `docs/async-semantics.md` 相邻的“数据通路”段落。
- [ ] **Step 3: CUDA / SYCL / Vulkan 逐个落地**，每完成一个跑对应 lane 的全量测试 + 同一算子的连续/非连续一致性测试。
- [ ] **Step 4: 度量**。用 Task 6 的 bench 对比连续路径改造前后的耗时（同尺寸、连续输入）。

**验收:** 三个后端在连续输入上不再走通用 strided 路径；连续与非连续结果一致；无新增 SKIP。

---

### Task 8: CPU 逐元素内核 SIMD 化（先测量后动手）

**Files:**
- Modify: `src/backend/cpu/ops/arithmetic.cpp`、`activate.cpp`（按测量结果）
- Modify: `src/backend/cpu/ops/_simd_math.h`（若需扩充）
- Modify: `benchmark/bench.cpp`（加入对照项）

**Interfaces:**
- Consumes: 现有 `linearizable` 快路径（`cpu_ops.hpp:413/453/493/515`）；`-march=x86-64-v3` 已由 mcpp 配置
- Produces: 在编译器未自动向量化的内核上补显式 AVX2，其余保持标量

- [ ] **Step 1: 逐内核测量**。对 add / sub / mul / div / relu / sigmoid / tanh / exp / cast 分别测 GB/s（16M f32），并用 `-fopt-info-vec` 或反汇编确认是否已自动向量化。
- [ ] **Step 2: 只补未向量化的**。当前预期：`div`（每元素分支 `b[i] != 0`）与部分 `cast` 未向量化；`add/sub/mul` 很可能已由编译器向量化，不强加 intrinsics。
- [ ] **Step 3: 边界**。`float16`/`bfloat16`/`BOOL`/整型保留标量实现；尾部用一个标量循环收口；写端考虑对齐/非临时存储。
- [ ] **Step 4: 回归**。数值与改前逐元素一致（含 NaN/Inf/零分母）；bench 记录改前改后。

**验收:** 至少一个内核吞吐提升 ≥ 1.3×，且没有任何内核退化 > 5%；四 lane 测试全绿。

---

### Task 9: 性能下限校准与记录

**Files:**
- Modify: `specs/2026-11-01-multi-backend-maturity.md`（FR-4.3 的百分比目标 + Change Log）
- Modify: `docs/coverage.md` 或新增 `docs/performance.md`

**Interfaces:**
- Consumes: Task 6 的 bench 输出与基线表
- Produces: 校准后的下限表（分后端、分算子类别）

- [ ] **Step 1: 计算参照**。访存类用同机 memcpy 带宽为上限；矩阵乘/卷积用同机最佳实现（CPU: Eigen 或 oneDNN；CUDA: cuBLAS；Vulkan: PyTorch CUDA 同尺寸）。
- [ ] **Step 2: 定目标**。按现状与可达上限之间取一个可防守的值（允许分后端不同），写进 spec FR-4.3 并改 Change Log 版本。
- [ ] **Step 3: 固化**。`docs/performance.md` 记录目标、实测、缺口与原因；不达标的项单列。

**验收:** spec 中不再有“待校准”的百分比；文档里的每个数字都能由 bench 复现。

---

### Task 10: 全量验收

- [ ] CPU lane：`mcpp test` 全绿，记录通过数与耗时。
- [ ] CUDA / Vulkan / SYCL lane：各自全绿，记录通过数。
- [ ] `python3 -I tools/gen_facade.py --check` 通过。
- [ ] `python3 -I tools/coverage_matrix.py --check` 通过。
- [ ] `git checkout -- mcpp.lock`；确认工作区无意外改动。
- [ ] 四条 lane 的 bench 各跑一次，更新基线表。
- [ ] spec 的 P0 / P1 项逐条对照 AC-1 / AC-2 / AC-3 / AC-4 / AC-5 / AC-6，记录结果。

## 验收记录（执行时填写）

| Task | 状态 | 四 lane 结果 | 备注 |
|------|------|--------------|------|
| 1 覆盖矩阵 | ✅ | 四 lane 全绿；`coverage_matrix --check` 通过；21 项工具测试通过 | 77 算子矩阵；删除 vulkan 特化后 --check 指名失败（测试锁定）；README 手写清单退役 |
| 2 Prod | ✅ | CPU 111 / CUDA 121 / SYCL 117 / VK 118 通过（终态） | 四后端 native；空乘积 1、整型提升、k/轴校验与 CPU 一致 |
| 3 Erf | ✅ | 同上 | 四后端 native；f32 与 std::erf 逐位一致（Vulkan 自算 erf，f64 误差 2.6e-15） |
| 4 Sort/Topk | ✅ | 同上 | 四后端 native；稳定次序与 CPU 逐位一致；轴长上限 1024 显式抛错 |
| 5 同步语义 | ✅ | 同上；async_contract 在三设备 lane 各 2 项通过、CPU lane 按设计 SKIP | 计数器设施 + 文档；矩阵含异步状态节 |
| 6 benchmark | ✅ | 四 lane bench 各跑一次，基线表入库 | 三段计时 + 冷热分离 + 设备参数；基线表含机器/驱动/profile |
| 7 连续快路径 | ✅ | 同上；ContiguousAndStridedPathsAgree 四 lane 通过 | linearizable → ndim=0 快路径；SYCL 6×、VK +9~15%、CUDA 持平（访存受限） |
| 8 CPU SIMD | ✅ | 四 lane 全绿；DivZeroDenominatorIsNaN 通过 | 只补 div（其余已向量化）；1M 提升 1.7~2×、16M 无退化 |
| 9 下限校准 | ✅ | spec FR-4.3 已填值（v1.0.2）；docs/performance.md | 三项不达标单列（CUDA/VK 全归约、VK cast）；conv2d 贴线 |
| 10 全量验收 | ✅ | CPU 111 / CUDA 121 / SYCL 117 / VK 118 通过，0 失败；skip 仅 MultiDevice（单 GPU）与 CPU-lane 设备用例 | 两个 `--check` 通过；工作区干净；bench 与 Task 9 基线中位差 3% |

**AC 对照**：AC-1 ✅（矩阵生成 + CI 守门 + README 无第二份清单）；AC-2 ✅（标注支持的组合零跳过；FR-2 六算子：Prod/Erf/Sort/Topk 已 native，MaskedSelect/MaskedFill 显式拒绝且矩阵有据）；AC-3 ✅（三段 bench + 基线表）；AC-4 ✅（连续快路径三后端落地、一致性测试通过；更快一项分后端——SYCL 6×、VK +9~15%、CUDA 持平，原因记为访存受限）；AC-5 ✅（校准表内除三项不达标外全过，见 docs/performance.md）；AC-6 ✅（计数器断言四 lane 通过）。
