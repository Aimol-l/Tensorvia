# Specification: 全归约的并行化——三个设备后端的两趟归约

## Metadata
- **Version**: 1.0.0
- **Status**: Draft
- **Author**: Claude Code
- **Created**: 2026-11-02
- **Last Updated**: 2026-11-02

## Overview

**现状**：`docs/performance.md` 记录 CUDA 与 VULKAN 的全归约（`Sum` / `Min` / `Max` / `Mean` / `Prod` 的无轴形式）只有同机 memcpy 参照的 0.0x%，是 FR-4.3 校准中仅有的两项不达标之一（另一项是 VULKAN 类型转换）。根因已定位到**几何而非算法**：核心层把全归约展平为 1-D 后按 `axis=0` 下发，后端轴内核拿到的几何是 `outer=1, inner=1, axis=numel`，于是 `total = outer*inner = 1`；CUDA 的 `blocks_for(1)` 只起 1 个 block，内核里 `idx < total` 仅 thread 0 成立——**256 个线程中只有 1 个**串行走完整个 buffer（`sum f32 16M`：CUDA ~520ms / Vulkan ~560ms，CPU 2.7ms）。SYCL 同为单线程路径，只因绝对耗时较低而贴在 21%（下限 15%）之上。

**目标**：把三个设备后端的全归约从"单线程串行"改为"多 block 两趟归约"，达到 FR-4.3 的归约类下限，且**不新增任何按 dtype 复制的资产**——FR-8 记的维护债不得因本次修复复发。

**非目标**：不重写轴归约内核本身（轴归约的 `total` 通常远大于 1，已达标）；不动 CPU 后端（43% 达标，且是 OpenMP SIMD 路径）；不引入厂商库；不改变公开 API，也不改变核心层"全归约展平后走 axis-0"的策略。

**范围**：CUDA / SYCL / VULKAN 三个后端的归约 host driver 与轴内核、Vulkan 的 5 个归约 vec 着色器族（40 份既有文件）、测试与基准。CPU 后端与 `src/core/ops.cpp` 不在范围内。

**明确不在范围内**：`Var` / `Std` / `Norm` / `LayerNorm` / `RmsNorm` / `LogSoftmax` 的全归约是 CPU-only（`require_cpu` 显式拒绝设备调用），不涉及；`All` / `Any` / `NonZero` 走各自的 grid-stride + 原子内核，本就并行，不涉及。

## 设计

### 两趟几何

当轴归约的几何满足 `outer*inner == 1` 且 `axis >= 触发阈值` 时，后端改走两趟：

| | 几何 `(outer, axis, inner)` | 输出 |
|---|---|---|
| 趟 1 | `(B, ceil(N/B), 1)` | B 个部分值，写入同设备、`out_type` 的 scratch |
| 趟 2 | `(1, B, 1)` | 1 个元素，即最终结果 |

`Sum` / `Min` / `Max` / `Prod` 的部分值再折叠是精确结合的；`Mean` 见下。

### 轴循环按真实元素数截断

`ceil(N/B)` 使 `B*chunk >= N`，趟 1 的末段会读到 buffer 之外。轴循环上限因此从 `axis` 改为

```
limit = min(axis, numel - base)        // base = outer_idx*axis*inner + inner_idx
```

在一切正常布局下 `base + axis*inner <= numel` 恒成立，该截断是**恒等变换**——单趟路径的数值结果逐位不变，这正是它可以直接加进热循环的理由（每个输出元素只多一次 `min`，不在内层循环里）。

`numel` 的来源按后端而异：

- **CUDA / SYCL**：`ReduceAxis`（`cuda_launch.hpp` / `sycl_island.h`，后者经 `common.h` 被设备单元看到，接缝两侧同一份定义）增加 `int64_t numel`，语义为"输入缓冲区的真实元素数"。取 0 表示按几何 `outer*axis*inner` 推导，作为漏填时的兜底——直接信一个可能是 0 的字段会让 `limit` 变负、循环空转、静默算错。
- **VULKAN**：着色器内以 `params.outer_dim * params.axis_dim` 推导（`outer_dim` 是输出总数，其乘积正是元素数），**不增加 push constant**。这与"给全部 40 份着色器加第 4 个字段"相比改动最小；代价是依赖 int32 算术，与既有 `axis_dim` / `uint base_offset` 的取值上限一致（>2^31 元素的张量本来就不被现有几何支持）。

### Mean 的显式分母

两趟路径中 `Mean` 的趟 1 必须用纯累加（不能先除以 `chunk`——末段短于 `chunk`，先除会少算），趟 2 才除，且分母是全归约的元素数 N 而不是趟 2 自己的 `axis`（=B）。

- **CUDA / SYCL**：`ReduceAxis` 增加 `int64_t mean_n`；0 表示用 `axis`（既有单趟调用不变）。
- **VULKAN**：`shader/mean/mean_*.comp` 8 份增加第 4 个 push constant `mean_n`（0 = 用 `axis_dim`），`registerOp(OpType::Mean, ...)` 的 push 尺寸随之更新。这是本次唯一新增的 push constant，仅 mean 一族需要；其余 4 个 op 不加。

### 三处落点

| 后端 | 落点 | 说明 |
|---|---|---|
| CUDA | `src/backend/cuda/device/reduce.cu` 的 `axis_reduce` | 与既有单趟分支并列；常量置于该文件匿名 namespace |
| SYCL | `src/backend/sycl/host/reduce.cpp` 的 `axis_reduce` | 结构同上 |
| VULKAN | `src/backend/vulkan/reduce.cpp` | 5 个 `execute(const Tensor&, int)` 收敛为一个 `axis_reduce_vk(a, axis, out_type, op, what)` helper，两趟逻辑只写一份；5 个 public 入口转薄封装 |

Vulkan 的收敛是本次修复的前置条件：5 个入口目前的形状计算与提交逻辑完全同构，两趟分岔若复制 5 份将不可维护。

### 常量与阈值（初值，待 bench 标定）

- `kFlatReduceBlocks`（B）初值 **1024**：足以饱和消费级 GPU 的 SM 数量级；每 block 承担 `N/1024` 个元素，16M 时为 64 个/线程，仍是访存受限而非 launch 受限。
- `kFlatReduceMinNumel` 初值 **65536**：低于此值时单线程串行的耗时在数十 µs 量级，额外的派发 + scratch 分配不划算。Vulkan 的 submit 固定开销高于 CUDA，其阈值预期需上调，以实测为准。

每后端可独立取值；最终值与依据记入 `docs/performance.md`（NFR-2）。

### 复用既有资产

不新增任何内核或着色器文件。趟 2 复用与趟 1 同一套 (op, dtype) 组合的既有 kernel / pipeline：Vulkan 按**输入** dtype 查流水线，趟 2 的输入是 `out_type`，因此需要例如 `Sum(int8) → int64` 时的 `sumvec_int64`。这些组合在单趟路径中本就存在（同一批 `registerOp` 注册覆盖了它们），但实现前必须逐个核对 5 个 op × 各 dtype 的趟 2 组合齐备，缺失即暴露为一个真实的覆盖缺口。

## Requirements

### Functional Requirements

- **FR-1 两趟几何**：满足触发条件时后端必须走两趟（几何见上），不得以其他方式（如原子累加）替代——原子无法表达 `contract::min_contract` / `max_contract` 的 NaN 传播契约。
- **FR-2 恒等截断**：轴循环上限按真实 `numel` 截断，且在正常布局下恒等；单趟路径数值结果不得改变。
- **FR-3 显式分母**：`Mean` 两趟路径的趟 1 纯累加、趟 2 以 N 为分母；单趟路径行为不变。
- **FR-4 零新增资产**：不新增内核或着色器文件；对既有 40 份 Vulkan 归约着色器的改动仅限"增加截断"与 mean 的 `mean_n` 字段。
- **FR-5 Vulkan driver 收敛**：5 个归约入口收敛为单一 helper，public 入口为薄封装。

### Non-functional Requirements

- **NFR-1 正确性不回归**：四条 lane 既有测试全绿；轴归约结果逐位不变；全归约因累加顺序改变产生的偏差在既有容差内（浮点输入的累加器本就行 `double`，相对偏差 ~1e-16）。
- **NFR-2 可标定**：常量初值入码并注释理由；实测后按 bench 调整，最终值与依据记入 `docs/performance.md`。
- **NFR-3 维护预算**：FR-8 的度量指标"新增一个算子改动的文件数"不得因本次修复上升。

## Test Steps

```sh
# 四条 lane（改前改后各跑一次；mcpp.lock 每次跑完 checkout 回去）
mcpp test
mcpp test --features cuda --accel "cuda13.3+{sm_89}"
mcpp test --features sycl --accel sycl
LD_LIBRARY_PATH=$HOME/.mcpp/registry/data/xpkgs/compat-x-vulkan-runtime/2026.09.07/mcpp_generated/vulkan_runtime/lib \
    mcpp test --features vulkan --accel vulkan1.4

# 工具门禁
python3 -I tools/gen_facade.py --check
python3 -I tools/coverage_matrix.py --check
python3 -I tests/test_coverage_matrix.py

# bench：四条 lane 各跑一次，取 sum f32 16M 改前改后
mcpp build --features cuda --accel "cuda13.3+{sm_89}" && ./build/*/benchmark/bench cuda
```

新增用例（置于设备用例组，按后端可用性 SKIP）：

1. **两趟路径正确性**：`numel >= 阈值` 的全归约 × `Sum` / `Min` / `Max` / `Mean` / `Prod` × 至少 `FLOAT32` / `INT8` / `FLOAT64` 三种 dtype，与 CPU 参考按**相对**容差比对。
2. **末段截断**：一条**素数尺寸**（如 1'000'003）的全归约，专门使 `B*chunk > N` 且末段长度既非 1 也非 `chunk`。这是 FR-2 唯一能被测出的路径——尺寸取 2 的幂时末段长度为 0，截断写了错也发现不了。
3. **轴归约不受影响**：既有轴归约用例即为其证据（AC-4）。

容差说明：`tests/test_support.hpp` 的 `expect_close` 用 1e-5 **绝对**容差，对 1M 个元素的和（量级 ~5e5）远远不够——float32 输出的最终舍入就有 ~0.03。新用例必须用相对容差，或与 double 精度的 CPU 参考比对。

## Acceptance Criteria

- **AC-1 达标**：三个设备后端全归约达到 FR-4.3 归约类下限（CPU 30% / CUDA 1% / SYCL 15% / VULKAN 1%），实测值记入 `docs/performance.md` 与 `benchmark/baseline/`；不新增不达标项，SYCL 不得回退到下限之下。
- **AC-2 新用例**：FR-2 的两趟路径与素数尺寸末段用例在三条设备 lane 上真实执行（不 SKIP）。
- **AC-3 全绿**：四条 lane 全绿；两个 `--check` 与工具测试通过。
- **AC-4 轴归约不变**：既有轴归约用例结果不变（截断恒等的证据）。
- **AC-5 零新增资产**：`git diff --stat` 中无新增 `.comp` 文件；对着色器的改动仅限截断与 `mean_n`。

## Change Log

| 日期 | 版本 | 变更 | 作者 |
|------|------|------|------|
| 2026-11-02 | 1.0.0 | 初稿：两趟归约设计、恒等截断、Mean 显式分母、三后端落点与验收标准 | Claude Code |
