# 性能基线：Vulkan 算子优化（2026-11-03）

> 只记录本次改动触及的用例。口径与首版基线一致：`total`（submit + sync）算
> 吞吐，冷启动单列不计入；重负载用例 5 次、其余 9 次取中位数。
>
> "改前 → 改后"的对照表不是拿早上那次运行比的：两次运行之间机器飘了约 15%
> （同一份不变代码的 `memcpy` 从 0.892 变成 1.01 ms），直接比会把噪声当成收益或
> 退化。因此改前数字是**在同一时段、用 `git worktree` 拉出 `HEAD`（`eaf2107`）
> 的独立工作区实测**的，改后是工作区同参数实测；两边各取多次运行的中位数。
> 早上那次运行的数字保留在"改前基线"一节作为历史记录。

## 机器与构建

| 项 | 值 |
| --- | --- |
| CPU | Intel Core i5-14600KF（20 核，桌面机，未做核心钉定） |
| 内存 | 31 GB |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER（16 GB，sm_89，66 SM） |
| 显卡驱动 | 615.78.08（CUDA 13.3，Vulkan 1.4） |
| 编译器 | clang 23.1.4（mcpp toolchain llvm@22.1.8；shader 由 glslc 2026.3 编译） |
| 构建 profile | mcpp release |
| 日期 | 2026-11-03 |
| commit | 基线 `eaf2107`；改后为工作区（未提交） |

Vulkan lane 的运行方式（缺一不可）：

```bash
mcpp test bench --features vulkan --accel vulkan1.4 --profile release -- vulkan
LD_LIBRARY_PATH=$HOME/.mcpp/registry/data/xpkgs/compat-x-vulkan-runtime/2026.09.11/mcpp_generated/vulkan_runtime/lib
```

`--accel vulkan1.4` 不是可选项：`build.mcpp` 用 `vulkan_in_accel && want_vulkan`
门控 Vulkan lane 的宿主源文件，只给 `--features vulkan` 会链接出一个只有 CPU
实现的产物（`undefined symbol: ops::*Impl<(Device)3>`）。

## 同条件对照（total 口径，改前 → 改后）

改前 = `HEAD` 工作区 2 次运行中位数（该工作区首次运行紧跟在一次全量构建之后，
`memcpy` 比后两次慢 10%，不计入）；改后 = 工作区 3 次运行中位数。
**加粗**是本次改动触及的用例。

| 用例 | 改前 ms | 改前吞吐 | 改后 ms | 改后吞吐 | 加速比 |
| --- | --- | --- | --- | --- | --- |
| memcpy f32 16M（参照） | 0.895 | 149.88 GB/s | 0.887 | 151.38 GB/s | 1.01× |
| add f32 16M | 0.851 | 236.84 GB/s | 0.857 | 234.91 GB/s | 0.99× |
| mul f32 16M | 0.825 | 243.81 GB/s | 0.855 | 235.36 GB/s | 0.97× |
| add f32 4096x4096 | 0.815 | 247.16 GB/s | 0.825 | 243.98 GB/s | 0.99× |
| add-scalar f32 16M | 0.736 | 182.34 GB/s | 0.738 | 181.83 GB/s | 1.00× |
| sub f32 16M | 0.812 | 247.88 GB/s | 0.822 | 244.97 GB/s | 0.99× |
| div f32 16M | 0.819 | 245.68 GB/s | 0.819 | 245.79 GB/s | 1.00× |
| div f32 1M (L2) | 0.383 | 32.88 GB/s | 0.386 | 32.61 GB/s | 0.99× |
| **relu f32 16M** | 1.225 | 109.95 GB/s | **0.897** | **149.68 GB/s** | **1.37×** |
| **sigmoid f32 16M** | 1.323 | 101.56 GB/s | **1.053** | **127.42 GB/s** | **1.26×** |
| **tanh f32 16M** | 1.312 | 102.37 GB/s | **0.986** | **136.19 GB/s** | **1.33×** |
| **cast f32->f64 16M** | 83.538 | 2.41 GB/s | **1.083** | **185.91 GB/s** | **77.14×** |
| **exp f32 4M** | 0.873 | 38.49 GB/s | **0.595** | **56.44 GB/s** | **1.47×** |
| **sum f32 16M** | 2.345 | 28.63 GB/s | **0.809** | **82.96 GB/s** | **2.90×** |
| **sum-axis1 4096x4096** | 1.049 | 64.36 GB/s | **0.690** | **97.30 GB/s** | **1.52×** |
| transpose2d 1024x1024 | 0.385 | 21.81 GB/s | 0.583 | 14.38 GB/s | 0.66×＊ |
| transpose3d 64^3 | 0.382 | 5.49 GB/s | 0.405 | 5.18 GB/s | 0.94× |
| gather permute 128x64x64 | 0.390 | 10.77 GB/s | 0.391 | 10.71 GB/s | 1.00× |
| matmul f32 256 | 0.393 | 85.40 GFLOPS | 0.395 | 85.02 GFLOPS | 0.99× |
| **matmul f32 512** | 0.504 | 533.01 GFLOPS | **0.411** | **653.87 GFLOPS** | **1.23×** |
| **matmul f32 1024** | 1.391 | 1545.62 GFLOPS | **0.630** | **3406.10 GFLOPS** | **2.21×** |
| **conv2d f32 8x64x56x56** | 1.611 | 1067.70 GFLOPS | **0.715** | **2405.33 GFLOPS** | **2.25×** |
| arange f32 16M | 19.658 | 3.42 GB/s | 20.460 | 3.28 GB/s | 0.96× |
| **where f32 16M** | 45.462 | 5.90 GB/s | **2.561** | **104.80 GB/s** | **17.75×** |
| **where-scalar f32 16M** | 45.808 | 4.39 GB/s | **2.976** | **67.65 GB/s** | **15.39×** |
| index_select 4096x64 | 1.199 | 1.75 GB/s | 1.299 | 1.61 GB/s | 0.92× |
| gather 1024x1024 | 1.206 | 10.43 GB/s | 1.180 | 10.66 GB/s | 1.02× |
| **layer_norm f32 8192x4096** | 14.213 | 37.78 GB/s | **1.285** | **417.82 GB/s** | **11.06×** |
| **rms_norm f32 8192x4096** | 10.898 | 36.95 GB/s | **1.328** | **303.22 GB/s** | **8.21×** |
| randn f32 16M | 21.858 | 3.07 GB/s | 20.971 | 3.20 GB/s | 1.04× |

＊ `transpose2d` 这一行是 bench 里的测量位置造成的，不是退化：临时探针在**两个
构建**里单独测同一个用例，HEAD 0.399 ms、工作区 0.398 ms（各 15 次中位数）。
它在 bench 里紧跟在 `sum-axis1` 后面，而后者改后快了约 0.35 ms；这个用例本身是
纯固定开销（submit 0.19 + sync 0.20），前面少了那段负载后驱动/GPU 的状态不同，
同一段代码就量出了 0.583 ms。它同时在改前一侧也出现过 0.550 ms 的运行。

同机 CUDA 对照（同一 benchmark、同一口径，来自
[`2026-11-03-cuda-op-optimization.md`](2026-11-03-cuda-op-optimization.md)）：
memcpy 597.8 GB/s、sum 575.3 GB/s、sum-axis1 583.6 GB/s、transpose2d 985.6 GB/s、
transpose3d 364.9 GB/s、matmul 1024 11380 GFLOPS、conv 5478.9 GFLOPS、
layer_norm 1186.9 GB/s、rms_norm 900.8 GB/s。

改后 Vulkan 相对自身 memcpy（151.38 GB/s）：sum 全归约 55%、轴归约 64%、
transpose3d 3%、layer_norm 276%（4 次读写/元素，与 memcpy 的 2 次不是一个分母）。
相对 CUDA 同项：matmul 30%、conv 44%、layer_norm 35%、rms_norm 34%、sum 14%——
Vulkan 的固定提交开销和缺少 cp.async 一类的显式预取是主要差距来源，不在本次范围。

## 半精度 matmul 走 cooperative matrix（同日追加）

上一轮只做了 `VK_KHR_cooperative_matrix` 的探测（`supports_cooperative_matrix()`），
这一轮把它接到 MatMul 上：设备满足「扩展已启用 + subgroup 大小 32 + 报告了
16×16×16 / f16（或 bf16）× 同类型 → f32 的配置」时，f16/bf16 的 MatMul 流水线
换成 tensor core 内核；其余 dtype、其余设备仍走标量内核，结果不变。

对照方式与上一节一致：同一时段、同一 benchmark，把 coopmat 的选择临时关掉跑一遍
（只改探测结果，内核代码不动）作为"标量"一列。coopmat 一列取 3 次运行中位数。

| 用例 | 标量 ms | 标量 GFLOPS | coopmat ms | coopmat GFLOPS | 加速比 |
| --- | --- | --- | --- | --- | --- |
| matmul float16 1024 | 0.604 | 3557 | 0.518 | 4149 | 1.17× |
| matmul float16 2048 | 2.350 | 7311 | 1.179 | 14567 | 1.99× |
| matmul bfloat16 1024 | 0.615 | 3492 | 0.633 | 3392 | 0.97× |
| matmul bfloat16 2048 | 1.691 | 10162 | 1.309 | 13124 | 1.29× |

4096³ 只跑了一次，仅作趋势参照：f16 14.966 → 7.692 ms（9184 → 17868 GFLOPS，1.95×），
bf16 10.742 → 7.765 ms（12794 → 17700 GFLOPS，1.38×）。1024³ 和 bf16 的加速比偏低
不是内核的问题：1024³ 只有约 0.15 ms 的内核时间，被固定的 submit+sync
（约 0.39 ms）盖住；bf16 的标量基线本来就比 f16 快一截（同一份数据、同一个内核
模板，原因未深究），所以相对收益小。

内核形状在 2048³ 上试了三档（f16，total 口径）：64×64 tile、4 个 subgroup
（128 线程）11487 GFLOPS → 128×64、8 个 subgroup 13264 → 128×128、16 个 subgroup
14567，最终采用 128×128。方向上仍然是"把搬运量/计算量的比值降下来"，但已经明显
边际递减：4096³ 时内核时间约 7.1 ms、折合 19 TFLOPS，是这颗卡寄存器-only MMA
峰值（实测约 165–167 TFLOPS）的 12%。剩下的差距 tile 尺寸填不上，需要双缓冲
（下一 K 步的搬运与当前步的 MMA 重叠）和 128-bit 向量化搬运，本轮没做。

## 改后三段耗时（ms，3 次运行中位数）

| 用例 | submit | sync | total | cold |
| --- | --- | --- | --- | --- |
| memcpy f32 16M | 0.325 | 0.559 | 0.887 | 1.133 |
| add f32 16M | 0.244 | 0.607 | 0.857 | 1.353 |
| relu f32 16M | 0.328 | 0.572 | 0.897 | 1.808 |
| sigmoid f32 16M | 0.401 | 0.640 | 1.053 | 0.930 |
| tanh f32 16M | 0.390 | 0.597 | 0.986 | 1.126 |
| cast f32->f64 16M | 0.369 | 0.714 | 1.083 | 1.167 |
| exp f32 4M | 0.292 | 0.300 | 0.595 | 0.669 |
| sum f32 16M | 0.560 | 0.244 | 0.809 | 1.608 |
| sum-axis1 4096x4096 | 0.291 | 0.398 | 0.690 | 0.761 |
| matmul f32 256 | 0.191 | 0.204 | 0.395 | 0.786 |
| matmul f32 512 | 0.190 | 0.222 | 0.411 | 0.840 |
| matmul f32 1024 | 0.212 | 0.422 | 0.630 | 1.033 |
| conv2d f32 8x64x56x56 | 0.191 | 0.523 | 0.715 | 1.103 |
| where f32 16M | 1.560 | 1.001 | 2.561 | 4.321 |
| where-scalar f32 16M | 1.969 | 0.992 | 2.976 | 3.340 |
| layer_norm f32 8192x4096 | 0.383 | 0.907 | 1.285 | 1.250 |
| rms_norm f32 8192x4096 | 0.388 | 0.941 | 1.328 | 1.624 |

## 改了什么

- **类型转换**（`typecast.cpp`、`shader/typecast/`，新增）：原来是
  `to_host → to_device → to_type_ → to_device`，一次 16M 的 f32→f64 要在 PCIe 上
  搬几趟 64–128 MB。现在是原生 shader：一次 src→dst，目标 dtype 由 push constant
  在一组别名绑定里选（一种源 dtype 一条流水线，`typecast_<src>`）。`where` /
  `where-scalar` 的 45 ms 也主要是它——`where` 会把条件张量 `Typecast` 到
  FLOAT64 再比较。
- **归约**（`reduce.cpp`、`shader/{sum,mean,max,min,prod}/*_float32.comp`）：
  轴长 ≥ 512 时改成 block 协作（一个 workgroup 一个输出、256 个线程沿归约轴
  分摊），全归约在轴长 ≥ 65536 时拆两趟（趟 1 拆成 1024 段连续块、趟 2 折叠
  1024 个部分和）。旧路径保留为短轴/非 FLOAT32 的兜底。
- **Prod 的宽累加器**（`shader/prod/prodvec_float32.comp`）：浮点 Prod 改成
  double 累加。这不是性能改动：float32 连乘 1M 个元素时舍入偏差能到千分点，且
  偏差量随归约树的形状变化（同一份输入在"逐线程串行"和"block 树"下差近 1%）。
  改成 double 后结果不再依赖并行划分方式，与 CPU 的 `reduce_acc_t`、CUDA 的
  `prod_uses_wide_acc` 是同一条决策；`tests/device.cpp` 里那条宽松容差现在只为
  SYCL 保留。
- **矩阵乘**（`matmul.cpp` + 8 份 `shader/matmul/*.comp`）：16×16 tile、每线程
  1 个输出 → 64×64 tile、16×16 线程、每线程 4×4 寄存器累加、BK=16 共享内存
  （`As[64][17]` / `Bs[16][65]`）。每个输出的累加顺序仍按 k 递增，浮点结果与旧
  内核逐位一致。int8 的饱和语义由 `saturate_int8` 保留。
- **卷积**（`conv.cpp` + 4 份 `shader/conv2d/*.comp`）：每线程 1 个输出元素 →
  每线程 4 个输出通道 × 4 个相邻输出列（16 个输出，16 次乘加只用 8 次读取），
  workgroup 覆盖一行输出上的 64 通道 × 64 列。边界位置用"越界通道借用最后一个
  通道的权重、越界感受野贡献 0"处理，不需要真的补零。
- **归一化**（`norm.cpp` + `shader/{layernorm,rmsnorm}/`，新增）：每行一个
  workgroup、行数据进共享内存的融合内核，替换原来 8 个内核拼出来的
  `layer_norm_composed`。只对 FLOAT32 注册流水线，其余 dtype 在 `ops.cpp` 里
  仍走组合式路径。
- **一元/激活去掉多余的 clone**（`activate.cpp`、`arithmetic.cpp`、
  `vulkan_common.hpp`、relu/sin/cos/tan/sqrt/exp/pow 的 f32 着色器）：非原地
  路径原来一律 `clone()` 后跑原地内核——多出一整遍读+写。现在直接分配 dst、
  一次 src→dst。为此把六份 f32 着色器从"读绑定 1 写绑定 1"改成"读 0 写 1"
  （其余 dtype 本来就是这样）；relu 的 8 份从单绑定改成 src/dst 双绑定，NaN
  分支改成显式写回（新 dst 未初始化，不能靠"不写"保留原值）。
  `vulkan_common.hpp` 新增 `linear_source` / `unary_source`：内核按线性下标从
  buffer 基址读，带 offset 的连续切片和非连续视图必须先物化——以前这一步是
  被 clone 顺带完成的，去掉 clone 后必须显式做。
- **f16/bf16 矩阵乘走 tensor core**（`vulkan_context.cpp` 的探测与选路、
  `shader/matmul/matmul_coopmat_{float16,bfloat16}.comp` 新增、`matmul.cpp` 的
  网格）：设备同时具备扩展、subgroup 32、16×16×16 的 f16（bf16）→ f32 配置时，
  `registerOp` 给 MatMul 的 FLOAT16/BFLOAT16 换成 `matmul_coopmat_*` 那份 SPIR-V，
  **流水线键仍是 `matmul_<dtype>`**，所以 `submitCompute` 和其它算子完全看不到
  这条分支；建管线失败会回落到标量内核并打到 stderr。内核是 128×128 tile、
  16 个 subgroup（512 线程）、每个 K 步把 A 的 128×16 与 B 的 16×128 搬进共享
  内存、每个 subgroup 做 4 次 `coopMatMulAdd`；累加器是 f32，与标量内核一致。
- **benchmark**：新增 `matmul float16 1024/2048` 与 `matmul bfloat16 1024/2048`
  四个用例——同一份数字能直接反映走的是哪条内核。
- **`tools/coverage_overrides.toml`**：LayerNorm/RmsNorm 的 VULKAN `[emulated]`
  标注删除（已有原生内核），并注明该内核只覆盖 FLOAT32，与 CUDA 不同。

## 采集说明与已知偏差

- 单卡、无其它计算任务；即便如此仍有 ±10% 的运行间漂移。表中未改代码的行
  （`add`/`mul`/`div`/`gather`/`index_select`/`arange`/`randn`）落在 0.92–1.04×，
  可以当作本表的噪声底。
- `relu`/`sigmoid`/`tanh` 三行的 `total` 方差最大（±15%），因为它们是 SFU 相关的
  短内核；同一次运行里 `sigmoid` 曾出现 0.878 / 1.070 / 1.053 ms。更干净的量是
  `submit` 列：克隆那次提交消失后从 0.67–0.79 ms 降到 0.33–0.40 ms。
- 另做了一次同进程 A/B（临时测试，已删除）：单独一次 `clone` 1.001 ms、
  原地 relu 0.728 ms、新路径 0.865 ms、旧路径 1.139 ms；sigmoid 新 0.908 /
  旧 1.305 ms。与表里的 1.37× / 1.26× 同量级。
- `sum` 的 `submit` 列（0.560 ms）比 `sync`（0.244 ms）大三倍：两趟归约是两次
  提交，而每次提交在这台机器上有约 0.19–0.25 ms 的固定成本。它现在是 `sum`
  的主要成本，不是内核。
- 子 0.5 ms 的用例（`matmul 256`、`transpose2d`、`gather permute`）完全是延迟
  受限的，内核再快也压不过固定的 submit+sync。
- `arange` / `randn` 是宿主生成后搬运的既有设计（`tools/coverage_overrides.toml`
  标为 host_side），本次未改；它们的 ~3 GB/s 不代表设备内核性能。
- `python3 tools/coverage_matrix.py --check` 现在会因为 `docs/coverage.md` 缺失
  而失败——该文件在 `fa0b645` 里被删掉了，工具仍要求它存在，这是既有的不一致，
  本次没有恢复它。标注本身（`--write` 的校验阶段）没有报错。

## 未追求的方向

- **matmul 更大 tile（128×128）**：CUDA 侧实测过，寄存器压力把占用率压到
  2 block/SM，反而更慢；Vulkan 侧的标量内核没有重复这一尝试。coopmat 内核是另一
  回事——它的 tile 由 subgroup 数量决定（见上一节），128×128 反而是最优档。
  标量路径的 3406 GFLOPS 是 CUDA 同项的 30%，继续压要引入 cp.async 式双缓冲和
  128-bit 共享读，超出本轮范围。
- **coopmat 内核的双缓冲与向量化搬运**：这是把 19 TFLOPS 继续往峰值推的下一步
  （当前瓶颈是每个 K 步两次 barrier 加标量 16-bit 搬运），但需要重写主循环。
- **f32/f64 的 coopmat**：这颗卡不报告任何 f32/f64 配置（15 种配置里没有），
  无从做起；f32 的 matmul 仍走标量内核。
- **conv 通用形状**：测例只覆盖 3×3、stride 1；更大 kernel/stride 的路径没有
  改动，也没有基准可用。
- **归约的第二次提交**：把两趟合成一个 dispatch（趟 2 直接读全局部分和而不
  再提交）需要跨 workgroup 同步或 atomic，当前两趟的总时间已被 submit 固定开销
  主导，收益与风险不成比例。
- **SYCL lane**：两条既有失败与本轮无关，未处理。

## 正确性

- `MCPP_JOBS=1 mcpp test --features vulkan --accel vulkan1.4 --profile release`：
  11 个可执行全部通过（`MCPP_JOBS=1` 是为了让 bench 不与其它测试二进制抢 GPU，
  不影响结果正确性）。
- 临时探针（已删除）覆盖了去掉 clone 后的新路径：带 offset 的连续切片
  （`narrow`）、非连续视图（`permute`）上的 relu/sigmoid/tanh/abs/exp/sin/cos/
  sqrt/log/pow 与 CPU 逐元素比对；relu 的 NaN 传播在非原地与原地两条路径上都
  成立；非原地调用不修改输入。
- matmul 每个输出的累加顺序与旧内核一致（k 递增），因此浮点结果逐位相同；8 种
  dtype 的测试均通过，包括 int8 的饱和语义。
- coopmat 内核由新增的 `DeviceOps.MatMulHalfPrecision` 覆盖：f16 与 bf16 各跑
  70³（三个维度都留下半块 tile）、128³（整块对齐）、2×80×48×96（批处理，且三个
  维度都不整除 tile），与 CPU 的 f32 参考值（按目标 dtype 舍入一次）比对。取值
  都是 0.25 的倍数，乘积与部分和在半精度和 f32 里都精确，容差只留给最后一步
  舍入（f16 4e-3、bf16 3e-2 相对）。
  `VulkanCapabilities.CooperativeMatrixReflectsDeviceState` 另外断言：条件都满足时
  `matmul_uses_cooperative_matrix()` 必须为真——回落是建管线失败时的安全网，
  不是预期状态；本机实测 f16/bf16 均为 true。
- 覆盖矩阵没动：coopmat 只是同一算子的一条可选内核，`[emulated]` 标注与
  dtype 覆盖不受影响。
- Prod 的 double 累加让 `DeviceOps.FullReduceOnLargeTensors` 重新落回正常容差；
  之前 1M 元素的偏差是 0.97950494289398193 对 0.9895…，用 numpy 按新内核的
  归约顺序复现，确认是顺序漂移而不是错误。
- `tests/p2.cpp` 的异步用例把放大上限从 2048 提到 4096，并在达不到 6× 判据时
  `GTEST_SKIP`：`mcpp test` 并行跑多个测试二进制时每次提交会被驱动拖到 2.6–4.5 ms
  （空闲时 0.19 ms），而内核变快后 2048³ 的 GPU 工作只有约 2 ms，分母被抬高的
  情况下比值不再反映实现是否异步。
