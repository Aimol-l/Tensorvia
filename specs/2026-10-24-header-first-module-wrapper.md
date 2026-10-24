# Specification: 头文件事实源 + mcpp 薄模块包装（Header-first with module wrapper）

## Metadata
- **Version**: 1.0.0
- **Status**: Draft
- **Author**: agent + aimol
- **Created**: 2026-10-24
- **Last Updated**: 2026-10-24

## Overview

Tensorvia 0.3.0 当前以 C++23 命名模块为单一事实源，CMake 路径与 CUDA lane
依赖 `tools/gen_header_mode.py` 在构建期把模块源码翻译成头文件树（77 个文
件）。nvcc 13.3 不支持模块这一事实决定了头文件形态必须存在——与其让它作为
生成产物存在，不如提升为事实源，模块退化为投影。

本 spec 将库主体改为 0.2.x 式的头文件形态（`tensorvia.hpp` + 分区 `.hpp`
+ `.cpp` 实现），mcpp 侧只保留一个薄包装模块 `tensorvia.ixx`（GMF
include 总头文件 + 手写 export 清单），从而删除整套“两种形态翻译/守卫”设
施。使用方零变化：mcpp 用户照旧 `import tensorvia;`，CMake 用户照旧
`#include <tensorvia.hpp>`。0.3.0 尚未发布，无迁移成本。

## Requirements

### Functional Requirements

- FR-1: `src/interface/` 的模块分区全部转为 `.hpp`（`export module X;` →
  include guard；`import :part;` → `#include`；剥离 `export`），
  `tensorvia.hpp` 为总头文件（分区 `.hpp` 的 include 列表）；所有 `.cpp`
  实现单元（含 `src/backend/*/host/`）改为 include 头文件。
- FR-2: `src/interface/tensorvia.ixx` 为薄包装模块：仅 GMF
  `#include "tensorvia.hpp"` + 手写 export 清单（预计 60–100 行，无逻
  辑）；`[lib] path` 保持指向它，`mcpp.toml` 不动。
- FR-3: 导出清单校验工具：改造 `tools/gen_facade.py`，从头文件扫公开
  符号，与 `tensorvia.ixx` 的 export 块对比；漏导出与幽灵导出均报错；
  可在 CI 与 `mcpp test` 中运行。
- FR-4: `build.mcpp` 合并为单 lane：删除 module/CUDA lane 分叉与
  `gen_header_mode.py` 调用；源集 = 按 feature 门控选择的 `.cpp`
  （CPU/Vulkan/SYCL host 驱动的条件编译逻辑保留）+ `tensorvia.ixx`；
  `.cu`/`.sycl` 设备规则、Vulkan glslang action 保持现状。
- FR-5: `CMakeLists.txt` 删除头文件生成段（生成调用、输出声明、缓存清
  理、`tensorvia_header_gen` target），直接编译 `src/`；install 改为直
  接安装 `src/interface/*.hpp` 与 `tensorvia.hpp`。
- FR-6: 删除 `src/interface/header_core.ixx`、`tools/gen_header_mode.py`、
  `tools/facade_exports.def`（清单宿主迁入 `tensorvia.ixx`）。
- FR-7: 使用方 API 面零变化：`import tensorvia;` 与
  `#include <tensorvia.hpp>` 两种消费方式均可用，公开符号集一致。

### Non-functional Requirements

- NFR-1: 行为不变量——现有测试套件（CPU 全量用例 + vulkan/cuda/sycl
  smoke）在 mcpp 与 CMake 两条构建路径全部通过，作为转换正确性的证明。
- NFR-2: 转换可回退——全部改动落在单一分支/提交内，测试不过即整体回
  退，不留半模块半头文件的中间态。
- NFR-3: 头文件自洽——每个 `.hpp` 有 include guard，可独立 include，
  无循环包含；总头文件包含全部公开分区。
- NFR-4: 包装模块可读——`tensorvia.ixx` 不出现任何逻辑代码，export 清
  单与公开 API 面一一对应，可人工 review。

## Test Steps

1. **mcpp CPU 路径**：`mcpp build` + `mcpp run` 冒烟（CPU 数值校验，
   如 `ops::Sum(ones+fill)`）。
2. **mcpp 后端 lane**：`mcpp build --features vulkan`；
   `mcpp build --features cuda --accel "cuda13.3+{sm_89}"`；
   `mcpp build --features sycl --accel sycl`——三条均构建通过。
3. **mcpp 测试**：`mcpp test`（CPU 全量）；`mcpp test vulkan_smoke
   --features vulkan`；`mcpp test cuda_smoke --features cuda --accel
   "cuda13.3+{sm_89}"`；`mcpp test sycl_smoke --features sycl --accel
   sycl`。
4. **导出校验**：校验工具 `--check` 通过；人为从 `tensorvia.ixx` 删
   ／造一条 export，工具分别报“漏导出”与“幽灵导出”。
5. **CMake 路径**：`cmake -B build -DBUILD_TEST=ON` + `ctest`；启用
   Vulkan/CUDA/SYCL 的组合各配置一次。
6. **消费方冒烟**：外部项目以 path 依赖 `import tensorvia;` 默认 CPU
   构建并运行（数值正确）；以 `#include <tensorvia.hpp>` 的 CMake 工
   程链接安装产物。
7. **边界**：nvcc lane 下 `.cu` 直接 include 头文件编译通过；`mcpp
   pack tensorvia --features sycl`（无 accel）仍产出 CPU 包且不被诊断误
   伤。

## Acceptance Criteria

- 上述 7 步全部通过。
- 删除清单（FR-6）中的文件与 `build.mcpp` 的 lane 分叉、
  `CMakeLists.txt` 的生成段确实不存在。
- `tensorvia.ixx` ≤ 100 行且不含逻辑代码。
- 消费项目（mcpp 与 CMake 两种）**不改一行代码**完成构建与运行。
- 校验工具接入 CI 或 `mcpp test`，漏导出可在构建期暴露。

## Change Log

| Date       | Version | Description                       | Author     |
|------------|---------|-----------------------------------|------------|
| 2026-10-24 | 1.0.0   | Initial draft                     | agent      |
