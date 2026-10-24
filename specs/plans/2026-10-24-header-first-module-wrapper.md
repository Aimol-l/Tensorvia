# Header-First + Module Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Tensorvia 库主体从头文件生成模式反转为一等头文件源码，mcpp 侧只留一个薄模块包装，删除整套"两种形态翻译/守卫"设施。

**Architecture:** `src/interface/` 的 12 个 `.ixx` 分区转为 `.hpp`（头文件为唯一事实源）；`src/interface/tensorvia.ixx` 重写为薄包装（GMF include `tensorvia.hpp` + 由 `gen_facade.py` 从 `facade_exports.def` 生成的 export 块）；`build.mcpp` 合并为单 lane；`CMakeLists.txt` 删除头文件生成段。使用方 API 零变化。

**Tech Stack:** C++23 命名模块（仅包装层）、mcpp 2026.10.17.1、CMake ≥ 3.28、Python 3（校验工具/转换脚本）、GCC 16.1.0、GoogleTest

**Spec:** `specs/2026-10-24-header-first-module-wrapper.md`

## Global Constraints

- 使用方 API 面零变化：`import tensorvia;` 与 `#include <tensorvia.hpp>` 均可用，公开符号集一致。
- `[lib] path = "src/interface/tensorvia.ixx"` 不变，`mcpp.toml` 不动。
- 测试不变量：`mcpp test`（CPU 全量 + 三后端 smoke）与 `ctest` 全绿才算任务完成。
- 不留半模块半头文件的中间态：全部改动在一个分支内，测试不过整体回退。
- 每个 `.hpp` 有 include guard，可独立 include；总头文件 `tensorvia.hpp` 包含全部公开分区。
- `tensorvia.ixx` ≤ 100 行，只含 GMF include 与 export 块，无逻辑代码。

## Review Focus

1. **Clang/LLVM 工具链下的包装模块**（本机只验证过 GCC 16；Clang 对模块内 `using` 声明 internal-linkage 重载更严格）——Task 6 用 `llvm@22.1.8` 工具链构建一次消费项目。
2. **校验工具漏报**（正则未覆盖的声明形态：运算符重载、模板函数、`static` 自由函数、嵌套命名空间）——Task 3 对每种形态造反向用例。
3. **install 出的头文件树自洽**（缺文件、路径错误、包含关系断裂）——Task 4 在 install 后从安装树消费一次。
4. **全量头文件的编译开销**（转换后每个 TU 都 include `tensorvia.hpp`）——Task 2 记录 `mcpp build` CPU 基线时间并与转换前对比（增幅 >50% 需在验收中记录原因）。
5. **nvcc lane 中 `.cu` include 头文件**（头文件残留模块语法/宏会让 nvcc 前端炸）——Task 2 的 CUDA lane 构建覆盖，`.cu` 必须零 warning 通过。

---

### Task 1: 源码转换（`.ixx` → `.hpp` + 重写 `tensorvia.ixx`）

**Files:**
- Modify: `src/interface/*.ixx` → 新建同名 `.hpp`（11 个分区；`header_core.ixx` 删除）
- Modify: `src/interface/tensorvia.ixx`（重写为薄包装，约 60-100 行）
- Modify: `src/backend/{cpu,vulkan,sycl}*/**/*.cpp`、`src/core/*.cpp`、`src/init*.cpp`（`import tensorvia:x;` → `#include "x.hpp"`）
- Create: `/tmp/opencode/ixx2hpp.py`（一次性转换脚本，不入库）

**Interfaces:**
- Consumes: 现有 12 个 `.ixx` 的内容；`tools/facade_exports.def` 的清单（Task 3 工具化前的 export 块来源）
- Produces: `src/interface/*.hpp`（头文件事实源）；`src/interface/tensorvia.ixx`（`module;` + `#include "tensorvia.hpp"` + `export module tensorvia;` + export 块）

- [ ] **Step 1: 建立转换前基线**

```bash
cd /home/aimol/Documents/C++/Workspace/MCPP/Tensorvia
mcpp build 2>&1 | tail -2          # 记录基线：Finished 时间
mcpp test 2>&1 | grep "test result"  # 记录基线：CPU 全量通过数
```

- [ ] **Step 2: 写一次性转换脚本**

创建 `/tmp/opencode/ixx2hpp.py`，规则（逐条对应）：

```python
# 对 src/interface/*.ixx（tensorvia.ixx 与 header_core.ixx 除外）：
# 1. 删除 "export module tensorvia:<part>;" 行
# 2. 删除全局模块片段标记 "module;" 行及其后的 #include 行（保留 #include 内容，
#    去掉 "module;" 标记——注意：GMF 里的 #include 要保留在文件头部）
# 3. "import tensorvia:<part>;" / "import :<part>;" → '#include "<part>.hpp"'
# 4. 剥离所有行首 "export " 前缀（export class → class 等）
# 5. 文件头部插入 "#pragma once"
# 6. 原 "export module tensorvia;<part> 分区内声明" 保持原样（去 export）
# 输出为同名 .hpp
```

- [ ] **Step 3: 运行脚本并逐文件 review**

```bash
python3 /tmp/opencode/ixx2hpp.py
ls src/interface/*.hpp            # 应得 11 个（不含 tensorvia.hpp 见 Step 6）
```

人工 review 清单（逐个 `head -40` + `grep -n "export\|module;\|import "`）：

```
version types context tensor factory cpu_context cpu_tensor cpu_ops ops core
backend/vulkan/vulkan_ops.ixx → backend/vulkan/vulkan_ops.hpp
backend/cuda/cuda_ops.ixx      → backend/cuda/cuda_ops.hpp
backend/sycl/sycl_ops.ixx      → backend/sycl/sycl_ops.hpp
```

每个文件必须满足：无 `export`、无 `module;`、无 `import `、有 `#pragma once`。

- [ ] **Step 4: 转换 backend host `.cpp` 的 import**

```bash
grep -rn "^import tensorvia" src/backend src/core src/*.cpp
```

每个命中处 `import tensorvia:<part>;` → `#include "<part>.hpp"`（相对 `src/interface/`，用 `#include "interface/<part>.hpp"` 或补 `-I src`——与现有 `include_dirs = ["include"]` 协调，凡 `src/interface/*.hpp` 的 include 一律写 `#include "interface/<part>.hpp"`，并在 `mcpp.toml [build] include_dirs` 追加 `"src"`）。

- [ ] **Step 5: 写总头文件 `src/interface/tensorvia.hpp`**

```cpp
#pragma once
// 总头文件：公开 API 面。分区头文件的 include 列表。
#include "interface/version.hpp"
#include "interface/types.hpp"
#include "interface/context.hpp"
#include "interface/tensor.hpp"
#include "interface/factory.hpp"
#include "interface/cpu_context.hpp"
#include "interface/cpu_tensor.hpp"
#include "interface/cpu_ops.hpp"
#include "interface/ops.hpp"
#include "interface/core.hpp"
```

（路径以 `src/` 为 include 根；若某分区头之间有多层依赖，按依赖序排列，`ops.hpp` 在最后。）

- [ ] **Step 6: 重写 `src/interface/tensorvia.ixx` 为薄包装**

```cpp
module;
#include "interface/tensorvia.hpp"

export module tensorvia;

// 公开 API 的 export 清单——初版从 tools/facade_exports.def 手工转写，
// Task 3 起由 tools/gen_facade.py 生成/校验。不得出现任何逻辑代码。
export using ::tensorvia::Tensor;
// ……（.def [global] 节逐条：class/using/enum/常量）
export namespace ops {
    using ::ops::Add;
    // ……（.def [ops] 节逐条：每个重载一条）
}
```

- [ ] **Step 7: 删除 `header_core.ixx`，验证无残留**

```bash
rm src/interface/header_core.ixx
grep -rn "header_core" src/ tests/ tools/ CMakeLists.txt build.mcpp   # 应只剩注释/无
grep -rn "^import \|^export module" src/ | grep -v "tensorvia.ixx"    # 应为空
```

- [ ] **Step 8: 提交**

```bash
git add -A src/ && git commit -m "refactor: header-first sources with thin module wrapper"
```

---

### Task 2: `build.mcpp` 合并为单 lane

**Files:**
- Modify: `build.mcpp`（删除 `add_module_lane_sources` / `add_cuda_lane_sources` / `gen_header_mode.py` 调用；新的源选择函数）
- Modify: `mcpp.toml`（`include_dirs` 追加 `"src"`，供 `interface/*.hpp` 解析）

**Interfaces:**
- Consumes: Task 1 的 `src/interface/*.hpp` 与 `src/interface/tensorvia.ixx`
- Produces: 单 lane 构建：无条件编译 `src/core/*.cpp` + `src/init.cpp` + `src/backend/cpu/**/*.cpp` + `tensorvia.ixx`；`want_vulkan` 加 `src/init_vulkan.cpp` + `src/backend/vulkan/*.cpp`；`want_sycl` 加 `src/init_sycl.cpp` + `src/backend/sycl/host/*.cpp`；`want_cuda` 加 `src/init_cuda.cpp`（`.cu` 由 `mcpp.toml` 的 accel glob + rules-cuda 编译，不变）

- [ ] **Step 1: 确认红**

```bash
mcpp build 2>&1 | tail -5    # 预期失败：add_module_lane_sources 列的 .ixx 已不存在
```

- [ ] **Step 2: 重写源选择逻辑**

将 `add_module_lane_sources` + `add_cuda_lane_sources` 替换为单一函数（保留 `files_with_extension`、`join`、`glslang_program`、`cuda_line`、诊断块不动）：

```cpp
// Every host translation unit of the library, gated by feature exactly as
// before. The headers are the sources of truth; only the wrapper module
// tensorvia.ixx and the .cpp units compile.
bool add_library_sources(const std::string& root, bool want_vulkan,
                         bool want_sycl, bool want_cuda) {
    mcpp::source(join(root, "src/interface/tensorvia.ixx").c_str());
    for (const auto& path : files_with_extension(join(root, "src/core"), ".cpp"))
        mcpp::source(path.c_str());
    mcpp::source(join(root, "src/init.cpp").c_str());
    for (const auto& path : files_with_extension(join(root, "src/backend/cpu"), ".cpp"))
        mcpp::source(path.c_str());
    if (want_vulkan) {
        mcpp::source(join(root, "src/init_vulkan.cpp").c_str());
        for (const auto& path : files_with_extension(join(root, "src/backend/vulkan"), ".cpp"))
            mcpp::source(path.c_str());
    }
    if (want_sycl) {
        mcpp::source(join(root, "src/init_sycl.cpp").c_str());
        for (const auto& path : files_with_extension(join(root, "src/backend/sycl/host"), ".cpp"))
            mcpp::source(path.c_str());
        // sycl_island.h sits beside host/ and device/; both include it by name.
        mcpp::include_dir(join(root, "src/backend/sycl").c_str());
    }
    if (want_cuda) {
        // Host driver; the .cu device units come from [build] sources'
        // accel-constrained glob, compiled by rules-cuda.
        mcpp::source(join(root, "src/init_cuda.cpp").c_str());
    }
    return true;
}
```

`main()` 中删除 `if (cuda_lane) { ... gen_header_mode.py ... }` 整块与 `add_cuda_lane_sources` 调用，统一为 `add_library_sources(root, want_vulkan, want_sycl, want_cuda);`。保留：accel/feature 诊断块。`rules::cuda` 调用的 `opt.includes` 由 `{out + "/gen/header", ...}` 改为 `{root + "/src", root + "/src/backend/cuda", root + "/include"}`（`.cu` 的 `#include "interface/tensorvia.hpp"` 等以 `src` 为根解析；同步修正 `.cu` 内 include 写法使其与 host 一致）。

- [ ] **Step 3: `mcpp.toml` include 路径**

`[build] include_dirs = ["include", "src"]`（`interface/*.hpp` 以 `src` 为根解析）。

- [ ] **Step 4: CPU 绿 + 记录构建时间（Review Focus #4）**

```bash
time mcpp build          # 与 Step 1 基线对比，记录增幅
mcpp run                 # 数值冒烟：cpu sum = 18 / softmax sum = 1
mcpp test 2>&1 | grep "test result"   # CPU 全量通过，数量与基线一致
```

- [ ] **Step 5: 三后端 lane 绿（Review Focus #5）**

```bash
mcpp build --features vulkan
mcpp build --features cuda --accel "cuda13.3+{sm_89}"
mcpp build --features sycl --accel sycl
mcpp test cuda_smoke --features cuda --accel "cuda13.3+{sm_89}" 2>&1 | grep "test result"
```

`.cu` 编译输出不得有 warning（`grep -i warning` 为空）。

- [ ] **Step 6: 提交**

```bash
git add -A build.mcpp mcpp.toml && git commit -m "build: single lane — headers are the sources of truth"
```

---

### Task 3: `gen_facade.py` 改造为 `tensorvia.ixx` 的生成/校验器

**Files:**
- Modify: `tools/gen_facade.py`（生成/校验目标 `src/interface/header_core.ixx` → `src/interface/tensorvia.ixx`）
- Modify: `tools/facade_exports.def`（docstring 注释更新；清单内容不变）
- Test: `tests/facade_api.cpp`（不改——它已经是"每个名字经 `import tensorvia;` 编译"的守卫）

**Interfaces:**
- Consumes: `tools/facade_exports.def`（`[global]` / `[ops]` 等分节的手写清单）；Task 1 的 `tensorvia.ixx`
- Produces: `python3 tools/gen_facade.py --write`（从 `.def` 重建 `tensorvia.ixx` 的 export 块）；`--check`（不一致即非零退出）

- [ ] **Step 1: 读现有实现**

```bash
cat tools/gen_facade.py    # 理解 --write/--check 与 .def 分节的现有解析
```

- [ ] **Step 2: 确认红**

```bash
python3 tools/gen_facade.py --check 2>&1 | head -5   # 预期失败：目标文件形态已变
```

- [ ] **Step 3: 改造输出目标与生成形态**

生成内容从 `export module tensorvia:core;` 门面改为根模块包装的 export 块：

```
module;
#include "interface/tensorvia.hpp"

export module tensorvia;

<由 .def 生成的 export using / export namespace 块>
```

即 `--write` 重写 `tensorvia.ixx` 时保留头部四行与尾部结构，只替换 export 块；`--check` 对比磁盘上的 export 块与 `.def` 推导结果。

- [ ] **Step 4: 绿**

```bash
python3 tools/gen_facade.py --check && echo CHECK-OK
mcpp build && mcpp test facade_api 2>&1 | grep "test result"   # 每个名字编译通过
```

- [ ] **Step 5: 反向用例（Review Focus #2）**

对每种声明形态各造一次“清单缺失”，`--check` 必须非零退出并点名：

```bash
# ① 漏导出：从 .def 删掉 ops 节的一行 → --check 报 "missing"
# ② 幽灵导出：向 .def 加一个不存在的名字 → --check 报 "not declared"
# ③ 运算符重载：确认 .def 中 operator+ 等条目的解析（.def 现有条目里若有
#    operator 形态必须被 --check 正确处理，不被正则漏掉）
# ④ 模板函数：若公开面有模板（如 ops 的重载经 using 导出），向 .def 加一条
#    模板名 → --write 后 import 侧编译通过
```

每条用例执行后恢复 `.def` 原状。

- [ ] **Step 6: 接入 `mcpp test`**

在 `build.mcpp` 或 CI 脚本中加一步 `python3 tools/gen_facade.py --check`（本任务直接接入 `tests/` 之外的方式：在 `build.mcpp` 的 `run()` 列表追加一次调用，失败即构建失败）。

- [ ] **Step 7: 提交**

```bash
git add -A tools/ build.mcpp && git commit -m "tools: gen_facade generates/checks the wrapper's export block"
```

---

### Task 4: `CMakeLists.txt` 删除头文件生成段

**Files:**
- Modify: `CMakeLists.txt`（删除 69–144 行区域的生成逻辑、`tvv_generated_paths`、`TVV_HEADER_DIR` 映射；install 改直装 `src/interface/*.hpp`）

**Interfaces:**
- Consumes: Task 1 的 `src/interface/*.hpp`
- Produces: 无生成的 CMake 构建：源直接用 `src/` 路径；install 树 = `include/tensorvia.hpp` + `include/tensorvia/*.hpp`

- [ ] **Step 1: 确认红**

```bash
cmake -B build-cmake -DBUILD_TEST=ON 2>&1 | tail -5   # 预期失败：生成器输入（.ixx 模块树）已不存在
```

- [ ] **Step 2: 删除生成段**

删除：`find_package(Python3 ...)`、`TVV_GEN_DIR/TVV_HEADER_DIR`、`--print-outputs` 探测、`add_custom_command`/`tensorvia_header_gen`、`TVV_HEADER_OUTPUTS_CACHE` 清理、`tvv_generated_paths()` 函数。所有 `${TVV_HEADER_DIR}/...` 路径替换为 `${PROJECT_SOURCE_DIR}/src/interface/...`（头文件）与 `${PROJECT_SOURCE_DIR}/src/...`（`.cpp`）。

- [ ] **Step 3: install 段改写**

```cmake
install(FILES "${PROJECT_SOURCE_DIR}/src/interface/tensorvia.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
file(GLOB TVV_PARTITION_HEADERS "${PROJECT_SOURCE_DIR}/src/interface/*.hpp")
list(REMOVE_ITEM TVV_PARTITION_HEADERS
     "${PROJECT_SOURCE_DIR}/src/interface/tensorvia.hpp")
install(FILES ${TVV_PARTITION_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensorvia)
```

- [ ] **Step 4: CPU 绿**

```bash
cmake -B build-cmake -DBUILD_TEST=ON && cmake --build build-cmake -j6
ctest --test-dir build-cmake --output-on-failure 2>&1 | tail -3
```

- [ ] **Step 5: install 树消费验证（Review Focus #3）**

```bash
cmake --install build-cmake --prefix /tmp/opencode/tv-install
ls /tmp/opencode/tv-install/include/ /tmp/opencode/tv-install/include/tensorvia/ | head -20
# 用一个最小 CMake 工程 find_package(Tensorvia CONFIG REQUIRED) 链接并编译运行
```

- [ ] **Step 6: 后端组合各配置一次**

```bash
cmake -B build-vk -DBACKEND_VULKAN=ON && cmake --build build-vk -j6
cmake -B build-cuda -DBACKEND_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build-cuda -j6
```

- [ ] **Step 7: 提交**

```bash
git add -A CMakeLists.txt && git commit -m "cmake: compile the header sources directly, drop the generator"
```

---

### Task 5: 删除剩余设施 + README 更新

**Files:**
- Delete: `tools/gen_header_mode.py`
- Modify: `README.md`（构建两卡片的措辞：不再有"构建时生成头文件模式"）

- [ ] **Step 1: 删除并验证无引用**

```bash
rm tools/gen_header_mode.py
grep -rn "gen_header_mode" . --include="*.py" --include="*.md" --include="*.txt" --include="*.mcpp" --include="*.cpp" --include="*.toml" 2>/dev/null   # 应为空
```

- [ ] **Step 2: README 措辞更新**

- 开头特性区"模块源码，双路径构建"改为"头文件源码 + mcpp 薄模块包装"
- mcpp 卡片删除对 `header_core.ixx` / 两条 lane 表的引用；两条构建 lane 表整体删除（单 lane 了）
- CMake 卡片删除"构建时把 `.ixx` 生成为头文件模式"的说法
- 安装与消费段落："安装包只包含头文件与共享库"保持不变（现在本来就是）

- [ ] **Step 3: 全量构建回归**

```bash
mcpp build && mcpp test 2>&1 | grep "test result"
```

- [ ] **Step 4: 提交**

```bash
git add -A tools/ README.md && git commit -m "chore: remove the header-mode generator, update README"
```

---

### Task 6: 全量验收（spec Test Steps）

**Files:**
- Modify: `specs/2026-10-24-header-first-module-wrapper.md`（Status → Realized）

- [ ] **Step 1: spec Test Steps 1–7 全跑**

```bash
# 1. mcpp CPU：mcpp build && mcpp run
# 2. 三 lane：mcpp build --features vulkan / --features cuda --accel cuda13.3+{sm_89} / --features sycl --accel sycl
# 3. mcpp test 全量 + 三 smoke（--features 对应）
# 4. python3 tools/gen_facade.py --check
# 5. cmake -B build-cmake -DBUILD_TEST=ON && ctest（CPU + Vulkan + CUDA 组合）
# 6. 消费方冒烟（Step 2）
# 7. 边界：mcpp pack tensorvia --features sycl（无 accel）出 CPU 包且不被诊断误伤
```

- [ ] **Step 2: 消费方零改动冒烟**

```bash
cd /tmp/opencode/tvconsumer
# main.cpp 与 mcpp.toml 一行不改：
mcpp build                          # 默认 CPU
mcpp build --accel "cuda13.3+{sm_89}"   # CUDA（features=["cuda"] 已在 toml）
mcpp run --accel "cuda13.3+{sm_89}"     # 数值：1024 / 32768 / 32
```

- [ ] **Step 3: LLVM 工具链消费（Review Focus #1）**

```bash
cd /tmp/opencode/tvconsumer
mcpp build --toolchain llvm@22.1.8    # Clang 下 import 包装模块
```

失败则按 SKILL.md gotcha（internal-linkage using）修复 export 清单后重跑。

- [ ] **Step 4: 构建时间记录（Review Focus #4）**

对比 Task 2 Step 1 基线与当前 `mcpp build` CPU 时间，写入本计划末尾的验收记录。

- [ ] **Step 5: 更新 spec 状态并提交**

```bash
# spec 的 Status: Draft → Realized；Test Steps 每项打勾
git add -A specs/ && git commit -m "spec: header-first conversion realized"
```

---

## 验收记录（执行时填写）

| 项 | 基线（转换前） | 转换后 | 备注 |
|---|---|---|---|
| `mcpp build` CPU 时间 | | | |
| `mcpp test` CPU 通过数 | | | |
| LLVM 消费 | — | | |
| `gen_facade.py --check` | — | | |
