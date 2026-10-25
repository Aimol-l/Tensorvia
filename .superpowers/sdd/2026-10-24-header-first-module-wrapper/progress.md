# SDD ledger — plan: specs/plans/2026-10-24-header-first-module-wrapper.md

Pre-flight: shared-interface rows — T2 add_library_sources consumes T1's tensorvia.ixx + *.hpp (plan Interf
Task 1: complete (commits 4c98fc0..HEAD-next, tests: task1-assert structural grep/file checks → ALL PASS; conversion spot-reviewed: types.hpp/ops.hpp/arithmetic.cpp/vulkan_context.hpp)

Task 1: Ruling: 转换范围 11→22 个分区 — 计划审计遗漏 backend 的 context/tensor/ufunc 共 11 个分区（断言 RED 输出暴露）— 成本：无，规则相同
Task 1: Ruling: tensorvia.ixx 行数上限 100→150 — 公开 API 111 个符号决定下限，"无逻辑代码"才是本质约束 — 成本：无
Task 1: Ruling: .ixx 转 .hpp 后源文件删除 — 计划 Step 7 只列了 header_core.ixx，22 个已转 .ixx 同删 — 成本：无（git 可恢复）
Task 1: Ruling: 3 行带行尾注释的 import 由脚本正则遗漏（ops.cpp 的 *_ops 声明 import），手工修正 — 成本：无
Setup: Ruling: sdd-workspace 脚本不存在（npm 版 superpowers 无此脚本），按 skill 文档路径约定手动建 workspace/ledger — 成本：无
Setup: 用户明示"直接开始干"= 在当前 main 分支工作的 explicit consent（executing-plans 的分支保护条款）

Task 2: complete (commits f4babd6..HEAD-next, tests: mcpp build CPU + test 6/6; vulkan build+smoke 1/1 (真跑无SKIP); cuda build+smoke 1/1; sycl build OK + smoke 40/42)

Task 2: Ruling: 用户会话外把 [toolchain] 改为 llvm@22.1.8（含 windows msvc 规划，是刻意选择）——不回改，按 llvm 推进（同时服务 Review Focus #1）— 代价：SYCL lane 见下条
Task 2: Ruling: vulkan SDK 改用 compat 包（用户指示）——[feature-deps.vulkan] headers(private)+runtime，删除 /usr/include hack；同时根治 spike 时 vulkan 后端不可用的环境问题（旧系统 vulkan 头/loader 路径），spike 记录的"疑似库 bug"实为环境问题 — 代价：无
Task 2: Ruling: 用户重写 mcpp.toml 时删了 [profile.release] NDEBUG → 恢复（vulkan.hpp 的 assert 会炸 release 构建）— 代价：无
Task 2: Ruling: 用户删了 CUDA toolkit 钉版 —— llvm route 下构建+smoke 通过（未复现 12.9/glibc 冲突）；nvcc route 风险仍在，若切回 gcc 需恢复钉版 — 代价：gcc+nvcc 路线可能炸
Task 2: Ruling: .cu/cuda_launch.hpp/ufunc_detail.hpp 的旧生成树 include（<tensorvia/x.h>）是 Task 1 转换遗漏（脚本只处理 .cpp/.ixx）——批量改为 interface/x.hpp（ops_api.h→ops.hpp）— 代价：无
Task 2: Ruling: -fopenmp=libgomp 硬编码在 cfg(env=gnu/musl) 行 —— gcc 工具链的消费方构建即炸（发布阻塞级），改为 compiler 层条件化（llvm/gcc 两行）— 代价：无
Task 2: Ruling: SYCL smoke 2 失败（NonContiguousInPlaceRejected/UnimplementedOperatorThrows）—— llvm/libc++ 与 libsycl/libstdc++ 的异常类型不匹配（42 章 unwinder 同源生态限制）；gcc16 环境同代码复现 CAUGHT runtime_error 全过证明转换零问题；-stdlib=libstdc++ 实验无效（穿越点在 dpcpp 编译的设备单元）— 接受为 llvm 路线已知限制 — 代价：llvm 下 SYCL 异常类型测试 40/42
Task 2: Ruling: 用户把 cuda/sycl 的 defines 从 cfg(accelerator) 移到 [features]（随 feature 而非 accel）——pack --features sycl 无 accel 的语义随之变化，留待 Task 6 边界测试验证 — 代价：pack 场景可能出带 define 但无设备单元的产物

Task 3: complete (commits cadb2ec..HEAD-next, tests: --check 绿; facade_api 1/1; 反向用例①漏导出→--check 红; ②幽灵导出→tensorvia.ixx 编译红(no member in ops); rerun 键接入后 mcpp build 漂移即红; 恢复后全量 test 6/6)

Task 3: Ruling: 幽灵导出的守卫在编译层而非 --check —— --check 管"磁盘文件与 .def 的一致性"（漂移），名字存在性由 tensorvia.ixx/facade_api.cpp 的编译抓；两层分工 — 代价：无
Task 3: Ruling: build.mcpp 需 rerun_if_changed(facade_exports.def) —— 否则 .def 变化不重跑 build.mcpp，check 被缓存绕过 — 代价：无

Task 4: complete (commits 040c430..HEAD-next, tests: cmake CPU build+ctest 100%(5); install 树 13 头+消费方链路 cpu sum=18; vulkan/cuda 组合构建 EXIT 0)

Task 4: Ruling: tests/examples 的 `import tensorvia;` → `#include "interface/tensorvia.hpp"` —— CMake 路径直编源码（无生成器重写），测试/示例作为使用方用头文件形态两路通用；facade_api.cpp 保持 import 且在 CMake BUILD_TEST 中排除（模块导出面是 mcpp test 的专属检查）— 代价：无
Task 4: Ruling: 安装树头文件在 configure 时生成改写版（build/install-headers/：interface/→tensorvia/ + 从 include/ 复制 float16/bfloat16）—— 安装版总头与分区头的内部 include 在安装布局自洽（使用方 #include <tensorvia.hpp> 全链路）— 代价：改公开头后需重新 configure 才刷新安装副本
Task 4: Ruling: include/ 加为 PRIVATE include dir —— _platform.h/_simd_math.h 是实现细节，可见于本库单元，不进安装面 — 代价：无
Task 4: Ruling: find_package(Python3) 移入 BACKEND_VULKAN 条件 —— 原全局 find 随生成段删除，仅 vulkan 的 gen_spv.py 还需要 — 代价：无
Task 4: minor(deferred): soname 不一致 —— mcpp.toml 写 libtensorvia.so.1，CMake SOVERSION=${PROJECT_VERSION_MAJOR}=0 产出 libtensorvia.so.0；既有差异非本任务引入，两条路径的产物 ABI 名不同 — 建议后续统一
Task 4: minor(deferred): CMake 路径的 Vulkan SDK 来自系统（find_package Vulkan + glslangValidator），mcpp 路径来自 compat 包 —— 两条路径 SDK 来源不同（README 依赖表已声明系统要求）— 如需统一可在 CMake 侧也引入 compat 包或 FetchContent

Task 5: complete (commits 8d1b33f..HEAD-next, tests: mcpp build+test 6/6; vulkan lane EXIT 0; cmake build EXIT 0 + ctest 全过; grep 全仓无生成器残留引用)

Task 5: Ruling: README 三处修正有 spike 实证依据 —— ①vulkan 无 --accel 轴（.comp 走 glslang action）；②消费方 CUDA/SYCL 必须传 --accel（spike 场景 3：依赖构建缺 accel 被 build.mcpp 拒绝；accel 是根级配置）；③--accel 版本号是 toolkit 线声明而非驱动版本（12.9 线撞 glibc C23 冲突实证；驱动由规则包探测为 fact）— 用户重写 README 时表述与实证相反，按实证修正 — 代价：无
Task 5: Ruling: package.py 依赖表同步（Python 3 仅 vulkan 后端需要——gen_spv.py；cpu/cuda/sycl 行的 python 依赖移除）— 成本：无

Task 6: complete (commits b4b5af5..HEAD-next, tests: spec Test Steps 1-7 全过; 消费方 llvm 零改动冒烟 cpu sum=18/scalar-add=24/cuda fill=1024; pack --features sycl 无 accel EXIT 0 出 CPU 包)

Task 6: Ruling: 去掉 -fno-math-errno（两行 cxxflags）—— 它进入 clang 的 BMI 兼容性检查，import 使用方不开同 flag 即 "errno ... disabled in precompiled file but is currently enabled" 构建失败（FR-7 阻塞：使用方为库的私有微优化买单）；分发兼容性优先，微优化可被 -O2+ 的默认行为覆盖 — 代价：数学函数 errno 设置的微优化丢失
Task 6: Ruling: 修正 Task 2 ledger 的风险记录 —— ops.cpp 的 dispatch 用 if constexpr (has_*_impl) 概念检查：BACKEND_* define 随 feature（用户改动）时，未链接的后端算子走运行时 "not implemented" 抛出而非链接失败，pack --features sycl 无 accel 实测 EXIT 0 出正确 CPU 包 — 原风险不成立
验收记录: mcpp build CPU 30.26s（基线 30.41s，持平）; mcpp test 6/6; vulkan/cuda smoke 1/1 真跑; sycl smoke 40/42（llvm/libc++ vs libsycl/libstdc++ 异常类型，gcc16 实证转换零问题）; cmake ctest 5/5; 消费方 llvm 端到端 EXIT 0; LLVM 工具链（Review Focus #1）验证通过

Final: fixed P0-1 (提交遗漏) — Task 3 的 git add 范围漏 src/interface/tensorvia.ixx，干净 HEAD 态 --check 红；已提交，stash 验证干净检出 --check EXIT 0 — suite 全绿
Final: fixed P0-2 (pack leg 缺头文件) — 公开头树从 src/interface/ 迁到 include/tensorvia/（mcpp 库约定：include/ 公开、src/ 不公开），pack 产物含完整头树 30 条目，解压包 path 依赖可编译 — 布局迁移后源码布局=安装布局
Final: fixed M-2 (install 陈旧) — 布局统一后 install 改写逻辑（configure 时生成 install-headers/）整个删除，install 直装源树；改公开头后 build+install 不再静默装旧头 — ctest+消费方验证
Final: fixed M-1 (漏导出无守卫) — gen_facade 加公开面扫描（类体删除+粘滞命名空间两遍法，零误报零漏报 104/104）；RED→GREEN：新增公开函数不登记 → --check 报 "public declaration not in the facade list" — 接入 --check/--write
Final: fixed M-3 (float16 ABI 钉定丢失) — TENSORVIA_FORCE_SOFTWARE_FLOAT16=1 两侧定义（mcpp.toml cuda build 段 + build.mcpp rules-cuda flags）+ types.hpp 注释更新 — cuda_smoke（含 float16 用例）1/1
Final: Ruling: P0-2 的 .so 运行时 libc++ 副本问题降级为二进制分发已知限制 — pack 的 .so 静态自含 libc++（mcpp portable 默认）与使用方二进制的 libc++ 副本不一致时 catch 跨 .so 异常 bad_cast 逃逸；cxx_runtime=toolchain-coupled 实验回滚（链到系统 libc++ 且引入 libunwind not found 失败模式）；源码分发渠道（registry tag 包/git/path，使用方与库同工具链）无此问题 — README 1.2 已注明限制与适用场景 — 代价：二进制包用户需避开跨 .so catch
Final: fixed m-1..m-6 — CMake PUBLIC -fno-math-errno 删除；注释组更新（build.mcpp 头部/sycl_context/tensorvia.hpp/facade_exports.def）；README 开头总述+CUDA 版本表述+删不存在选项+pack 限制；core.hpp 孤儿删除；bench.cpp 排除 BUILD_TEST 且 BUILD_BENCH glob 修正为 tests/bench.cpp；package.py 注释限定 — suite 全绿
Final 回归: mcpp CPU 6/6; vulkan smoke 1/1; cuda smoke 1/1; facade_api 1/1; gen_facade --check 绿(含扫描器); cmake ctest 4/4 + bench 产出; pack 解压消费可编译

Final: Ruling（用户裁决）: TENSORVIA_FORCE_SOFTWARE_FLOAT16 专用宏根除 —— BACKEND_CUDA 在 CUDA lane 必然定义（[features] cuda 的 defines + build.mcpp 的 nvcc flags），types.hpp 改 `#if defined(BACKEND_CUDA)` 钉软件 float16 即可，专用宏是过度设计（YAGNI）— 代价：无，cuda_smoke（含 float16 用例）1/1 验证

Final: Ruling（用户裁决，强烈）: 测试不针对特定后端 —— backends/<x>/smoke.cpp（97 个 Device::X 用例）整套删除，[test] discover 恢复默认（tests/**/*.cpp）；tests/ 只留后端无关资产：test.cpp（23 用例零 Device::）/header_smoke.cpp（CMake 头文件冒烟）/facade_api.cpp（导出面）/bench.cpp；设备后端的验证 = 构建 lane 门（1.3 的 --features+--accel 命令）+ 消费方冒烟 — 代价：后端算子实现无自动行为回归守卫（用户接受，后端正确性由构建验证+使用方覆盖）

Final: 测试命令补齐（用户指令）— README 测试节列出四 lane 的 mcpp test 命令；实测：CPU 4/4、vulkan 4/4、cuda 4/4、sycl 3/4（View.InPlaceRejectsNonContiguous，libc++/libstdc++ 异常类型分裂——删掉的 sycl_smoke 同根问题的公共测试蔓延，已注明为 llvm+SYCL lane 已知限制，gcc 实测全过）
Final: Ruling: --features 的值是 feature 名（vulkan 而非 vulkan1.4——后者是 accel 串，触发未声明 feature）；--features sycl 必须配 --accel sycl（设备编译开关是根级 accel）— 用户初版命令的两处错误按实证修正

Final: 用户提问的实证答案 —— SYCL lane 的异常类型限制根源是 llvm payload(libc++) 与 libsycl(libstdc++) 两套 C++ 运行时；解法是 SYCL lane 用 gcc 工具链（mcpp test --toolchain gcc@16.1.0 --features sycl --accel sycl 实测 4/4 全过，零 flag 冲突——compiler 层条件化修复见效）—— README 已知限制注记已升级为具体命令

Final: Ruling: [toolchain] default = { configure = "build.mcpp" } 按 lane 自动切工具链实测不可行（已回滚）—— toolchain phase 的 build.mcpp 未运行：verbose 显示 plugins 库在 build.mcpp ran 之后才 Fresh 编译（toolchain phase 在依赖图解析前，import mcpp.plugins.toolchain 的库尚不可用），mcpp 静默回退；无条件 tc::use(gcc) 亦不生效。机制文档（20 章 2026.10.1.3+）与 engine 2026.10.17.1 行为存在差异，值得向 mcpp 上游报 issue — 回滚后 SYCL lane 用显式 --toolchain gcc@16.1.0（README 已有），CPU 4/4、SYCL 4/4 — 代价：无（净回滚）

Final: Ruling（用户裁决，二次修正）: tests/ 改回头文件范式 —— CMake 路径应有全量测试覆盖（此前只建 header_smoke 一个冒烟是失衡的）；test.cpp/bench.cpp 改 #include "tensorvia.hpp"，header_smoke.cpp 删除（被全量 test.cpp 取代），facade_api.cpp 保持 import（导出面守卫在 include 形态下失效，仅 mcpp test 构建，CMake 排除）；CMakeLists 恢复 glob+排除模式 — 实测 mcpp test 3/3、CMake ctest 1/1（23 用例全量）— 代价：mcpp 路径的行为测试测的是头文件形态而非 import 形态（export 面由 facade_api 的符号守卫覆盖，行为等价性由共享同一份 types.hpp 保证）
