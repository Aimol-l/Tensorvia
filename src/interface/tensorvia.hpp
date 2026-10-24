#pragma once
// 总头文件：tensorvia 的公开 API 面（与模块 lane :core 分区的再导出列表一致）。
//
// - 实现单元（src/**/*.cpp）include 本文件；
// - mcpp 的模块包装 src/interface/tensorvia.ixx 在全局模块片段 include 本文件，
//   其 export 清单由 tools/gen_facade.py 从 tools/facade_exports.def 生成/校验；
// - CMake 路径与安装包的消费方直接 include 本文件（#include <tensorvia.hpp>）。
#include "interface/version.hpp"
#include "interface/types.hpp"
#include "interface/context.hpp"
#include "interface/tensor.hpp"
#include "interface/factory.hpp"
#include "interface/cpu_context.hpp"
#include "interface/ops.hpp"
