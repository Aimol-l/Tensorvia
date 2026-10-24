#pragma once
// 总头文件：tensorvia 的公开 API 面（= mcpp 模块包装 tensorvia.ixx 的 export 面）。
//
// - 实现单元（src/**/*.cpp）include 本文件；
// - mcpp 的模块包装 src/interface/tensorvia.ixx 在全局模块片段 include 本文件，
//   其 export 清单由 tools/gen_facade.py 从 tools/facade_exports.def 生成/校验；
// - CMake 路径与安装包的消费方直接 include 本文件（#include <tensorvia.hpp>）。
#include "tensorvia/version.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/context.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/factory.hpp"
#include "tensorvia/cpu_context.hpp"
#include "tensorvia/ops.hpp"
