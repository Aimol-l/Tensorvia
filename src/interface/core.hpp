#pragma once

// Module-lane provider of :core (CPU / Vulkan artifacts).
//
// The CUDA lane provides the same partition from header_core.ixx instead, so
// exactly one definition of `tensorvia:core` participates in any build.

#include "interface/version.hpp"
#include "interface/types.hpp"
#include "interface/context.hpp"
#include "interface/tensor.hpp"
#include "interface/factory.hpp"
#include "interface/cpu_context.hpp"
#include "interface/ops.hpp"
