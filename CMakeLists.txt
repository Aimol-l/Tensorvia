cmake_minimum_required(VERSION 3.28)

# The SYCL backend needs icpx for both host and device code, and the compiler
# has to be fixed before project(). Everything in this build is therefore
# compiled with icpx when BACKEND_SYCL is on (the CPU/Vulkan lanes are plain C++
# and icpx handles them too).
set(BACKEND_SYCL OFF CACHE BOOL "Enable the SYCL backend")
if(BACKEND_SYCL AND NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(TVV_ICX NAMES icpx REQUIRED)
    set(CMAKE_CXX_COMPILER "${TVV_ICX}" CACHE FILEPATH "C++ compiler" FORCE)
endif()

project(tensorvia
    VERSION 0.3.0
    DESCRIPTION "Tensorvia: C++23 module tensor library with CPU, Vulkan, CUDA and SYCL backends"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Predictable artifact layout: <build>/lib, <build>/bin, <build>/tests.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# CMake builds the library's header form directly: src/interface/*.hpp are the
# single source of truth (mcpp wraps them in a module for `import tensorvia;`).
# Module scanning stays off, so no module support is required from the toolchain
# and consumers only need an ordinary #include <tensorvia.hpp>.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BACKEND_CPU     "Enable the CPU backend"     ON)
option(BACKEND_VULKAN  "Enable the Vulkan backend"  OFF)
option(BACKEND_CUDA    "Enable the CUDA backend"    OFF)
option(BUILD_TEST      "Build tests/ (one gtest executable per .cpp)" OFF)
option(BUILD_BENCH     "Build benchmarks/"          OFF)
option(BUILD_EXAMPLES  "Build examples/"            OFF)

if(NOT BACKEND_CPU AND NOT BACKEND_VULKAN AND NOT BACKEND_CUDA AND NOT BACKEND_SYCL)
    message(FATAL_ERROR
        "Enable at least one backend: BACKEND_CPU, BACKEND_VULKAN, BACKEND_CUDA and/or BACKEND_SYCL")
endif()

if(BACKEND_SYCL AND NOT CMAKE_CXX_COMPILER MATCHES "icpx|icpc|dpcpp|clang\\+\\+")
    message(FATAL_ERROR
        "BACKEND_SYCL requires the Intel oneAPI DPC++ compiler (icpx). "
        "Configure with -DCMAKE_CXX_COMPILER=icpx or leave CMAKE_CXX_COMPILER unset.")
endif()

# ---------------------------------------------------------------------------
# Sources are plain C++: the headers under src/interface are the single source
# of truth (mcpp wraps them in a module for `import tensorvia;`). Nothing is
# generated — this build compiles the very same .cpp units against the very
# same headers.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Source lists. Keep this in sync with mcpp.toml: the base list never contains
# Vulkan implementation units, so a CPU-only build never sees <vulkan.hpp>.
# ---------------------------------------------------------------------------
set(TVV_BASE_INPUTS
    src/core/tensor.cpp
    src/core/ops.cpp
    src/core/factory.cpp
    src/core/copy.cpp
    src/init.cpp
    src/backend/cpu/cpu_context.cpp
    src/backend/cpu/cpu_tensor.cpp
)
file(GLOB TVV_CPU_OPS_INPUTS CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/backend/cpu/ops/*.cpp"
)
list(APPEND TVV_BASE_INPUTS ${TVV_CPU_OPS_INPUTS})

set(TVV_SOURCE_INPUTS ${TVV_BASE_INPUTS})

if(BACKEND_CPU)
    find_package(OpenMP REQUIRED)
endif()

if(BACKEND_VULKAN)
    find_package(Vulkan REQUIRED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    find_program(TVV_GLSLANG_VALIDATOR NAMES glslangValidator REQUIRED)

    list(APPEND TVV_SOURCE_INPUTS src/init_vulkan.cpp)
    file(GLOB TVV_VULKAN_INPUTS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/backend/vulkan/*.cpp"
    )
    list(APPEND TVV_SOURCE_INPUTS ${TVV_VULKAN_INPUTS})
    # SPIR-V registry: generated into the build tree so the source tree stays
    # clean. This is the same generator mcpp's build.mcpp invokes.
    file(GLOB_RECURSE TVV_SHADERS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/shader/*.comp"
    )
    set(TVV_SPV_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen")
    set(TVV_SPV_REGISTRY_CPP "${TVV_SPV_GEN_DIR}/spirv/spv_registry.cpp")
    set(TVV_SPV_REGISTRY_H   "${TVV_SPV_GEN_DIR}/spirv/spv_registry.h")

    add_custom_command(
        OUTPUT ${TVV_SPV_REGISTRY_CPP} ${TVV_SPV_REGISTRY_H}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${TVV_SPV_GEN_DIR}/spirv"
        COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tools/gen_spv.py"
                --out "${TVV_SPV_GEN_DIR}"
                --shader-root "${PROJECT_SOURCE_DIR}/shader"
                --glslang "${TVV_GLSLANG_VALIDATOR}"
        DEPENDS ${TVV_SHADERS}
                "${PROJECT_SOURCE_DIR}/tools/gen_spv.py"
        COMMENT "Compiling SPIR-V shaders and generating spv_registry"
        VERBATIM
    )
endif()

# CUDA. The device units are plain .cu files consumed directly by nvcc; they
# are never handed to the header-mode generator (which only reads .ixx/.cpp).
# Each .cu includes the generated headers as well as the typed launch contract,
# so host drivers and kernels are one translation unit — there is no C island
# and no nvcc module support to worry about.
if(BACKEND_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    # No relocatable device code: every kernel is self-contained, so the link
    # needs no -dlink step.
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()
    find_package(CUDAToolkit REQUIRED)

    list(APPEND TVV_SOURCE_INPUTS src/init_cuda.cpp)
    file(GLOB TVV_CUDA_DEVICE_INPUTS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/backend/cuda/device/*.cu"
    )
endif()

# SYCL. Unlike CUDA, the host drivers stay ordinary C++ (they go through the
# same header-mode generator as everything else) and only the .sycl units hold
# SYCL types. Both are compiled by icpx with -fsycl, so the extern "C" seam
# between them degenerates to a normal call — but the exact same seam is what
# mcpp uses across its two compilers, which keeps the two build lanes
# behaviourally identical.
if(BACKEND_SYCL)
    list(APPEND TVV_SOURCE_INPUTS src/init_sycl.cpp)
    file(GLOB TVV_SYCL_HOST_INPUTS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/backend/sycl/host/*.cpp"
    )
    list(APPEND TVV_SOURCE_INPUTS ${TVV_SYCL_HOST_INPUTS})
    file(GLOB TVV_SYCL_DEVICE_INPUTS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/backend/sycl/device/*.sycl"
    )
    # CMake has no built-in rule for .sycl; make it compile as C++ and pass
    # -fsycl so the device code is compiled for SPIR-V as well.
    set_source_files_properties(${TVV_SYCL_DEVICE_INPUTS}
        PROPERTIES LANGUAGE CXX)
endif()

set(TVV_SOURCES ${TVV_SOURCE_INPUTS})
if(BACKEND_VULKAN)
    list(APPEND TVV_SOURCES ${TVV_SPV_REGISTRY_CPP})
endif()
if(BACKEND_CUDA)
    list(APPEND TVV_SOURCES ${TVV_CUDA_DEVICE_INPUTS})
endif()
if(BACKEND_SYCL)
    list(APPEND TVV_SOURCES ${TVV_SYCL_DEVICE_INPUTS})
endif()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(tensorvia SHARED
    ${TVV_SOURCES}
)

set_target_properties(tensorvia PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

target_compile_features(tensorvia PUBLIC cxx_std_23)

target_include_directories(tensorvia
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
# _platform.h / _simd_math.h are implementation details: they sit under the
# same root but are never installed (see the install section below).

if(BACKEND_VULKAN)
    target_include_directories(tensorvia
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/gen>
    )
    target_link_libraries(tensorvia PRIVATE Vulkan::Vulkan)
endif()

if(BACKEND_CUDA)
    # The .cu units include cuda_launch.hpp / ufunc_detail.hpp next to the
    # generated headers; -I covers cuda_launch.hpp from device/.
    target_include_directories(tensorvia PRIVATE ${PROJECT_SOURCE_DIR}/src/backend/cuda)
    target_link_libraries(tensorvia PRIVATE CUDA::cudart)
    # curand_kernel.h is header-only, but the toolkit target keeps the device
    # random library available when a form of it is needed at link time.
    if(TARGET CUDA::curand)
        target_link_libraries(tensorvia PRIVATE CUDA::curand)
    endif()
    set_target_properties(tensorvia PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared
        POSITION_INDEPENDENT_CODE ON
    )
endif()

if(BACKEND_CPU)
    target_link_libraries(tensorvia PUBLIC OpenMP::OpenMP_CXX)
endif()

if(BACKEND_SYCL)
    # sycl_island.h sits next to host/ and device/; both include it by name.
    target_include_directories(tensorvia PRIVATE ${PROJECT_SOURCE_DIR}/src/backend/sycl)
    # -fsycl on the whole CXX language is fine: the CPU/Vulkan/init units are
    # plain C++ and icpx compiles them unchanged. The expression keeps the flag
    # away from CUDA sources, whose nvcc host pass (g++) rejects it. The
    # link-side flag pulls in libsycl.
    target_compile_options(tensorvia PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-fsycl>")
    target_link_options(tensorvia PRIVATE -fsycl)
endif()

set(TVV_DEFINITIONS)
if(BACKEND_CPU)
    list(APPEND TVV_DEFINITIONS BACKEND_CPU)
endif()
if(BACKEND_VULKAN)
    list(APPEND TVV_DEFINITIONS BACKEND_VULKAN)
endif()
if(BACKEND_CUDA)
    list(APPEND TVV_DEFINITIONS BACKEND_CUDA)
endif()
if(BACKEND_SYCL)
    list(APPEND TVV_DEFINITIONS BACKEND_SYCL)
endif()
target_compile_definitions(tensorvia
    PUBLIC  ${TVV_DEFINITIONS}
            TENSORVIA_VERSION="${PROJECT_VERSION}"
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(tensorvia PRIVATE -Wall -Wextra)
endif()

# Match mcpp.toml's baseline: _simd_math.h uses AVX2/FMA intrinsics
# unconditionally, so the whole library is built for x86-64-v3.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
        target_compile_options(tensorvia PUBLIC -march=x86-64-v3)
    endif()
endif()

add_library(Tensorvia::tensorvia ALIAS tensorvia)

# ---------------------------------------------------------------------------
# Tests and examples
# ---------------------------------------------------------------------------
if(BUILD_TEST)
    enable_testing()
    # Tests are GoogleTest suites; gtest_main supplies main().
    find_package(GTest REQUIRED)

    file(GLOB TVV_TEST_INPUTS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/tests/*.cpp"
    )
    # facade_api.cpp compiles only through the module wrapper (`import
    # tensorvia;`) — it is mcpp test's surface check, not a header-path test.
    # bench.cpp has its own main() and is a benchmark, not a test.
    list(REMOVE_ITEM TVV_TEST_INPUTS
        "${PROJECT_SOURCE_DIR}/tests/facade_api.cpp"
        "${PROJECT_SOURCE_DIR}/tests/bench.cpp")
    set(TVV_TEST_SOURCES ${TVV_TEST_INPUTS})
    foreach(test_file ${TVV_TEST_SOURCES})
        get_filename_component(test_name ${test_file} NAME_WE)
        # CTest reserves the target name "test" for the aggregate command.
        if(test_name STREQUAL "test")
            set(test_target tensorvia_test)
        else()
            set(test_target ${test_name})
        endif()

        add_executable(${test_target} "${test_file}")
        set_target_properties(${test_target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests")
        target_link_libraries(${test_target} PRIVATE tensorvia GTest::gtest_main)
        if(BACKEND_CPU)
            target_link_libraries(${test_target} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        add_test(NAME ${test_target} COMMAND ${test_target})
    endforeach()
endif()

if(BUILD_BENCH)
    set(TVV_BENCH_SOURCES "${PROJECT_SOURCE_DIR}/tests/bench.cpp")
    foreach(bench_file ${TVV_BENCH_SOURCES})
        get_filename_component(bench_name ${bench_file} NAME_WE)
        add_executable(${bench_name} "${bench_file}")
        target_link_libraries(${bench_name} PRIVATE tensorvia)
        if(BACKEND_CPU)
            target_link_libraries(${bench_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endforeach()
endif()

if(BUILD_EXAMPLES)
    file(GLOB TVV_EXAMPLE_INPUTS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/examples/*.cpp"
    )
    set(TVV_EXAMPLE_SOURCES ${TVV_EXAMPLE_INPUTS})
    foreach(example_file ${TVV_EXAMPLE_SOURCES})
        get_filename_component(example_name ${example_file} NAME_WE)
        add_executable(${example_name} "${example_file}")
        target_link_libraries(${example_name} PRIVATE tensorvia)
        if(BACKEND_CPU)
            target_link_libraries(${example_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Install / export
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS tensorvia
    EXPORT TensorviaTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Public surface, installed straight from the source tree: the source layout
# IS the install layout (include/tensorvia.hpp + include/tensorvia/*.hpp), so
# every internal include resolves the same way in both. Install explicit
# files, never a directory, so a CPU package cannot ship Vulkan headers. The
# backend/{cuda,sycl} partition headers stay internal: nothing in the installed
# surface includes them. _platform.h / _simd_math.h stay out as well.
set(TVV_IFACE_DIR "${PROJECT_SOURCE_DIR}/include/tensorvia")
file(GLOB TVV_PUBLIC_PARTITION_HEADERS CONFIGURE_DEPENDS
    "${TVV_IFACE_DIR}/*.hpp"
)
if(BACKEND_VULKAN)
    file(GLOB TVV_VULKAN_PARTITION_HEADERS CONFIGURE_DEPENDS
        "${TVV_IFACE_DIR}/backend/vulkan/*.hpp"
    )
    list(APPEND TVV_PUBLIC_PARTITION_HEADERS ${TVV_VULKAN_PARTITION_HEADERS})
endif()

install(FILES "${PROJECT_SOURCE_DIR}/include/tensorvia.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${TVV_PUBLIC_PARTITION_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensorvia)
install(EXPORT TensorviaTargets
    FILE TensorviaTargets.cmake
    NAMESPACE Tensorvia::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensorvia
)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/TensorviaConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/TensorviaConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensorvia
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TensorviaConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/TensorviaConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/TensorviaConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensorvia
)
