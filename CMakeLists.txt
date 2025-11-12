cmake_minimum_required(VERSION 3.25)
# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_BUILD_PARALLEL_LEVEL 6)
# 项目根路径
get_filename_component(PROJECT_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_DIR}/lib)
# ------------------------- 后端选项定义 -------------------------
option(BUILD_TEST      "build test file with ./tests/*.cpp" OFF)
option(BACKEND_CPU     "Enable CPU backend" OFF)
option(BACKEND_CUDA    "Enable CUDA backend" OFF)
option(BACKEND_SYCL    "Enable SYCL backend" OFF)
option(BACKEND_VULKAN  "Enable Vulkan backend" OFF)


# 编译器配置
if(BACKEND_CPU)
    set(CMAKE_C_COMPILER "/usr/bin/gcc")
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
    add_definitions(-DBACKEND_CPU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=skylake -ffast-math -march=native -fopenmp")
endif()

if(BACKEND_CUDA)
    add_definitions(-DBACKEND_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")
    set(CMAKE_CUDA_ARCHITECTURES "native")  # 自动检测GPU架构
endif()

if(BACKEND_SYCL)
    add_definitions(-DBACKEND_SYCL)
    set(CMAKE_C_COMPILER "/opt/intel/oneapi/compiler/latest/bin/icx")
    set(CMAKE_CXX_COMPILER "/opt/intel/oneapi/compiler/latest/bin/icpx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -O3 -fiopenmp -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -O3 -fiopenmp -ffast-math -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64")
endif()

if(BACKEND_VULKAN)
    set(CMAKE_C_COMPILER "/usr/bin/clang")
    set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
    add_definitions(-DBACKEND_VULKAN)
    find_package(Vulkan REQUIRED)
endif()

project(
    tensorvia
    LANGUAGES CXX C CUDA    #编程语言
    DESCRIPTION "多后端 Tensor 加速计算库"
)
# ------------------------- 依赖查找 -------------------------
find_package(TBB REQUIRED)
# ------------------------- .h,hpp文件路径 -------------------------
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/core
)
if(BACKEND_CPU)
    find_package(OpenMP REQUIRED)
    include_directories(${CMAKE_SOURCE_DIR}/include/backend/cpu)
endif()
if(BACKEND_CUDA)
    find_package(OpenMP REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    include_directories(${CMAKE_SOURCE_DIR}/include/backend/cpu)
    include_directories(${CMAKE_SOURCE_DIR}/include/backend/cuda)
endif()
if(BACKEND_SYCL)
    find_package(IntelSYCL REQUIRED)
    include_directories(${CMAKE_SOURCE_DIR}/include/backend/cpu)
    include_directories(${CMAKE_SOURCE_DIR}/include/backend/sycl)
endif()
if(BACKEND_VULKAN)
    find_package(OpenMP REQUIRED)
    include_directories(${CMAKE_SOURCE_DIR}/include/backend/cpu)
    include_directories(${CMAKE_SOURCE_DIR}/include/backend/vulkan)
endif()
# ------------------------- .cpp,cu,cxx源文件收集 -------------------------
file(GLOB SRC_FILES 
    CMAKE_CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/backend/cpu/*.cpp"
    "${PROJECT_SOURCE_DIR}/include/backend/cpu/*.cpp"
)
# 各个后端的源文件
if(BACKEND_CPU)
    file(GLOB_RECURSE CPU_OPS_SOURCES 
        "${PROJECT_SOURCE_DIR}/include/backend/cpu/*.cpp"
    )
    list(APPEND SRC_FILES ${CPU_OPS_SOURCES})
endif()
if(BACKEND_CUDA)
    file(GLOB_RECURSE CUDA_OPS_SOURCES 
        "${PROJECT_SOURCE_DIR}/src/backend/cuda/*.cu"
        "${PROJECT_SOURCE_DIR}/include/backend/cuda/*.cpp"
    )
    list(APPEND SRC_FILES ${CUDA_OPS_SOURCES})
endif()
if(BACKEND_SYCL)
    file(GLOB_RECURSE SYCL_OPS_SOURCES 
        "${PROJECT_SOURCE_DIR}/src/backend/sycl/*.cpp"
        "${PROJECT_SOURCE_DIR}/include/backend/sycl/*.cpp"
    )
    list(APPEND SRC_FILES ${SYCL_OPS_SOURCES})

endif()
if(BACKEND_VULKAN)
    file(GLOB_RECURSE VULKAN_OPS_SOURCES 
        "${PROJECT_SOURCE_DIR}/src/backend/vulkan/*.cpp"
        "${PROJECT_SOURCE_DIR}/include/backend/vulkan/*.cpp"
    )
    list(APPEND SRC_FILES ${VULKAN_OPS_SOURCES})
endif()


# ------------------------- 主库构建 -------------------------
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb)
if(BACKEND_CPU)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()
if(BACKEND_SYCL)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${IntelSYCL_LIBRARIES})
endif()

if(BACKEND_VULKAN)
    target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan})
endif()

if(BACKEND_CUDA)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
endif()
# ------------------------- 测试系统 -------------------------
if(BUILD_TEST)

    file(GLOB TEST_FILES "${PROJECT_SOURCE_DIR}/tests/*.cpp")

    foreach(test_file ${TEST_FILES})
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file})
        target_link_libraries(${test_name} PRIVATE ${PROJECT_NAME})
        target_include_directories(${test_name} PRIVATE ${INCLUDE_DIRS})
        target_compile_definitions(${test_name} PRIVATE ${BACKEND_MACROS})
        if(BACKEND_CPU)
           target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        if(BACKEND_SYCL)
            target_compile_options(${test_name} PRIVATE -fsycl -O2 -fiopenmp -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64)
            target_link_libraries(${test_name} PRIVATE ${IntelSYCL_LIBRARIES})
        endif()
        if(BACKEND_CUDA)
           target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        if(BACKEND_VULKAN)
        #    target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endforeach()
endif()