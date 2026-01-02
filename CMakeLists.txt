cmake_minimum_required(VERSION 3.25)# 项目信息

# ------------------------- 设置 C++ 标准 -------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")



# ------------------------- 后端选项定义 -------------------------
option(BUILD_TEST      "build test file with ./tests/*.cpp" OFF)
option(BACKEND_CPU     "Enable CPU backend" OFF)
option(BACKEND_CUDA    "Enable CUDA backend" OFF)
option(BACKEND_SYCL    "Enable SYCL backend" OFF)
option(BACKEND_VULKAN  "Enable Vulkan backend" OFF)

# 检查是否启用了至少一个后端
if(NOT BACKEND_CPU AND NOT BACKEND_CUDA AND NOT BACKEND_SYCL AND NOT BACKEND_VULKAN)
    message(WARNING "No backend enabled. Please enable at least one backend (BACKEND_CPU, BACKEND_CUDA, BACKEND_SYCL, or BACKEND_VULKAN). Enabling CPU backend by default.")
    set(BACKEND_CPU ON)
endif()

# 编译器配置
if(BACKEND_CPU)
    find_package(OpenMP REQUIRED)
    add_definitions(-DBACKEND_CPU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx2 -mfma -fopenmp")
endif()

if(BACKEND_CUDA)
    add_definitions(-DBACKEND_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    
    find_package(OpenMP REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    
    set(CMAKE_CUDA_ARCHITECTURES "native")  # 常见现代GPU架构
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx2 -mfma -fopenmp")
endif()

if(BACKEND_SYCL)
    add_definitions(-DBACKEND_SYCL)
    set(CMAKE_C_COMPILER "/opt/intel/oneapi/compiler/latest/bin/icx")
    set(CMAKE_CXX_COMPILER "/opt/intel/oneapi/compiler/latest/bin/icpx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -O2 -fiopenmp -mavx2 -mfma -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64")
endif()

if(BACKEND_VULKAN)
    add_definitions(-DBACKEND_VULKAN)
    find_package(Vulkan REQUIRED)
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx2 -mfma -fopenmp")
endif()

project(
    tensorvia
    LANGUAGES CXX C CUDA    #编程语言
    DESCRIPTION "多后端 Tensor 加速计算库"
    VERSION 0.1.0
)
# 设置输出目录到构建目录下
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
# ------------------------- 头文件目录 -------------------------
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/core
)
# 后端头文件目录
if(BACKEND_CPU)
    include_directories(${PROJECT_SOURCE_DIR}/include/backend/cpu)
endif()
if(BACKEND_CUDA)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    include_directories(${CMAKE_SOURCE_DIR}/include/backend/cpu)
    include_directories(${PROJECT_SOURCE_DIR}/include/backend/cuda)
endif()
if(BACKEND_SYCL)
    find_package(IntelSYCL REQUIRED)
    include_directories(${CMAKE_SOURCE_DIR}/include/backend/cpu)
    include_directories(${PROJECT_SOURCE_DIR}/include/backend/sycl)
endif()
if(BACKEND_VULKAN)
    include_directories(${CMAKE_SOURCE_DIR}/include/backend/cpu)
    include_directories(${PROJECT_SOURCE_DIR}/include/backend/vulkan)

endif()

# ------------------------- 源文件收集 -------------------------
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
        "${PROJECT_SOURCE_DIR}/include/backend/cuda/*.cu"
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

# 设置库的属性
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

# 链接库
if(BACKEND_CPU AND OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(BACKEND_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
endif()

if(BACKEND_SYCL AND CMAKE_SYCL_ENABLED)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${IntelSYCL_LIBRARIES})
endif()

if(BACKEND_VULKAN AND Vulkan_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${Vulkan_LIBRARIES})
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

# 设置库的别名
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# ------------------------- 安装配置 -------------------------
include(GNUInstallDirs)

# 安装目标库
install(TARGETS ${PROJECT_NAME}
    EXPORT TensorviaTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 安装头文件
install(
    FILES
        ${PROJECT_SOURCE_DIR}/include/core/ops.h
        ${PROJECT_SOURCE_DIR}/include/core/types.h
        ${PROJECT_SOURCE_DIR}/include/core/tensor.h
        ${PROJECT_SOURCE_DIR}/include/core/type_traits.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensorvia
)


# 设置CMake包配置
install(EXPORT TensorviaTargets
    FILE TensorviaTargets.cmake
    NAMESPACE Tensorvia::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensorvia
)

# 配置包配置文件
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/Tensorvia.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/TensorviaConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensorvia
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/TensorviaConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensorvia
)

# ------------------------- 测试系统 -------------------------
if(BUILD_TEST)
    enable_testing()
    
    file(GLOB TEST_FILES "${PROJECT_SOURCE_DIR}/tests/*.cpp")

    foreach(test_file ${TEST_FILES})
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file})
        target_link_libraries(${test_name} PRIVATE ${PROJECT_NAME})
        target_include_directories(${test_name} PRIVATE ${INCLUDE_DIRS})
        
        # 为测试目标添加后端定义
        if(BACKEND_CPU)
           target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        if(BACKEND_SYCL)
            target_compile_options(${test_name} PRIVATE -fsycl -O2 -fiopenmp -mavx2 -mfma -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64)
            target_link_libraries(${test_name} PRIVATE ${IntelSYCL_LIBRARIES})
        endif()
        if(BACKEND_CUDA)
           target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        if(BACKEND_VULKAN)
            target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC ${Vulkan_LIBRARIES})
            target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()
