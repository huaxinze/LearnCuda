cmake_minimum_required(VERSION 3.14)

project(gemm)

set(CMAKE_CXX_STANDARD 11)
 
# 包含CUDA的头文件路径
find_package(CUDA REQUIRED)

# 添加可执行文件
cuda_add_executable(gemm main.cu)
target_link_libraries( gemm ${CUDA_cublas_LIBRARY} )

