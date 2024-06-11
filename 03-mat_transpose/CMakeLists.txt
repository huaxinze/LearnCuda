cmake_minimum_required(VERSION 3.14)

project(mat_transpose)

set(CMAKE_CXX_STANDARD 11)
 
# 包含CUDA的头文件路径
find_package(CUDA REQUIRED)

# 添加可执行文件
cuda_add_executable(mat_transpose main.cu)

