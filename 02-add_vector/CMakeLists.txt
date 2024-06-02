cmake_minimum_required(VERSION 3.14)

project(add_vector)

set(CMAKE_CXX_STANDARD 11)
 
# 包含CUDA的头文件路径
find_package(CUDA REQUIRED)

# 添加可执行文件
cuda_add_executable(add_vector main.cu)

