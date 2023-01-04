
# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.12)

# 默认是debug版本
if(NOT DEFINED CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# 项目信息
project (phonedet_test)
set(CMAKE_CXX_STANDARD 11)

#-O2是用来调节编译时的优化程度的，最高为-O3,最低为-O0（即不做优化）；
#-Wall表示编译后显示所有警告
#-pthread表示添加多线程使用
set(CMAKE_CXX_FLAGS "-O2 -Wall -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CMAKE_CXX_FLAGS}")

#是否开启测试，若否则直接生成库
OPTION (Build_Example       "Build example with source"       ON)

if(Build_Example)
    add_subdirectory(./src)
else()
    add_subdirectory(./phone_api_cmake)
endif()