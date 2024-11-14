cmake_minimum_required (VERSION 3.20)

project(SYCL-Advection LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(AdaptiveCpp REQUIRED)
find_package(benchmark REQUIRED)

add_subdirectory(tpls/mdspan)
add_subdirectory(src)
