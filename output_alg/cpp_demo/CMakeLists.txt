cmake_minimum_required(VERSION 3.23)

project(rt_filter_cpp_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

add_library(output_alg_filters
    ../one_euro_z.cpp
    ../ukf.cpp
)
target_include_directories(output_alg_filters PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(output_alg_filters PUBLIC Eigen3::Eigen)

add_executable(rt_filter_cpp_demo main.cpp)
target_link_libraries(rt_filter_cpp_demo PRIVATE output_alg_filters)

if(MSVC)
    target_compile_options(output_alg_filters PUBLIC /utf-8)
    target_compile_options(rt_filter_cpp_demo PRIVATE /utf-8)
endif()
