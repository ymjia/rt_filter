cmake_minimum_required(VERSION 3.23)

project(rt_filter_cpp_demo LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_GENERATOR STREQUAL "Xcode")
    set(CMAKE_XCODE_GENERATE_SCHEME ON)
endif()

find_package(Eigen3 REQUIRED)

add_library(output_alg_filters
    ../butterworth.cpp
    ../butterworth.hpp
    ../filter_timing.hpp
    ../one_euro_z.cpp
    ../one_euro_z.hpp
    ../ukf.cpp
    ../ukf.hpp
)
target_include_directories(output_alg_filters PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(output_alg_filters PUBLIC Eigen3::Eigen)
target_compile_features(output_alg_filters PUBLIC cxx_std_14)
set_target_properties(output_alg_filters PROPERTIES CXX_EXTENSIONS OFF)

add_executable(rt_filter_cpp_demo main.cpp)
target_link_libraries(rt_filter_cpp_demo PRIVATE output_alg_filters)
target_compile_features(rt_filter_cpp_demo PRIVATE cxx_std_14)
set_target_properties(rt_filter_cpp_demo PROPERTIES CXX_EXTENSIONS OFF)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} FILES main.cpp)
source_group(
    TREE ${CMAKE_CURRENT_LIST_DIR}/..
    FILES
        ../filter_timing.hpp
        ../butterworth.cpp
        ../butterworth.hpp
        ../one_euro_z.cpp
        ../one_euro_z.hpp
        ../ukf.cpp
        ../ukf.hpp
)

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT rt_filter_cpp_demo)

if(MSVC)
    set_property(TARGET output_alg_filters PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set_property(TARGET rt_filter_cpp_demo PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_compile_options(output_alg_filters PUBLIC /utf-8)
    target_compile_options(rt_filter_cpp_demo PRIVATE /utf-8)
endif()
