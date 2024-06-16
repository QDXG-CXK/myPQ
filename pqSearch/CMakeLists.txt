cmake_minimum_required(VERSION 3.10)

project(pqSearch)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(pqSearch pqSearch.cpp utils/Heap.cpp)

target_include_directories(pqSearch PRIVATE ${PROJECT_SOURCE_DIR}/utils)
