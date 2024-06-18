cmake_minimum_required(VERSION 3.10)

project(pqSearch)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(pqSearch pqSearch.cpp utils/Heap.cpp)

target_include_directories(pqSearch PRIVATE ${PROJECT_SOURCE_DIR})
