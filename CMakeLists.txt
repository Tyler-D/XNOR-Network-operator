cmake_minimum_required(VERSION 2.8.8)
project(XNOR-Net)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS "--std=c++11 -fopenmp -Wall -pedantic -Wextra -Wfatal-errors -mpopcnt -fabi-version=6 -msse4 -DHAVE_SSE_INSTRUCTIONS")
set(CMAKE_CXX_FLAGS_DEBUG "-g ")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 ")

include_directories(${CMAKE_SOURCE_DIR})
add_subdirectory(test)
