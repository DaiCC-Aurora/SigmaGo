cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(chess)

find_package(Torch REQUIRED)
include_directories("${TORCH_INCLUDE_DIRS}")

add_library(chess SHARED chess.cpp chess.h)

target_link_libraries(chess "${TORCH_LIBRARIES}")
set_property(TARGET chess PROPERTY CXX_STANDARD 17)
