cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(Sigma)

set(Torch_DIR "/mnt/f/sigma/sigma/3rdparty/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories("${TORCH_INCLUDE_DIRS}" "/mnt/f/sigma/chess/")

add_executable(Sigma main.cpp policyValueNet.cpp policyValueNet.h MCTS.cpp MCTS.h)

target_link_libraries(Sigma "${TORCH_LIBRARIES}" "/mnt/f/sigma/chess/libchess.so")

set_property(TARGET Sigma PROPERTY CXX_STANDARD 20)
