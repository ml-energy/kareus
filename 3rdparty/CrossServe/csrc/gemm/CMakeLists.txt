cmake_minimum_required(VERSION 3.26.1)

project(gemm LANGUAGES CUDA CXX)

#set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

set(CUTLASS_DIR /usr/include/cutlass CACHE PATH "Path to modified cutlass src")
set(SPDLOG_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/spdlog CACHE PATH "Path to spdlog src")
set(MSCCLPP_DIR /usr/local/mscclpp CACHE PATH "Path to mscclpp install")

find_package(MPI QUIET)
if (NOT MPI_FOUND)
    message(FATAL_ERROR "MPI not found, try apt install libopenmpi-dev")
endif()

add_subdirectory(${SPDLOG_DIR} spdlog)

include_directories(${MSCCLPP_DIR}/include)

# ------------- Build Network Test -----------------#

find_library(MSCCLPP_LIBRARY NAMES mscclpp PATHS ${MSCCLPP_DIR}/lib NO_DEFAULT_PATH)

add_executable(test_gemm "${CMAKE_CURRENT_SOURCE_DIR}/gemm_test.cu")
target_include_directories(test_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(test_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../comm/include)
target_include_directories(test_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cutlass/include)
target_link_libraries(test_gemm PRIVATE MPI::MPI_CXX)
target_link_libraries(test_gemm PRIVATE ${MSCCLPP_LIBRARY})
target_link_libraries(test_gemm PRIVATE spdlog::spdlog)
target_link_libraries(test_gemm PRIVATE nvidia-ml)
target_compile_definitions(test_gemm PRIVATE -DENABLE_MPI)
