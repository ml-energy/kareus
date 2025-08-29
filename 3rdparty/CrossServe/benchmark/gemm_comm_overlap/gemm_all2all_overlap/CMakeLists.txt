cmake_minimum_required(VERSION 3.26.1)

project(comm LANGUAGES CUDA CXX)

#set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

set(ROOT_DIR /workspaces/CrossServe CACHE PATH "Path to CrossServe src")
set(CUTLASS_DIR ${ROOT_DIR}/3rdparty/cutlass CACHE PATH "Path to cutlass src")
set(SPDLOG_DIR  ${ROOT_DIR}/3rdparty/spdlog CACHE PATH "Path to spdlog src")
set(MSCCLPP_DIR /usr/local/mscclpp CACHE PATH "Path to mscclpp install")

find_package(MPI QUIET)
if (NOT MPI_FOUND)
    message(FATAL_ERROR "MPI not found, try apt install libopenmpi-dev")
endif()

add_subdirectory(${SPDLOG_DIR} spdlog)

include_directories(${MSCCLPP_DIR}/include)

# ------------- Build Network Test -----------------#

find_library(MSCCLPP_LIBRARY NAMES mscclpp PATHS ${MSCCLPP_DIR}/lib NO_DEFAULT_PATH)

# Define common include directories
set(COMMON_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ROOT_DIR}/csrc/comm/include
    ${ROOT_DIR}/csrc/gemm/include
    ${CUTLASS_DIR}/include
)

# Define common libraries
set(COMMON_LIBS
    MPI::MPI_CXX
    ${MSCCLPP_LIBRARY}
    spdlog::spdlog
    nvidia-ml
)

# Function to configure common target properties
function(configure_target target)
    target_include_directories(${target} PRIVATE ${COMMON_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${COMMON_LIBS})
    target_compile_definitions(${target} PRIVATE -DENABLE_MPI)
endfunction()

# Add executables and configure them
add_executable(test_overlap "${CMAKE_CURRENT_SOURCE_DIR}/test_overlap.cu" "${ROOT_DIR}/csrc/comm/comm.cu")
configure_target(test_overlap)

add_executable(test_sequential "${CMAKE_CURRENT_SOURCE_DIR}/test_sequential.cu" "${ROOT_DIR}/csrc/comm/comm.cu")
configure_target(test_sequential)

add_executable(test_overlap_cpu "${CMAKE_CURRENT_SOURCE_DIR}/test_overlap_cpu.cu" "${ROOT_DIR}/csrc/comm/comm.cu")
configure_target(test_overlap_cpu)
