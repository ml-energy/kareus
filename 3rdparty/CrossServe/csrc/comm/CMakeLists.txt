cmake_minimum_required(VERSION 3.18)

project(comm LANGUAGES CUDA CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80 86)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR}/../../../ DIRECTORY)

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

# ------------- Build MSCCLPP Test -----------------#

find_library(MSCCLPP_LIBRARY NAMES mscclpp PATHS ${MSCCLPP_DIR}/lib NO_DEFAULT_PATH)

# Define common include directories
set(COMMON_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ROOT_DIR}/csrc/comm/include
    ${ROOT_DIR}/csrc/gemm/include
    ${CUTLASS_DIR}/include
)

# Define common libraries
set(COMMON_LIBS
    MPI::MPI_CXX
    ${MSCCLPP_LIBRARY}
    spdlog::spdlog
)

# Function to configure common target properties
function(configure_target target)
    target_include_directories(${target} PRIVATE ${COMMON_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${COMMON_LIBS})
    target_compile_definitions(${target} PRIVATE -DENABLE_MPI)
endfunction()

add_executable(test_comm "${CMAKE_CURRENT_SOURCE_DIR}/comm_test.cu" "${CMAKE_CURRENT_SOURCE_DIR}/comm.cu")
configure_target(test_comm)

add_executable(test_mscclpp "${CMAKE_CURRENT_SOURCE_DIR}/mscclpp_example.cu" "${CMAKE_CURRENT_SOURCE_DIR}/include/msccl_init.cuh" "${CMAKE_CURRENT_SOURCE_DIR}/comm.cu")
configure_target(test_mscclpp)

add_executable(test_mscclpp_all2allv "${CMAKE_CURRENT_SOURCE_DIR}/test_all2all/all2allv_test.cu" "${CMAKE_CURRENT_SOURCE_DIR}/include/msccl_init.cuh" "${CMAKE_CURRENT_SOURCE_DIR}/comm.cu")
configure_target(test_mscclpp_all2allv)

add_executable(test_cached_mscclpp_all2allv "${CMAKE_CURRENT_SOURCE_DIR}/test_all2all/test_cached_all2allv.cu" "${CMAKE_CURRENT_SOURCE_DIR}/include/msccl_init.cuh" "${CMAKE_CURRENT_SOURCE_DIR}/comm.cu")
configure_target(test_cached_mscclpp_all2allv)

# ------------- Build NCCL Test -----------------#
# nm csrc/comm/build/libcustom_nccl_all2all.so | grep ncclAllToAllv
add_library(custom_nccl_all2all SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/nccl/all2all_kernel.cu"
)
target_include_directories(custom_nccl_all2all
    PRIVATE ${PARENT_DIR}/3rdparty/nccl-tests/src
)
target_link_libraries(custom_nccl_all2all PUBLIC nccl)

# Test executable
add_executable(test_nccl_all2all
    "${CMAKE_CURRENT_SOURCE_DIR}/nccl/nccl_comm_all2all_test.cu"
)
target_include_directories(test_nccl_all2all PRIVATE ${PARENT_DIR}/3rdparty/nccl-tests/src)
target_link_libraries(test_nccl_all2all
    PRIVATE spdlog::spdlog
    PRIVATE custom_nccl_all2all
    PRIVATE MPI::MPI_CXX
    PRIVATE nccl
)

# ------------- Build Pybind -----------------#

# pybind
find_package(CUDA REQUIRED)
add_subdirectory(${ROOT_DIR}/3rdparty/pybind11 pybind11)

# Create a separate CUDA library first
add_library(msccl_comm_cuda STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/mscclpp_example.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/comm.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/msccl_init.cuh"
)
target_include_directories(msccl_comm_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${CUTLASS_DIR}/include
    ${ROOT_DIR}/csrc/gemm/include
)
target_link_libraries(msccl_comm_cuda PRIVATE
    ${CUDA_LIBRARIES}
    ${MSCCLPP_LIBRARY}
    MPI::MPI_CXX
    spdlog::spdlog
)

# Create the pybind11 module with just the binding code
pybind11_add_module(msccl_comm
    "${CMAKE_CURRENT_SOURCE_DIR}/msccl_bindings.cpp"  # Change back to .cpp
)

# Link the CUDA library to the pybind11 module
target_link_libraries(msccl_comm PRIVATE
    msccl_comm_cuda
    ${CUDA_LIBRARIES}
    ${MSCCLPP_LIBRARY}
    MPI::MPI_CXX
    spdlog::spdlog
    ${TORCH_LIBRARIES}
)
target_include_directories(msccl_comm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${CUTLASS_DIR}/include
    ${ROOT_DIR}/csrc/gemm/include
    ${TORCH_INCLUDE_DIRS}
)
