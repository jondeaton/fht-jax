cmake_minimum_required(VERSION 3.18)

# https://github.com/OpenNMT/CTranslate2/issues/1703 ???
# https://github.com/OpenNMT/CTranslate2/pull/1706 ???
# set(INTEL_ROOT_DEFAULT $ENV{MKLROOT}/../../..)
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#   set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
# endif()

# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy
# https://archlinux.org/packages/?name=gcc13
# export CC=/usr/bin/gcc-13
# export CXX=/usr/bin/g++-13
# export CUDAHOSTCXX=/usr/bin/g++-13

set(ENV{CC} /usr/bin/gcc-13)
set(ENV{CXX} /usr/bin/g++-13)
set(ENV{CUDAHOSTCXX} /usr/bin/g++-13)

set(CMAKE_C_COMPILER /usr/bin/gcc-13)
set(CMAKE_CXX_COMPILER /usr/bin/g++-13)
set(CUDA_HOST_COMPILER /usr/bin/gcc-13)


# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# TODO: This is specifically for compilation on my 2080TI
# If you have a different architecture then you'd wanna change this.
set(CMAKE_CUDA_ARCHITECTURES 75)


project(fwht LANGUAGES C CXX CUDA)

# Find pybind11
set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# CPU op library
# pybind11_add_module(cpu_ops ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_ops.cc)
# install(TARGETS cpu_ops LIBRARY DESTINATION .)

# Include the CUDA extensions if possible
# include(CheckLanguage)
# check_language(CUDA)

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
# include_directories(/opt/cuda/include/)

# enable_language(CUDA)


# Compiler options
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/hadamard_cuda_kernel.cu
                            PROPERTIES
                            COMPILE_OPTIONS "--use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")



pybind11_add_module(
    hadamard_transform_cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hadamard_cuda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hadamard_cuda_kernel.cu
)
install(TARGETS hadamard_transform_cuda LIBRARY DESTINATION .)


