cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpc)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp -parallel -mkl -ipo -m64 -xCORE-AVX2 -mtune=core-avx2 -O3 -no-prec-div -fp-model fast=2 -static-intel")

# libtorch
set(CMAKE_PREFIX_PATH /home/yshen/Software/Programming/libtorch-cpu)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(CL
    source/utility.cpp
    source/LinearAlgebra.cpp
    source/TorchSupport.cpp
)

target_link_libraries(CL "${TORCH_LIBRARIES}")
set_property(TARGET CL PROPERTY CXX_STANDARD 14)