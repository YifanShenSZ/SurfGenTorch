cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# libtorch
set(CMAKE_PREFIX_PATH ~/Software/Programming/libtorch-cpu-1.6.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(CL
    source/utility.cpp
    source/LinearAlgebra.cpp
    source/TorchSupport/utility.cpp
    source/TorchSupport/LinearAlgebra.cpp
    source/TorchSupport/chemistry.cpp
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp -m64 -xCORE-AVX2 -mtune=core-avx2 -O3 -no-prec-div -fp-model fast=2 -static-intel -parallel -ipo")
    target_link_libraries(CL "${TORCH_LIBRARIES}")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -m64 -march=core-avx2 -mtune=core-avx2 -O3")
    target_link_libraries(CL "${TORCH_LIBRARIES}")
endif ()