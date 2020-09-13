cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(libSGT)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# libtorch
set(CMAKE_PREFIX_PATH ~/Software/Programming/libtorch-cpu-1.6.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)

# Cpp-Library
include_directories(../Cpp-Library_v1.0.0)
set(CLSOURCE
    ../Cpp-Library_v1.0.0/source/utility.cpp
    ../Cpp-Library_v1.0.0/source/LinearAlgebra.cpp
    ../Cpp-Library_v1.0.0/source/TorchSupport/utility.cpp
    ../Cpp-Library_v1.0.0/source/TorchSupport/LinearAlgebra.cpp
    ../Cpp-Library_v1.0.0/source/TorchSupport/InternalCoordinate.cpp
    ../Cpp-Library_v1.0.0/source/TorchSupport/chemistry.cpp)

include_directories(../include)
set(SOURCE
    ../source/SSAIC.cpp
    ../source/DimRed.cpp
    ../source/Hd.cpp
    libSGT.cpp)
add_library(SGT SHARED ${CLSOURCE} ${SOURCE})

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp -m64 -xCORE-AVX2 -mtune=core-avx2 -O3 -no-prec-div -fp-model fast=2 -static-intel -parallel -ipo -mkl")
    target_link_libraries(SGT "${TORCH_LIBRARIES}" CL)
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -m64 -march=core-avx2 -mtune=core-avx2 -O3")
    link_directories(${MKLROOT}/lib/intel64)
    target_link_libraries(SGT "${TORCH_LIBRARIES}" CL)
endif ()