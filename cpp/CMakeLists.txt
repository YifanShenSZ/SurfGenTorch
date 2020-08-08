cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpc)

project(SurfGenTorch)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp -parallel -mkl -ipo -m64 -xCORE-AVX2 -mtune=core-avx2 -O3 -no-prec-div -fp-model fast=2 -static-intel")

# libtorch
set(CMAKE_PREFIX_PATH /home/yshen/Software/Programming/libtorch-cpu)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)

# Fortran-Library
include_directories(/home/yshen/Library/Fortran-Library/include)
link_directories(/home/yshen/Library/Fortran-Library/lib)

# Cpp-Library
add_subdirectory(Cpp-Library_v1.0.0)

include_directories(include)
add_executable(SurfGenTorch.exe
    source/utility.cpp
    source/SSAIC.cpp
    source/AbInitio.cpp
    source/net/DimRed.cpp
    source/net/FLopt.cpp
    source/net/pretrain.cpp
    source/main.cpp
)
target_link_libraries(SurfGenTorch.exe "${TORCH_LIBRARIES}" FL CL)
set_property(TARGET SurfGenTorch.exe PROPERTY CXX_STANDARD 14)
