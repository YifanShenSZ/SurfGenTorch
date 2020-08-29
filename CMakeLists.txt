cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(SurfGenTorch)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# libtorch
set(CMAKE_PREFIX_PATH ~/Software/Programming/libtorch-cpu-1.6.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)

# Fortran-Library
include_directories(~/Library/Fortran-Library/include)
link_directories(~/Library/Fortran-Library/lib)

# Cpp-Library
include_directories(Cpp-Library_v1.0.0)
add_subdirectory(Cpp-Library_v1.0.0)

include_directories(include)
add_executable(SurfGenTorch.exe
    source/SSAIC.cpp
    source/DimRed.cpp
    source/Hd.cpp
    source/AbInitio.cpp
    source/pretrain.cpp
    source/train.cpp
    source/main.cpp
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp -m64 -xCORE-AVX2 -mtune=core-avx2 -O3 -no-prec-div -fp-model fast=2 -static-intel -parallel -ipo -mkl")
    target_link_libraries(SurfGenTorch.exe "${TORCH_LIBRARIES}" FL CL)
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -m64 -march=core-avx2 -mtune=core-avx2 -O3")
    link_directories(${MKLROOT}/lib/intel64)
    target_link_libraries(SurfGenTorch.exe "${TORCH_LIBRARIES}" FL CL mkl_intel_lp64 mkl_gnu_thread mkl_core gomp pthread m dl)
endif ()