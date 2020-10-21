cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(data-view)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# libtorch
set(CMAKE_PREFIX_PATH ~/Software/Programming/libtorch-cuda10.1-1.6.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)

# Cpp-Library
include_directories(Cpp-Library_v1.0.0)
add_subdirectory(Cpp-Library_v1.0.0)

include_directories(../../include)
add_executable(data-view.exe
    ../../source/SSAIC.cpp
    ../../source/DimRed.cpp
    ../../source/observable_net.cpp
    ../../source/AbInitio.cpp
    main.cpp
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp -m64 -xCORE-AVX2 -mtune=core-avx2 -O3 -no-prec-div -fp-model fast=2 -static-intel -parallel -ipo -mkl")
    target_link_libraries(data-view.exe "${TORCH_LIBRARIES}" CL)
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -m64 -march=core-avx2 -mtune=core-avx2 -O3")
    link_directories(${MKLROOT}/lib/intel64)
    target_link_libraries(data-view.exe "${TORCH_LIBRARIES}" CL)
endif ()