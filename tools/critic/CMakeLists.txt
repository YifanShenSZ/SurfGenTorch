cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(critic)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# libtorch
set(CMAKE_PREFIX_PATH ~/Software/Programming/libtorch-cuda10.1-1.7.1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)

# Cpp-Library
include_directories(Cpp-Library)
add_subdirectory(Cpp-Library)

# libSGT
include_directories(libSGT)
add_subdirectory(libSGT)

add_executable(critic.exe
    min.cpp
    sad.cpp
    mex.cpp
    main.cpp
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -qopenmp -parallel -static-intel -ipo")
    target_link_libraries(critic.exe SGT CL "${TORCH_LIBRARIES}" FL)
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")
    link_directories(${MKLROOT}/lib/intel64)
    target_link_libraries(critic.exe SGT CL "${TORCH_LIBRARIES}" FL mkl_intel_lp64 mkl_gnu_thread mkl_core)
endif ()