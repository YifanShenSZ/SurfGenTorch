cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(libSGT)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# libtorch
set(CMAKE_PREFIX_PATH ~/Software/Programming/libtorch-cuda10.1-1.7.1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)

# Cpp-Library
set (CLROOT ~/Library/Cpp-Library)
include_directories(${CLROOT})
set(CLSOURCE
    ${CLROOT}/source/utility.cpp
    ${CLROOT}/source/LinearAlgebra.cpp
    ${CLROOT}/source/TorchSupport/utility.cpp
    ${CLROOT}/source/TorchSupport/LinearAlgebra.cpp
    ${CLROOT}/source/TorchSupport/InternalCoordinate.cpp
    ${CLROOT}/source/TorchSupport/chemistry.cpp)

set (SGTROOT ~/Software/Mine/SurfGenTorch)
include_directories(${SGTROOT}/include)
set(SOURCE
    ${SGTROOT}/source/SSAIC.cpp
    ${SGTROOT}/source/DimRed.cpp
    ${SGTROOT}/source/observable_net.cpp
    ${SGTROOT}/source/Hd.cpp
    libSGT.cpp)
add_library(SGT STATIC ${CLSOURCE} ${SOURCE})
add_library(SGT-dynamic SHARED ${CLSOURCE} ${SOURCE})

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -qopenmp -parallel -static-intel -ipo")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")
endif ()
target_link_libraries(SGT "${TORCH_LIBRARIES}")
target_link_libraries(SGT-dynamic "${TORCH_LIBRARIES}")