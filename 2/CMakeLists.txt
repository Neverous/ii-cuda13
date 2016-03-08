PROJECT(fraktal)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA REQUIRED)

# BASE C COMPILER FLAGS
SET(CMAKE_C_FLAGS                       "-pipe -Wall -Wshadow -Woverflow -Wextra -Wstrict-overflow=5")
SET(CMAKE_C_FLAGS_RELEASE               "-O3 -DNDEBUG")
SET(CMAKE_C_FLAGS_DEBUG                 "-g -O0 -Werror")

# BASE CXX COMPILER FLAGS
SET(CMAKE_CXX_FLAGS                     "-pipe -Wall -Wshadow -Woverflow -Wextra -Wstrict-overflow=5 -std=c++11")
SET(CMAKE_CXX_FLAGS_RELEASE             "-O3 -DNDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG               "-g -O0 -Werror")

# BASE NVCC COMPILER FLAGS
SET(CUDA_NVCC_FLAGS                     "-arch=sm_13")
SET(CUDA_NVCC_FLAGS_RELEASE             "-O3 -DNDEBUG -use_fast_math")
SET(CUDA_NVCC_FLAGS_DEBUG               "-G -g -O0")

MESSAGE(STATUS "Using host compiler: ${CMAKE_CXX_COMPILER_ID} v${CMAKE_CXX_COMPILER_VERSION}")
MESSAGE(STATUS "Using CUDA version: ${CUDA_VERSION}")

SET(__CPU__     TRUE)
SET(__ISPC__    TRUE)
SET(__CUDA__    TRUE)
SET(__OPENCL__  TRUE)

# Samples common headers
INCLUDE_DIRECTORIES(
    # HELIOS
    /opt/cuda/samples/common/inc
    /opt/cuda/include

    # II7
    /usr/local/cuda-5.5/common/inc
    /usr/local/cuda-5.5/include
)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/src/defines.h.in ${CMAKE_CURRENT_SOURCE_DIR}/src/defines.h)

ADD_SUBDIRECTORY(src/)
