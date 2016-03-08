PROJECT(addmatrix)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA)

# BASE C COMPILER FLAGS
SET(CMAKE_C_FLAGS                       "-pipe -Wall -Wshadow -Woverflow -Wextra -Wstrict-overflow=5")
SET(CMAKE_C_FLAGS_RELEASE               "-O3 -DNDEBUG")
SET(CMAKE_C_FLAGS_DEBUG                 "-g -O0 -Werror")

# BASE CXX COMPILER FLAGS
SET(CMAKE_CXX_FLAGS                     "-pipe -Wall -Wshadow -Woverflow -Wextra -Wstrict-overflow=5")
SET(CMAKE_CXX_FLAGS_RELEASE             "-O3 -DNDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG               "-g -O0 -Werror")

MESSAGE(STATUS "Using host compiler: ${CMAKE_CXX_COMPILER_ID} v${CMAKE_CXX_COMPILER_VERSION}")
MESSAGE(STATUS "Using CUDA version: ${CUDA_VERSION}")

CUDA_ADD_EXECUTABLE(addmatrix addmat.cu)
