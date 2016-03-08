PROJECT(trafficsim)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(__CPU__     FALSE)
SET(__CUDA__    TRUE)

# VERSION
SET(VERSION_MAJOR  0)
SET(VERSION_MINOR  2)
SET(VERSION_PATCH  0)
SET(VERSION_FULL   "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

# ADDITIONAL CMAKE MODULES
SET(CMAKE_MODULE_PATH                   ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# BASE C COMPILER FLAGS
SET(CMAKE_C_FLAGS                       "-pipe -Wall -Wshadow -Woverflow -Wextra -Wstrict-overflow=5")
SET(CMAKE_C_FLAGS_RELEASE               "-O3 -DNDEBUG")
SET(CMAKE_C_FLAGS_DEBUG                 "-g -O0 -Werror")

# BASE CXX COMPILER FLAGS
SET(CMAKE_CXX_FLAGS                     "-pipe -Wall -Wshadow -Woverflow -Wextra -Wstrict-overflow=5 -std=c++11")
SET(CMAKE_CXX_FLAGS_RELEASE             "-O3 -DNDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG               "-g -O0 -Werror")

SET(CMAKE_AUTOMOC                       TRUE)
SET(CMAKE_INCLUDE_CURRENT_DIR           TRUE)
SET(CMAKE_SKIP_BUILD_RPATH              FALSE)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH   TRUE)
SET(CMAKE_EXPORT_COMPILE_COMMANDS       TRUE)

FIND_PACKAGE(Protobuf)
FIND_PACKAGE(OpenGL     REQUIRED)

IF(__CUDA__)
    FIND_PACKAGE(CUDA)
    IF(NOT CUDA_FOUND)
        MESSAGE(WARNING "Couldn't find CUDA headers, no support for GPU workers!")
        SET(__CUDA__    FALSE)
        SET(__CPU__     TRUE)
    ENDIF(NOT CUDA_FOUND)
ENDIF(__CUDA__)

FIND_PACKAGE(Qt5Widgets REQUIRED)
FIND_PACKAGE(Qt5OpenGL  REQUIRED)

MESSAGE(STATUS "Using host compiler: ${CMAKE_CXX_COMPILER_ID} v${CMAKE_CXX_COMPILER_VERSION}")

IF(NOT PROTOBUF_FOUND)
    MESSAGE(WARNING "Couldn't find google-protobuf headers, OSM importer disabled!")
ENDIF(NOT PROTOBUF_FOUND)

# Samples common headers
IF(__CUDA__)
# BASE NVCC COMPILER FLAGS
    SET(CUDA_NVCC_FLAGS                     "-arch=sm_13")
    SET(CUDA_NVCC_FLAGS_RELEASE             "-O3 -DNDEBUG -use_fast_math")
    SET(CUDA_NVCC_FLAGS_DEBUG               "-G -g -O0")

    MESSAGE(STATUS "Using CUDA version: ${CUDA_VERSION}")

    INCLUDE_DIRECTORIES(
        # HELIOS
        /opt/cuda/samples/common/inc
        /opt/cuda/include

        # II7
        /usr/local/cuda-5.5/common/inc
        /usr/local/cuda-5.5/include

        ${CUDA_INCLUDE_DIRS}
    )
ENDIF(__CUDA__)

INCLUDE_DIRECTORIES(
    src/
    ${OPENGL_INCLUDE_DIR}
)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/src/defines.h.in ${CMAKE_CURRENT_SOURCE_DIR}/src/defines.h)

ADD_SUBDIRECTORY(src/)
