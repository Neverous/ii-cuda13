INCLUDE_DIRECTORIES(libs/ ./)

ADD_EXECUTABLE(matrix main.cpp)

IF(__CPU__)
    ADD_SUBDIRECTORY(cpu/)
    TARGET_LINK_LIBRARIES(matrix cpu-matrix)
ENDIF(__CPU__)

IF(__CUDA__)
    ADD_SUBDIRECTORY(cuda/)
    TARGET_LINK_LIBRARIES(matrix cuda-matrix)
ENDIF(__CUDA__)
