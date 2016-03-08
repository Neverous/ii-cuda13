INCLUDE_DIRECTORIES(libs/ ./)

ADD_EXECUTABLE(fraktal main.cpp)

IF(__CPU__)
    ADD_SUBDIRECTORY(cpu/)
    TARGET_LINK_LIBRARIES(fraktal cpu-fraktal)
ENDIF(__CPU__)

IF(__ISPC__)
    ADD_SUBDIRECTORY(ispc/)
    TARGET_LINK_LIBRARIES(fraktal ispc-fraktal)
ENDIF(__ISPC__)

IF(__CUDA__)
    ADD_SUBDIRECTORY(cuda/)
    TARGET_LINK_LIBRARIES(fraktal cuda-fraktal)
ENDIF(__CUDA__)

IF(__OPENCL__)
    ADD_SUBDIRECTORY(opencl/)
    TARGET_LINK_LIBRARIES(fraktal opencl-fraktal)
ENDIF(__OPENCL__)
