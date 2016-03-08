ADD_LIBRARY(simulation  simulation.cpp)
QT5_USE_MODULES(simulation  Core Gui)

IF(__CPU__)
    ADD_SUBDIRECTORY(cpu)
    TARGET_LINK_LIBRARIES(simulation cpu-worker)
ENDIF(__CPU__)

IF(__CUDA__)
    ADD_SUBDIRECTORY(cuda)
    TARGET_LINK_LIBRARIES(simulation cuda-worker)
ENDIF(__CUDA__)
