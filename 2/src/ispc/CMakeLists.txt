ADD_CUSTOM_COMMAND(
  OUTPUT fraktal.o
  COMMAND ispc
    -O3
    -I ${CMAKE_CURRENT_SOURCE_DIR}/
    -I ${CMAKE_CURRENT_SOURCE_DIR}/../libs/
    -o fraktal.o
    -h ${CMAKE_CURRENT_SOURCE_DIR}/fraktal_ispc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/fraktal.ispc
  DEPENDS fraktal.ispc
)

ADD_LIBRARY(ispc-tasksys STATIC tasksys.cpp)
ADD_LIBRARY(ispc-fraktal STATIC fraktal.cpp fraktal.o)
TARGET_LINK_LIBRARIES(ispc-fraktal ispc-tasksys pthread)
