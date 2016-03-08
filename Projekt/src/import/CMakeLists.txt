INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/../OSMPBF/)

ADD_EXECUTABLE(../../trafficsim-import import.cpp)
QT5_USE_MODULES(../../trafficsim-import Core)
TARGET_LINK_LIBRARIES(../../trafficsim-import osmpbf)
SET_TARGET_PROPERTIES(../../trafficsim-import PROPERTIES COMPILE_FLAGS "-Wno-shadow")
