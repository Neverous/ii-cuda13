IF(PROTOBUF_FOUND)
    ADD_SUBDIRECTORY(OSMPBF)
    ADD_SUBDIRECTORY(import)
ENDIF(PROTOBUF_FOUND)

ADD_SUBDIRECTORY(trafficsim)
ADD_SUBDIRECTORY(gui)

QT5_ADD_RESOURCES(resources resources.qrc)
ADD_EXECUTABLE(../trafficsim main.cpp ${resources})
QT5_USE_MODULES(../trafficsim Core Widgets OpenGL)
TARGET_LINK_LIBRARIES(../trafficsim simulation gui)
