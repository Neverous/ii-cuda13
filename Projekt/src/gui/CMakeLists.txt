ADD_LIBRARY(gui trafficSimWindow.cpp consoleWidget.cpp drawerWidget.cpp)
QT5_USE_MODULES(gui Core Widgets OpenGL)

TARGET_LINK_LIBRARIES(gui ${OPENGL_LIBRARIES} GLEW)
