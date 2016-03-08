#ifndef __DRAWER_WIDGET_H__
#define __DRAWER_WIDGET_H__

#include <GL/glew.h>
#include <QGLWidget>
#include <QVector2D>
#include <QVector3D>
#include <QMatrix4x4>
#include <QQuaternion>
#include <QGLShaderProgram>
#include <QTimer>
#include <QElapsedTimer>

#include "local.h"

namespace trafficsim
{

namespace gui
{

class DrawerWidget: public QGLWidget
{
    Q_OBJECT

    enum Buffers
    {
        MAP_POINTS  = 0,
        MAP_LINES   = 1,
        MAP_ROADS   = 2,
        CAR_SHAPES  = 3,
        GUI_SELECT  = 4,
    }; // enum Buffers

    enum Programs
    {
        MAP = 0,
        CAR = 1,
        GUI = 2,
    }; // enum Programs

    Local               &local;

    QElapsedTimer       fps;
    QTimer              realtime;

    GLuint              buffer[5];
    QGLShaderProgram    program[3];

    float               width;
    float               height;
    float               zoom;
    QQuaternion         rotation;
    QMatrix4x4          projection;

    QVector3D           eye;
    QMatrix4x4          view;

    QVector2D           mousePressPosition;
    QVector2D           mousePrevPosition;

    unsigned int        points;

    public:
        DrawerWidget(Local &_local, QWidget *_parent);
        ~DrawerWidget(void);

    protected:
        void initializeGL(void);
        void loadGraph(void);
        void paintGL(void);
        void paintGraph(void);
        void paintCars(void);
        void resizeGL(int _width, int _height);

        void mousePressEvent(QMouseEvent *_event);
        void mouseReleaseEvent(QMouseEvent *_event);
        void mouseMoveEvent(QMouseEvent *_event);
        void wheelEvent(QWheelEvent *_event);

    private:
        void updateViewport(void);
        void updateView(void);

}; // class DrawerWidget

} // namespace gui

} // namespace trafficsim

#endif // __DRAWER_WIDGET_H__
