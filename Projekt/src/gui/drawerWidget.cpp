#include "defines.h"
#include "drawerWidget.h"

#include <GL/glew.h>
#include <cassert>
#include <limits>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QGLFormat>

#include "local.h"
#include "GLObjects.h"
#include "graph/graph.h"
#include "colorscheme.h"

using namespace std;
using namespace trafficsim;
using namespace trafficsim::gui;

inline
static const QGLFormat GLOptions(void)
{
    QGLFormat fmt;
    fmt.setSwapInterval(1);
    fmt.setSampleBuffers(true);
    return fmt;
}

DrawerWidget::DrawerWidget(Local &_local, QWidget *_parent)
:QGLWidget(GLOptions(), _parent)

,local(_local)
,fps()
,realtime()

,buffer()
,program()

,width(800.0f)
,height(600.0f)
,zoom(1.0f)
,rotation(QQuaternion::fromAxisAndAngle(0.0f, 0.0f, 1.0f, 0.0f))
,projection()

,eye(0.0f, 0.0f, 1.0f)
,view()

,mousePressPosition()
,mousePrevPosition()

,points(0)
{
    connect(&realtime, SIGNAL(timeout()), this, SLOT(updateGL()));
    if(format().swapInterval() == -1)
    {
        qWarning() << "VSync not available. Tearing may occur!";
        realtime.setInterval(16); // ~60fps
    }

    else
        realtime.setInterval(0); // VSync

    realtime.start();
}

DrawerWidget::~DrawerWidget(void)
{
}

void DrawerWidget::initializeGL(void)
{
    if(glewInit() != GLEW_OK)
        exit(0);

    GLColor background = GLCOLOR(Colorscheme::Black[DARK]);
    glClearColor(background.R, background.G, background.B, 1.0);

    setlocale(LC_NUMERIC, "C");
    qDebug() << "Loading shaders";
    if(!program[MAP].addShaderFromSourceFile(QGLShader::Vertex, ":/shaders/map.vertex.glsl"))
        exit(0);

    if(!program[MAP].addShaderFromSourceFile(QGLShader::Fragment, ":/shaders/map.fragment.glsl"))
        exit(0);

    if(!program[MAP].link())
        exit(0);

    if(!program[CAR].addShaderFromSourceFile(QGLShader::Vertex, ":/shaders/car.vertex.glsl"))
        exit(0);

    if(!program[CAR].addShaderFromSourceFile(QGLShader::Fragment, ":/shaders/car.fragment.glsl"))
        exit(0);

    if(!program[CAR].link())
        exit(0);

    if(!program[GUI].addShaderFromSourceFile(QGLShader::Vertex, ":/shaders/gui.vertex.glsl"))
        exit(0);

    if(!program[GUI].addShaderFromSourceFile(QGLShader::Fragment, ":/shaders/gui.fragment.glsl"))
        exit(0);

    if(!program[GUI].link())
        exit(0);

    setlocale(LC_ALL, "");

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_SCISSOR_TEST);

    glGenBuffers(5, buffer);
    loadGraph();
}

inline
void DrawerWidget::loadGraph(void)
{
    // POINTS
    {
        glBindBuffer(GL_ARRAY_BUFFER, buffer[MAP_POINTS]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLPosition) * local.graph.nodes, nullptr, GL_STATIC_DRAW);
        GLPosition *point = (GLPosition *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        for(uint32_t p = 0; p < local.graph.nodes; ++ p)
            point[p] = GLPosition(local.graph.node[p].x, local.graph.node[p].y);

        glUnmapBuffer(GL_ARRAY_BUFFER);
    }

    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer[MAP_LINES]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * local.graph.edges * 2, nullptr, GL_STATIC_DRAW);
        GLuint *point = (GLuint *) glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
        for(uint32_t e = 0, p = 0; e < local.graph.edges; ++ e)
        {
            point[p ++] = local.graph.edge[e].from;
            point[p ++] = local.graph.edge[e].to;
        }

        glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
    }

    {
        glBindBuffer(GL_ARRAY_BUFFER, buffer[MAP_ROADS]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLPosition) * local.graph.edges * 6, nullptr, GL_STATIC_DRAW);
        GLPosition *point = (GLPosition *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        for(uint32_t e = 0, p = 0; e < local.graph.edges; ++ e)
        {
            graph::Node &from   = local.graph.node[local.graph.edge[e].from];
            graph::Node &to     = local.graph.node[local.graph.edge[e].to];
            const vec2 normal   = vec2(from.y - to.y, to.x - from.x).normalized() * graph::LANE_WIDTH * local.graph.edge[e].lanes() / (local.graph.edge[e].oneway() + 1.0f);
            point[p ++] = GLPosition(from.x + normal.x, from.y + normal.y);
            point[p ++] = GLPosition(from.x - normal.x, from.y - normal.y);
            point[p ++] = GLPosition(to.x - normal.x, to.y - normal.y);
            point[p ++] = GLPosition(to.x - normal.x, to.y - normal.y);
            point[p ++] = GLPosition(to.x + normal.x, to.y + normal.y);
            point[p ++] = GLPosition(from.x + normal.x, from.y + normal.y);
        }

        glUnmapBuffer(GL_ARRAY_BUFFER);
    }

    float minX = numeric_limits<float>::max();
    float maxX = numeric_limits<float>::min();
    float minY = numeric_limits<float>::max();
    float maxY = numeric_limits<float>::min();
    for(uint32_t n = 0; n < local.graph.nodes; ++ n)
    {
        minX = min(local.graph.node[n].x, minX);
        maxX = max(local.graph.node[n].x, maxX);
        minY = min(local.graph.node[n].y, minY);
        maxY = max(local.graph.node[n].y, maxY);
    }

    eye.setX((minX + maxX) / 2.0f);
    eye.setY((minY + maxY) / 2.0f);
    updateView();

}

void DrawerWidget::paintGL(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    paintGraph();
    paintCars();
    //paintGui();
    uint64_t elapsed = fps.restart();
    if(elapsed > 40)
        qWarning() << "Drawing frame took" << elapsed << "ms";
}

inline
void DrawerWidget::paintGraph(void)
{
    glEnableVertexAttribArray(0);
    program[MAP].bind();
    program[MAP].setUniformValue("MVP", projection * view);

    if(zoom >= 0.01f)
    {
        glBindBuffer(GL_ARRAY_BUFFER, buffer[MAP_ROADS]);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, local.graph.edges * 6);
    }

    glBindBuffer(GL_ARRAY_BUFFER, buffer[MAP_POINTS]);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer[MAP_LINES]);
    glDrawElements(GL_LINES, local.graph.edges * 2, GL_UNSIGNED_INT, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
}

inline
void DrawerWidget::paintCars(void)
{
    glEnableVertexAttribArray(0);
    program[CAR].bind();
    program[CAR].setUniformValue("MVP", projection * view);

    glBindBuffer(GL_ARRAY_BUFFER, buffer[CAR_SHAPES]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLPosition) * local.car.size() * 6, nullptr, GL_STREAM_DRAW);
    GLPosition *point = (GLPosition *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    uint32_t p = 0;
    for(auto &car: local.car)
        if(car.destination != numeric_limits<uint32_t>::max())
        {
            const vec2 normal   = vec2(-car.direction.y, car.direction.x) * CAR_WIDTH / 2.0f;
            const vec2 bbox     = car.direction * CAR_LENGTH / 2.0f;
            const vec2 a        = car.position + bbox;
            const vec2 s        = car.position - bbox;
            point[p] = GLPosition(a.x + normal.x, a.y + normal.y); ++ p;
            point[p] = GLPosition(a.x - normal.x, a.y - normal.y); ++ p;
            point[p] = GLPosition(s.x - normal.x, s.y - normal.y); ++ p;
            point[p] = GLPosition(s.x - normal.x, s.y - normal.y); ++ p;
            point[p] = GLPosition(s.x + normal.x, s.y + normal.y); ++ p;
            point[p] = GLPosition(a.x + normal.x, a.y + normal.y); ++ p;
        }

    assert(p <= local.car.size() * 6);
    glUnmapBuffer(GL_ARRAY_BUFFER);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glDrawArrays(GL_TRIANGLES, 0, p);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
}

void DrawerWidget::resizeGL(int _width, int _height)
{
    qDebug() << "Resizing GL viewport";
    width   = _width;
    height  = _height;
    updateViewport();
    qDebug() << "Resized GL window";
}

void DrawerWidget::mousePressEvent(QMouseEvent *_event)
{
    switch(_event->button())
    {
        case Qt::RightButton:
        case Qt::LeftButton:
            mousePrevPosition = mousePressPosition = QVector2D(_event->localPos());
            break;

        case Qt::MiddleButton:
            break;

        default:
            return;
    }

    _event->accept();
}


void DrawerWidget::mouseReleaseEvent(QMouseEvent *_event)
{
    QVector2D diff = QVector2D(_event->localPos()) - mousePressPosition;
    switch(_event->button())
    {
        case Qt::LeftButton:
            if(diff.length() < 0.1f)
            {
                // TODO: car selection
            }

            break;

        case Qt::MiddleButton:
            rotation = QQuaternion::fromAxisAndAngle(0.0f, 0.0f, 1.0f, 0.0f);
            updateViewport();
            break;

        default:
            return;
    }

    _event->accept();
}

void DrawerWidget::mouseMoveEvent(QMouseEvent *_event)
{
    QVector2D mouseCurPosition = QVector2D(_event->localPos());
    if(_event->buttons() & Qt::LeftButton)
    {
        QVector2D diff = rotation.rotatedVector(mouseCurPosition - mousePrevPosition).toVector2D() / zoom;
        diff.setX(diff.x() * -1);
        eye += diff;
        updateView();
    }

    if(_event->buttons() & Qt::RightButton)
    {
        rotation = QQuaternion::fromAxisAndAngle(0.0f, 0.0f, 1.0f,
            QLineF(mousePressPosition.toPointF(), mouseCurPosition.toPointF()).angle() -
            QLineF(mousePressPosition.toPointF(), mousePrevPosition.toPointF()).angle()) * rotation;

        updateViewport();
    }

    mousePrevPosition = mouseCurPosition;
    _event->accept();
}

void DrawerWidget::wheelEvent(QWheelEvent *_event)
{
    QPoint  pixels  = _event->pixelDelta();
    QPoint  degrees = _event->angleDelta() / 8;
    float   step    = 0.0f;

    if(!pixels.isNull())
        step = pixels.y();

    else if(!degrees.isNull())
        step = degrees.y() / 15.0f;

    zoom = min(10.0f, max(0.00001f, zoom * powf(1.25f, step)));
    updateViewport();
    _event->accept();
}

inline
void DrawerWidget::updateViewport(void)
{
    glViewport(0, 0, width, height);
    glScissor(0, 0, width, height);
    float wres = width / 2.0f / zoom;
    float hres = height / 2.0f / zoom;
    projection.setToIdentity();
    projection.ortho(-wres, wres, -hres, hres, 0.0f, 2.0f);
    projection.rotate(rotation);
}

inline
void DrawerWidget::updateView()
{
    view.setToIdentity();
    view.lookAt(eye, eye.toVector2D(), QVector2D(0.0f, 1.0f));
}
