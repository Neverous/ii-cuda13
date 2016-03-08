#ifndef __GLOBJECTS_H__
#define __GLOBJECTS_H__

#include <GL/gl.h>

struct GLPosition
{
    GLfloat x;
    GLfloat y;

    GLPosition(GLfloat _x = 0.0f, GLfloat _y = 0.0f);
}; // class GLPosition

struct GLColor
{
    GLfloat R;
    GLfloat G;
    GLfloat B;

    GLColor(GLfloat _R = 0.0f, GLfloat _G = 0.0f, GLfloat _B = 0.0f);
}; // class GLColor

struct GLPoint: public GLPosition, public GLColor
{
    GLPoint(GLfloat _x = 0.0f, GLfloat _y = 0.0f, GLfloat _R = 0.0f, GLfloat _G = 0.0f, GLfloat _B = 0.0f);
    GLPoint &operator=(const GLPosition &position);
    GLPoint &operator=(const GLColor &color);
}; // class GLPoint

inline
GLPosition::GLPosition(GLfloat _x/* = 0.0f*/, GLfloat _y/* = 0.0f*/)
:x(_x)
,y(_y)
{
}

inline
GLColor::GLColor(GLfloat _R/* = 0.0f*/, GLfloat _G/* = 0.0f*/, GLfloat _B/* = 0.0f*/)
:R(_R)
,G(_G)
,B(_B)
{
}

inline
GLPoint::GLPoint(GLfloat _x, GLfloat _y, GLfloat _R, GLfloat _G, GLfloat _B)
:GLPosition(_x, _y)
,GLColor(_R, _G, _B)
{
}

inline
GLPoint &GLPoint::operator=(const GLPosition &position)
{
    x = position.x;
    y = position.y;
    return *this;
}

inline
GLPoint &GLPoint::operator=(const GLColor &color)
{
    R = color.R;
    G = color.G;
    B = color.B;
    return *this;
}

#endif // __GLOBJECTS_H__
