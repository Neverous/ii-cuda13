#ifndef __VEC2_H__
#define __VEC2_H__

#include <cmath>

#ifndef __cuda__
    #define __cuda__
#endif // __cuda__

struct vec2
{
    float x;
    float y;

    __cuda__ vec2(void) {};
    __cuda__ vec2(float _x, float _y);
    __cuda__ vec2 normalized(void) const;
    __cuda__ float length(void) const;
    __cuda__ float lengthSquared(void) const;

    __cuda__ vec2 &operator+=(const vec2 &_src);
    __cuda__ vec2 operator+(const vec2 &_src) const;

    __cuda__ vec2 &operator-=(const vec2 &_src);
    __cuda__ vec2 operator-(const vec2 &_src) const;

    __cuda__ vec2 &operator*=(float _factor);
    __cuda__ vec2 operator*(float _factor) const;

    __cuda__ vec2 &operator/=(float _factor);
    __cuda__ vec2 operator/(float _factor) const;

    __cuda__ bool operator==(const vec2 &second);
    __cuda__ bool operator!=(const vec2 &second);
}; // struct vec2

inline
__cuda__ vec2::vec2(float _x, float _y)
:x(_x)
,y(_y)
{
}

inline
__cuda__ vec2 vec2::normalized(void) const
{
    vec2 copy = *this;
    float len = length();
    if(len <= 0.0000001f)
        return vec2();

    return copy / length();
}

inline
__cuda__ float vec2::length(void) const
{
    return sqrtf(lengthSquared());
}

inline
__cuda__ float vec2::lengthSquared(void) const
{
    return x * x + y * y;
}

inline
__cuda__ vec2 &vec2::operator+=(const vec2 &_src)
{
    x += _src.x;
    y += _src.y;
    return *this;
}

inline
__cuda__ vec2 vec2::operator+(const vec2 &_src) const
{
    vec2 copy = *this;
    return copy += _src;
}

inline
__cuda__ vec2 &vec2::operator-=(const vec2 &_src)
{
    x -= _src.x;
    y -= _src.y;
    return *this;
}

inline
__cuda__ vec2 vec2::operator-(const vec2 &_src) const
{
    vec2 copy = *this;
    return copy -= _src;
}

inline
__cuda__ vec2 &vec2::operator*=(float _factor)
{
    x *= _factor;
    y *= _factor;
    return *this;
}

inline
__cuda__ vec2 vec2::operator*(float _factor) const
{
    vec2 copy = *this;
    return copy *= _factor;
}

inline
__cuda__ vec2 &vec2::operator/=(float _factor)
{
    x /= _factor;
    y /= _factor;
    return *this;
}

inline
__cuda__ vec2 vec2::operator/(float _factor) const
{
    vec2 copy = *this;
    return copy /= _factor;
}

inline
__cuda__ bool vec2::operator==(const vec2 &second)
{
    return x == second.x && y == second.y;
}

inline
__cuda__ bool vec2::operator!=(const vec2 &second)
{
    return x != second.x || y != second.y;
}

/** HELPERS **/
inline
__cuda__ float cross(const vec2 &a, const vec2 &b, const vec2 &c)
{
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

inline
__cuda__ bool segmentIntersect(const vec2 &a, const vec2 &b, const vec2 &c, const vec2 &d)
{
    return  cross(a, b, c) * cross(a, b, d) <= 0.0f
        &&  cross(c, d, a) * cross(c, d, b) <= 0.0f;
}

#endif // __VEC2_H__
