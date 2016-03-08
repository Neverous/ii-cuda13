#ifndef __COMPLEX_H__
#define __COMPLEX_H__

#ifdef __cplusplus
    #include <cmath>
#endif // __cplusplus

#ifndef _cuda
    #define _cuda
#endif // _cuda

typedef float Base;

struct Complex
{
    Base x,
         y;
};

inline _cuda
Complex Complex_(Base x, Base y)
{
    Complex ret;
    ret.x = x;
    ret.y = y;
    return ret;
}

inline _cuda
Complex complex_sub(Complex self, Complex sub)
{
    self.x -= sub.x;
    self.y-= sub.y;
    return self;
}

inline _cuda
Complex complex_add(Complex self, Complex add)
{
    self.x += add.x;
    self.y += add.y;
    return self;
}

inline _cuda
Complex complex_mul(Complex self, Complex mul)
{
    Complex ret;
    ret.x = self.x * mul.x - self.y * mul.y;
    ret.y = self.y * mul.x + self.x * mul.y;
    return ret;
}

inline _cuda
Complex complex_mul(Complex self, Base mul)
{
    self.x *= mul;
    self.y *= mul;
    return self;
}

inline _cuda
Complex complex_div(Complex self, Base div)
{
    self.x /= div;
    self.y /= div;
    return self;
}

inline _cuda
Complex complex_pow(Complex self, int _pow)
{
    Complex ret = self;
    while(-- _pow > 0)
        ret = complex_mul(ret, self);

    return ret;
}

inline _cuda
Complex complex_inv(Complex self)
{
    Complex ret = complex_div(self, self.x * self.x + self.y * self.y);
    ret.y *= -1;
    return ret;
}

inline _cuda
Base complex_norm(Complex self)
{
    return sqrt(self.x * self.x + self.y * self.y);
}

#endif // __COMPLEX_H__

