#ifndef __COMMON_H__
#define __COMMON_H__

#include "complex.h"

#ifndef _cuda
    #define _cuda
#endif // _cuda

#define MAX_STEPS 2048

inline _cuda
Complex newton(Complex start)
{
    Complex prev = Complex_(0., 0.);
    for(int s = 0; complex_norm(complex_sub(prev, start)) > __FLT_EPSILON__ && s < MAX_STEPS; ++ s)
    {
        Complex step = complex_div(complex_add(complex_mul(start, 3), complex_pow(complex_inv(start), 3)), 4);
        prev = start;
        start = step;
    }

    return start;
}

#endif // __COMMON_H__
