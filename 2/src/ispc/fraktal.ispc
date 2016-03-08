#include "complex.h"
#include "common.h"

task
void fraktal_ispc_subset(uniform int8 image[], uniform int width, uniform int height, uniform float s, uniform float wstep, uniform float hstep)
{
    foreach(w = 0 ... width)
    {
        float hs = -s + taskIndex * hstep,
              ws = -s + w * wstep;

        Complex res = newton(Complex_(ws, hs));
        if(complex_norm(complex_sub(res, Complex_(0., 1.))) < __FLT_EPSILON__)
            image[taskIndex * width + w] = 2;

        else if(complex_norm(complex_sub(res, Complex_(1., 0.))) < __FLT_EPSILON__)
            image[taskIndex * width + w] = 3;

        else if(complex_norm(complex_sub(res, Complex_(-1., 0.))) < __FLT_EPSILON__)
            image[taskIndex * width + w] = 4;

        else if(complex_norm(complex_sub(res, Complex_(0., -1.))) < __FLT_EPSILON__)
            image[taskIndex * width + w] = 5;

        else
            image[taskIndex * width + w] = 1;
    }
}

export
void fraktal_ispc(uniform int8 image[], uniform int width, uniform int height, uniform float s)
{
    uniform float hstep = 2. * s / height,
                  wstep = 2. * s / width;

    launch[height] fraktal_ispc_subset(image, width, height, s, wstep, hstep);
}
