#include "fraktal.h"
#include "complex.h"
#include "common.h"

void CPU_Atraktor(char *image, int width, int height, float s)
{
    float hstep = 2. * s / height,
          wstep = 2. * s / width;

    int h = 0,
        w = 0;
    float hs = -s,
          ws = -s;

    for(h = 0, hs = -s; h < height; ++ h, hs += hstep)
        for(w = 0, ws = -s; w < width; ++ w, ws += wstep)
        {
            Complex res = newton(Complex_(ws, hs));
            if(complex_norm(complex_sub(res, Complex_(0., 1.))) < __FLT_EPSILON__)
                image[h * width + w] = 2;

            else if(complex_norm(complex_sub(res, Complex_(1., 0.))) < __FLT_EPSILON__)
                image[h * width + w] = 3;

            else if(complex_norm(complex_sub(res, Complex_(-1., 0.))) < __FLT_EPSILON__)
                image[h * width + w] = 4;

            else if(complex_norm(complex_sub(res, Complex_(0., -1.))) < __FLT_EPSILON__)
                image[h * width + w] = 5;

            else
                image[h * width + w] = 1;
        }
}
