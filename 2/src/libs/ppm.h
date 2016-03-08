#ifndef __PPM_H__
#define __PPM_H__

#include <cassert>
#include <cstdio>
#include "profiler.h"

inline
void savePPM(char *image, int width, int height, const char * const filename)
{
    TGUARD("SavePPM");
    static const unsigned char COLORMAP[6][3] = {
        {255,   255,    255},
        {29,    37,     25},
        {217,   253,    206},
        {173,   162,    138},
        {76,    108,    75},
        {43,    193,    59},
    };
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for(int h = 0; h < height; ++ h)
        for(int w = 0; w < width; ++ w)
        {
            assert(image[h * width + w] > 0 && image[h * width + w] < 6);
            for (int j = 0; j < 3; ++j)
                fputc(COLORMAP[(int) image[h * width + w]][j], fp);
        }

    fclose(fp);
}

#endif // __PPM_H__
