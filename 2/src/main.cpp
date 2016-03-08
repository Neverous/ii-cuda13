/* 2013
 * Maciej Szeptuch
 * II UWr
 */
#include "defines.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <memory>
#include "profiler.h"
#include "ppm.h"

#if __CPU__
    #include "cpu/fraktal.h"
#endif // __CPU__

#if __ISPC__
    #include "ispc/fraktal.h"
#endif // __ISPC__

#if __CUDA__
    #include "cuda/fraktal.h"
#endif // __CUDA__

#if __OPENCL__
    #include "opencl/fraktal.h"
#endif // __OPENCL__

int main(int argc, char *argv[])
{
    if(argc < 3)
    {
        fprintf(stderr, "usage: %s width height\n", argv[0]);
        return 1;
    }

    int width   = atoi(argv[1]),
        height  = atoi(argv[2]);
    std::unique_ptr<char> image(new char[width * height]);

#if __ISPC__
    {
        memset(image.get(), 0, sizeof(char) * width * height);
        TGUARD("ISPC");
        ISPC_Atraktor(image.get(), width, height, S);
    }

    savePPM(image.get(), width, height, "ispc.ppm");
#endif // __ISPC__

#if __CUDA__
    {
        memset(image.get(), 0, sizeof(char) * width * height);
        TGUARD("CUDA");
        CUDA_Atraktor(image.get(), width, height, S);
    }

    savePPM(image.get(), width, height, "cuda.ppm");
#endif // __CUDA__

#if __OPENCL__
    {
        memset(image.get(), 0, sizeof(char) * width * height);
        TGUARD("OpenCL");
        OpenCL_Atraktor(image.get(), width, height, S);
    }

    savePPM(image.get(), width, height, "opencl.ppm");
#endif // __OPENCL__

#if __CPU__
    {
        memset(image.get(), 0, sizeof(char) * width * height);
        TGUARD("CPU");
        CPU_Atraktor(image.get(), width, height, S);
    }

    savePPM(image.get(), width, height, "cpu.ppm");
#endif // __CPU__

    return 0;
}
