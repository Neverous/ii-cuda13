/* 2013
 * Maciej Szeptuch
 * II UWr
 * ----------
 *  ...
 */
#include "defines.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <cfloat>
#include <memory>
#include <random>
#include "profiler.h"

#if __CPU__
    #include "cpu/matrix.h"
#endif // __CPU__

#if __CUDA__
    #include "cuda/matrix.h"
#endif // __CUDA__

bool valid(float *A, float *B, int size);

int main(int argc, char *argv[])
{
    std::mt19937 rng;
    rng.seed(42);

    std::uniform_real_distribution<float> rand(-10.00f, 10.00f);

    if(argc < 2)
    {
        fprintf(stderr, "usage: %s size\n", argv[0]);
        return 1;
    }

    int size   = atoi(argv[1]);
    std::unique_ptr<float> A(new float[size * size]);
    std::unique_ptr<float> B(new float[size * size]);
    std::unique_ptr<float> BT(new float[size * size]);
    std::unique_ptr<float> C(new float[size * size]);
    std::unique_ptr<float> R(new float[size * size]);

    for(int s = 0; s < size * size; ++ s)
    {
        A.get()[s] = rand(rng);
        B.get()[s] = rand(rng);
    }

#if __CPU__
    for(int h = 0; h < size; ++ h)
        for(int w = 0; w < size; ++ w)
            BT.get()[h * size + w] = B.get()[w * size + h];

    CPU_FastMul(A.get(), BT.get(), R.get(), size);
#endif // __CPU__

#if __CUDA__
    GPU_NormalMul(A.get(), B.get(), C.get(), size);

    for(int t = 0; t < TESTS; ++ t)
    {
        memset(C.get(), 0, sizeof(float) * size * size);
        {
            TGUARD("GPU_NormalMul");
            GPU_NormalMul(A.get(), B.get(), C.get(), size);
        }

        if(!valid(C.get(), R.get(), size))
            printf("ERROR: Invalid result\n");
    }

    for(int t = 0; t < TESTS; ++ t)
    {
        memset(C.get(), 0, sizeof(float) * size * size);
        {
            TGUARD("GPU_FastMul");
            GPU_FastMul(A.get(), B.get(), C.get(), size);
        }

        if(!valid(C.get(), R.get(), size))
            printf("ERROR: Invalid result\n");
    }

    for(int t = 0; t < TESTS; ++ t)
    {
        memset(C.get(), 0, sizeof(int) * size * size);
        {
            TGUARD("GPU_FastestMul");
            GPU_FastestMul(A.get(), B.get(), C.get(), size);
        }

        if(!valid(C.get(), R.get(), size))
            printf("ERROR: Invalid result\n");
    }
#endif // __CUDA__

#if __CPU__
    CPU_NormalMul(A.get(), B.get(), C.get(), size);

    for(int t = 0; t < TESTS; ++ t)
    {
        memset(C.get(), 0, sizeof(float) * size * size);
        {
            TGUARD("CPU_NormalMul");
            CPU_NormalMul(A.get(), B.get(), C.get(), size);
        }

        if(!valid(C.get(), R.get(), size))
            printf("ERROR: Invalid result\n");
    }

    for(int t = 0; t < TESTS; ++ t)
    {
        memset(C.get(), 0, sizeof(float) * size * size);
        memset(BT.get(), 0, sizeof(float) * size * size);
        {
            for(int h = 0; h < size; ++ h)
                for(int w = 0; w < size; ++ w)
                    BT.get()[h * size + w] = B.get()[w * size + h];

            TGUARD("CPU_FastMul");
            CPU_FastMul(A.get(), BT.get(), C.get(), size);
        }

        if(!valid(C.get(), R.get(), size))
            printf("ERROR: Invalid result\n");
    }
#endif // __CPU__

    return 0;
}

bool valid(float *A, float *B, int size)
{
#if __CPU__
    for(int h = 0; h < size; ++ h)
        for(int w = 0; w < size; ++ w)
            if(abs(A[h * size + w] - B[h * size + w]) > 0.000001)
                return false;
#endif // __CPU__

    return true;
}
