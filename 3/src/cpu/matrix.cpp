#include "defines.h"
#include "matrix.h"

void CPU_NormalMul(float* A, float* B, float* C, int size)
{
    for (int i = 0; i < size; ++ i)
        for (int j = 0; j < size; ++ j) {
            float sum = 0;
            for(int k = 0; k < size; ++ k)
                sum +=  A[i * size + k]  * B[k * size + j];

            C[i * size + j] = sum;
        }
}

void CPU_FastMul(float* A, float* BT, float* C, int size)
{
    for (int i = 0; i < size; ++ i)
        for (int j = 0; j < size; ++ j) {
            float sum = 0;
            for(int k = 0; k < size; ++ k)
                sum +=  A[i * size + k]  * BT[j * size + k];

            C[i * size + j] = sum;
        }
}
