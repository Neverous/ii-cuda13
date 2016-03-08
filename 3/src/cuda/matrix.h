#ifndef __CUDA_MATRIX_H__
#define __CUDA_MATRIX_H__

void GPU_NormalMul(float *A, float *B, float *C, int size);
void GPU_FastMul(float *A, float *B, float *C, int size);
void GPU_FastestMul(float *A, float *B, float *C, int size);

#endif // __CUDA_MATRIX_H__
