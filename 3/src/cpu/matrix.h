#ifndef __MATRIX_CPU_H__
#define __MATRIX_CPU_H__

void CPU_NormalMul(float *A, float *B, float *C, int size);
void CPU_FastMul(float *A, float *BT, float *C, int size);

#endif // __MATRIX_CPU_H__
