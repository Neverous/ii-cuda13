// ========================================================================
//    Kurs: Procesory graficzne w obliczeniach równoległych (lista 3)
//          A.Łukaszewski 2010
//=========================================================================
//    Kody przykładów mnożenia macierzy
//=========================================================================

#include "defines.h"

// Jądro z podziałem na bloki dla większych macierzy
__global__ void NormalMul(float *A, float *B, float *C, int size)
{
    float sum = 0;
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    for(int k = 0; k < size; ++ k)
    {
        float Ai = A[y * size + k];
        float Bi = B[k * size + x];
        sum += Ai * Bi;
    }

    C[y * size + x] = sum;
}

// Jądro z użyciem pamięci dzielonej dla zredukowania dostępów do
// pamieci globalnej
__global__ void FastMul(float *A, float *B, float *C, int size)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int ypos = by * TILE + ty;
    int xpos = bx * TILE + tx;

    float sum = 0;

    for(int m = 0; m < size / TILE; ++ m)
    {
        As[ty][tx] = A[ypos * size + (m * TILE + tx)];
        Bs[ty][tx] = B[(m * TILE + ty) * size + xpos];
        __syncthreads();

        for (int k = 0; k < TILE; ++ k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    C[ypos * size + xpos] = sum;
}

__global__ void FastestMul(float *A, float *B, float *C, int size)
{
    __shared__ float As[2][TILE][TILE];
    __shared__ float Bs[2][TILE][TILE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int ypos = by * TILE + ty;
    int xpos = bx * TILE + tx;

    float sum = 0;

    for(int m = 0; m < size / TILE; ++ m)
    {
        As[m&1][ty][tx] = A[ypos * size + (m * TILE + tx)];
        Bs[m&1][ty][tx] = B[(m * TILE + ty) * size + xpos];
        __syncthreads();

        for (int k = 0; k < TILE; ++ k)
            sum += As[m&1][ty][k] * Bs[m&1][k][tx];
    }

    C[ypos * size + xpos] = sum;
}

void GPU_NormalMul(float *A, float *B, float *C, int size)
{
    int bytes = size * size * sizeof(float);
    float *Ad = NULL,
          *Bd = NULL,
          *Cd = NULL;

    cudaMalloc(&Ad, bytes);
    cudaMemcpy(Ad, A, bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&Bd, bytes);
    cudaMemcpy(Bd, B, bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&Cd, bytes);

    dim3 dimGrid(size / TILE, size / TILE);
    dim3 dimBlock(TILE, TILE);
    NormalMul<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, size);

    cudaMemcpy(C, Cd, bytes, cudaMemcpyDeviceToHost);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
}

void GPU_FastMul(float *A, float *B, float *C, int size)
{
    int bytes = size * size * sizeof(float);
    float *Ad = NULL,
          *Bd = NULL,
          *Cd = NULL;

    cudaMalloc(&Ad, bytes);
    cudaMemcpy(Ad, A, bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&Bd, bytes);
    cudaMemcpy(Bd, B, bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&Cd, bytes);

    dim3 dimGrid(size / TILE, size / TILE);
    dim3 dimBlock(TILE, TILE);
    FastMul<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, size);

    cudaMemcpy(C, Cd, bytes, cudaMemcpyDeviceToHost);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
}

void GPU_FastestMul(float *A, float *B, float *C, int size)
{
    int bytes = size * size * sizeof(float);
    float *Ad = NULL,
          *Bd = NULL,
          *Cd = NULL;

    cudaMalloc(&Ad, bytes);
    cudaMemcpy(Ad, A, bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&Bd, bytes);
    cudaMemcpy(Bd, B, bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&Cd, bytes);

    dim3 dimGrid(size / TILE, size / TILE);
    dim3 dimBlock(TILE, TILE);
    FastestMul<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, size);

    cudaMemcpy(C, Cd, bytes, cudaMemcpyDeviceToHost);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
}
