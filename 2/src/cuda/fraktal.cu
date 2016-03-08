#define _cuda __device__

#include "fraktal.h"
#include "common.h"

__global__ void fraktal_cuda(char *image, int width, int height, float s)
{
    float hstep = 2. * s / height,
          wstep = 2. * s / width;

    int h = blockIdx.x,
        w = threadIdx.x * width / blockDim.x,
        end = (threadIdx.x + 1) * width / blockDim.x;

    while(w < end)
    {
        float hs = -s + hstep * h,
              ws = -s + wstep * w;

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

        ++ w;
    }
}

void CUDA_Atraktor(char *image, int width, int height, float s)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int threads = min(deviceProp.maxThreadsPerBlock, width);
    int size = sizeof(char) * width * height;
    char *gpuImage = NULL;
    cudaMalloc(&gpuImage, size);
    cudaMemcpy(gpuImage, image, size, cudaMemcpyHostToDevice);

    fraktal_cuda<<<height, threads>>>(gpuImage, width, height, s);

    cudaMemcpy(image, gpuImage, size, cudaMemcpyDeviceToHost);
    cudaFree(gpuImage);
}
