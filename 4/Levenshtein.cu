/* 2013
 * Maciej Szeptuch
 * II UWr
 * ----------
 * bez shared, pozbywanie sie jak najwiecej pamieci + loop unrolling
 * czasy okolo 10x szybciej niz na CPU.

        word       |              gpu               |              cpu               | distance
 ------------------|--------------------------------|--------------------------------|----------
  kot              | kot              [  13.373088] | kot              [ 120.616997] | 0
  czesc            | czescy           [  17.563328] | czescy           [ 182.584000] | 1
  onomatopeja      | onomatopeja      [  31.341473] | onomatopeja      [ 367.598022] | 0


 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <errno.h>

#define WORD_MAXLEN     16
#define STEP_0_THREADS  128
#define STEP_R_THREADS  128
#define __CUDA__
#define __CPU__

__device__ __host__ inline unsigned char MIN(const unsigned char a, const unsigned char b) { return a<b?a:b; }
__host__ unsigned char LevenshteinDistanceH(const unsigned char *const A, const unsigned char *const B);
__device__ unsigned char LevenshteinDistanceD(const unsigned char *const A, const unsigned char *const B);

unsigned char *loadDictionary(const char *const file, unsigned int &words, unsigned int &size);
void printHead(void);

#ifdef __CUDA__
__global__ void LevenshteinCUDA_STEP_0(const unsigned char *const dictionary, const unsigned int words, const unsigned char *const pattern, unsigned int *result);

__global__ void LevenshteinCUDA_STEP_R(const unsigned int *from, unsigned int *to, const unsigned int words);
#endif // __CUDA__

#ifdef __CPU__
unsigned int LevenshteinCPU(const unsigned char *const dictionary, const unsigned int words, const unsigned char *const pattern);
#endif // __CPU__

int main(const int argc, const char *const* argv)
{
    if(argc < 3)
    {
        fprintf(stderr, "usage: %s dictionary words...\nError: not enough arguments\n", argv[0]);
        return 1;
    }

    unsigned int dictionarySize = 0,
                 dictionaryWords = 0;
    unsigned char *dictionary = loadDictionary(argv[1], dictionaryWords, dictionarySize);

    if(!dictionary)
    {
        fprintf(stderr, "usage: %s dictionary words...\nError: loading dictionary: %s\n", argv[0], strerror(errno));
        return 2;
    }

#ifdef __CUDA__
    // GPU INIT
    unsigned char *cudaDictionary = NULL,
                  *cudaPattern = NULL;

    unsigned int *cudaResult = NULL;
    cudaMalloc(&cudaDictionary, dictionarySize * sizeof(unsigned char));
    cudaMemcpy(cudaDictionary, dictionary, dictionarySize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc(&cudaPattern, WORD_MAXLEN * sizeof(unsigned char));

    cudaMalloc(&cudaResult, dictionarySize * 2 * sizeof(unsigned int));
#endif // __CUDA__

    printHead();
    for(unsigned int a = 2; a < argc; ++ a)
    {
        unsigned int result[2] = {1 << 30, 1 << 30};
        unsigned char pattern[WORD_MAXLEN + 2] = {};
        memcpy(pattern, argv[a], strlen(argv[a]) * sizeof(unsigned char));

        printf(" %-16s | ", pattern);
#ifdef __CUDA__
        {
            // GPU TEST
            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);
            cudaEventRecord(start, NULL);

            cudaMemcpy(cudaPattern, pattern, WORD_MAXLEN * sizeof(unsigned char), cudaMemcpyHostToDevice);

            LevenshteinCUDA_STEP_0<<<(dictionaryWords + STEP_0_THREADS - 1) / STEP_0_THREADS, STEP_0_THREADS>>> (cudaDictionary, dictionaryWords, cudaPattern, cudaResult);
            for(unsigned int size = STEP_R_THREADS; size < dictionaryWords; size <<= 1)
                LevenshteinCUDA_STEP_R<<<(dictionaryWords + size - 1) / size, STEP_R_THREADS>>> (cudaResult, cudaResult, dictionaryWords);

            cudaMemcpy(result, cudaResult, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

            cudaEventRecord(end, NULL);
            cudaEventSynchronize(end);
            float gputotal = 0;
            cudaEventElapsedTime(&gputotal, start, end);

            printf("%-16s [%11.6f] | ", &dictionary[result[0] * WORD_MAXLEN], gputotal, result[1]);
        }
#endif // __CUDA__
#ifdef __CPU__
        {
            // CPU TEST
            timeval start, end;
            gettimeofday(&start, NULL);

            result[0] = LevenshteinCPU(dictionary, dictionaryWords, pattern);

            gettimeofday(&end, NULL);
            float cputotal = (end.tv_sec - start.tv_sec) * 1000.0f + (end.tv_usec - start.tv_usec) / 1000.0f;
            printf("%-16s [%11.6f] | ", dictionary + result[0] * WORD_MAXLEN, cputotal);
        }
#endif // __CPU__

        printf("%u\n", LevenshteinDistanceH(pattern, dictionary + result[0] * WORD_MAXLEN));
    }

#ifdef __CUDA__
    cudaFree(cudaDictionary);
#endif // __CUDA__
    free(dictionary);
    return 0;
}

unsigned char *loadDictionary(const char *const file, unsigned int &words, unsigned int &size)
{
    FILE *handle = fopen(file, "rb");
    if(!handle)
        return NULL;

    unsigned char *dictionary = NULL,
                  *current = NULL;
    char buffer[64] = {};
    words = 0;
    while(fgets(buffer, 64, handle))
        ++ words;

    fseek(handle, 0, SEEK_SET);
    size = words * WORD_MAXLEN;
    current = dictionary = new unsigned char[size];
    memset(dictionary, 0, size * sizeof(unsigned char));
    while(fgets((char *) current, WORD_MAXLEN + 8, handle))
    {
        current[strlen((const char *) current) - 1] = 0; // remove \n
        current[strlen((const char *) current) - 1] = 0; // remove \r
        current += WORD_MAXLEN;
    }

    fclose(handle);
    return dictionary;
}

#ifdef __CPU__
unsigned int LevenshteinCPU(const unsigned char *const dictionary, const unsigned int words, const unsigned char *const pattern)
{
    const unsigned char *word = dictionary;
    unsigned int best = 1 << 30,
                 r = 0;
    for(unsigned int w = 0; w < words; ++ w, word += WORD_MAXLEN)
    {
        unsigned int dist = LevenshteinDistanceH(pattern, word);
        if(dist < best)
        {
            best = dist;
            r = w;
        }
    }

    return r;
}
#endif // __CPU__

__host__ unsigned char LevenshteinDistanceH(const unsigned char *const A, const unsigned char *const B)
{
    unsigned char sb = strlen((const char *) B);
    unsigned char *AA = (unsigned char *) A,
                  *BB = (unsigned char *) B;

    unsigned char temp[2][WORD_MAXLEN + 1];
    unsigned char t = 1;
    for(unsigned char a = 0; a <= sb; ++ a)
        temp[0][a] = a;

    AA = (unsigned char *) A;
    for(unsigned char a = 1; *AA > 0; ++ a, t ^= 1, ++ AA)
    {
        temp[t][0] = a;
        BB = (unsigned char *) B;
        for(unsigned char b = 1; b <= sb; ++ b, ++ BB)
            temp[t][b] = MIN(temp[t ^ 1][  b  ] + 1,
                    MIN(temp[  t  ][b - 1] + 1,
                        temp[t ^ 1][b - 1] + (*AA != *BB)));
    }

    return temp[t ^ 1][sb];
}

__device__ unsigned char LevenshteinDistanceD(const unsigned char *const A, const unsigned char *const B)
{
    unsigned char sb = 0;
    unsigned char *AA = (unsigned char *) A,
                  *BB = (unsigned char *) B;

    while(*BB ++ > 0)
        ++ sb;

    unsigned char temp[2][WORD_MAXLEN + 1];
    unsigned char t = 1;
#pragma unroll
    for(unsigned char a = 0; a <= WORD_MAXLEN; ++ a)
        temp[0][a] = a;

    for(unsigned char a = 1; *AA > 0; ++ a, t ^= 1, ++ AA)
    {
        temp[t][0] = a;
        BB = (unsigned char *) B;
#pragma unroll
        for(unsigned char b = 1; b <= WORD_MAXLEN; ++ b, ++ BB)
            temp[t][b] = MIN(temp[t ^ 1][  b  ] + 1,
                    MIN(temp[  t  ][b - 1] + 1,
                        temp[t ^ 1][b - 1] + (*AA != *BB)));
    }

    return temp[t ^ 1][sb];
}

void printHead(void)
{
    printf("       word       | ");
#ifdef __CUDA__
    printf("             gpu               | ");
#endif // __CUDA__

#ifdef __CPU__
    printf("             cpu               | ");
#endif // __CPU__

    printf("distance\n");

    printf("------------------|-");
#ifdef __CUDA__
    printf("-------------------------------|-");
#endif // __CUDA__

#ifdef __CPU__
    printf("-------------------------------|-");
#endif // __CPU__

    printf("---------\n");
}

#ifdef __CUDA__
__global__ void LevenshteinCUDA_STEP_0(const unsigned char *dictionary, const unsigned int words, const unsigned char *pattern, unsigned int *result)
{
    int word = blockIdx.x * STEP_0_THREADS + threadIdx.x;
    if(word >= words)
        return;

    result[word * 2] = word;
    result[word * 2 + 1] = LevenshteinDistanceD(pattern, dictionary + word * WORD_MAXLEN);
}

__global__ void LevenshteinCUDA_STEP_R(const unsigned int *from, unsigned int *to, const unsigned int words)
{
    __shared__ unsigned int local_data[STEP_R_THREADS * 2];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    local_data[tid * 2] = from[i * 2];
    local_data[tid * 2 + 1] = from[i * 2 + 1];

    __syncthreads();
    for(unsigned int s = 1; s < blockDim.x && tid + s < words; s <<= 1)
    {
        if(tid % (2 * s) == 0 && local_data[tid * 2 + 1] > local_data[(tid + s) * 2 + 1])
        {
            local_data[tid * 2] = local_data[(tid + s) * 2];
            local_data[tid * 2 + 1] = local_data[(tid + s) * 2 + 1];
        }

        __syncthreads();
    }

    if(tid == 0)
    {
        to[blockIdx.x * 2] = local_data[0];
        to[blockIdx.x * 2 + 1] = local_data[1];
    }
}
#endif // __CUDA__

