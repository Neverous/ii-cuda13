/* 2013
 * Maciej Szeptuch
 * II UWr
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <errno.h>

#define WORD_MAXLEN     16
#define STEP_0_THREADS  256
#define __CUDA__
#define __CPU__

__device__ __host__ inline int MIN(const int a, const int b) { return a<b?a:b; }
__host__ int LevenshteinDistanceH(const char *const A, const char *const B);

char *loadDictionary(const char *const file, int &words, int &size);
void printHead(void);

#ifdef __CUDA__
    __global__ void LevenshteinCUDA_STEP_0(const char *const dictionary, const int words, const char *const pattern, int *result);
    __device__ int LevenshteinDistanceD(const char *const A, const char *const B);
#endif // __CUDA__

#ifdef __CPU__
    int LevenshteinCPU(const char *const dictionary, const int words, const char *const pattern);
#endif // __CPU__

int main(const int argc, const char *const* argv)
{
    if(argc < 3)
    {
        fprintf(stderr, "usage: %s dictionary words...\nError: not enough arguments\n", argv[0]);
        return 1;
    }

    int dictionarySize = 0,
        dictionaryWords = 0;
    char *dictionary = loadDictionary(argv[1], dictionaryWords, dictionarySize);

    if(!dictionary)
    {
        fprintf(stderr, "usage: %s dictionary words...\nError: loading dictionary: %s\n", argv[0], strerror(errno));
        return 2;
    }

#ifdef __CUDA__
    // GPU INIT
    char *cudaDictionary = NULL,
         *cudaPattern = NULL;

    int  *cudaResult = NULL,
         *cudaTemp = NULL;

    int alignedDictionarySize = 1;
    while(alignedDictionarySize < dictionarySize)
        alignedDictionarySize <<= 1;

    cudaMalloc(&cudaDictionary, alignedDictionarySize * sizeof(char));
    cudaMemcpy(cudaDictionary, dictionary, dictionarySize * sizeof(char), cudaMemcpyHostToDevice);

    cudaMalloc(&cudaPattern, WORD_MAXLEN * sizeof(char));

    cudaMalloc(&cudaResult, alignedDictionarySize * 2 * sizeof(int));
    cudaMalloc(&cudaTemp, alignedDictionarySize * 2 * sizeof(int));
#endif // __CUDA__

    printHead();
    for(int a = 2; a < argc; ++ a)
    {
        int result[2] = {1 << 30, 1 << 30};
        char pattern[WORD_MAXLEN] = {};
        memcpy(pattern, argv[a], strlen(argv[a]) * sizeof(char));

        printf(" %-16s | ", pattern);
#ifdef __CUDA__
        {
            // GPU TEST
            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);
            cudaEventRecord(start, NULL);

            cudaMemcpy(cudaPattern, pattern, WORD_MAXLEN * sizeof(char), cudaMemcpyHostToDevice);

            LevenshteinCUDA_STEP_0<<<(dictionaryWords + STEP_0_THREADS - 1) / STEP_0_THREADS, STEP_0_THREADS>>> (cudaDictionary, dictionaryWords, cudaPattern, cudaResult);

            cudaMemcpy(result, cudaResult, 2 * sizeof(int), cudaMemcpyDeviceToHost);

            cudaEventRecord(end, NULL);
            cudaEventSynchronize(end);
            float gputotal = 0;
            cudaEventElapsedTime(&gputotal, start, end);

            printf("%-16s [%11.6f] | ", &dictionary[result[0] * WORD_MAXLEN], gputotal);
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

        printf("%d\n", LevenshteinDistanceH(pattern, dictionary + result[0] * WORD_MAXLEN));
    }

#ifdef __CUDA__
    cudaFree(cudaDictionary);
#endif // __CUDA__
    free(dictionary);
    return 0;
}

char *loadDictionary(const char *const file, int &words, int &size)
{
    FILE *handle = fopen(file, "rb");
    if(!handle)
        return NULL;

    char *dictionary = NULL,
         *current = NULL;
    char buffer[64] = {};
    words = 0;
    while(fgets(buffer, 64, handle))
        ++ words;

    fseek(handle, 0, SEEK_SET);
    size = words * WORD_MAXLEN;
    current = dictionary = new char[size];
    memset(dictionary, 0, size * sizeof(char));
    while(fgets(current, WORD_MAXLEN + 8, handle))
    {
        current[strlen(current) - 1] = 0; // remove \n
        current[strlen(current) - 1] = 0; // remove \r
        current += WORD_MAXLEN;
    }

    fclose(handle);
    return dictionary;
}

#ifdef __CPU__
int LevenshteinCPU(const char *const dictionary, const int words, const char *const pattern)
{
    const char *word = dictionary;
    int best = 1 << 30,
        r = 0;
    for(int w = 0; w < words; ++ w, word += WORD_MAXLEN)
    {
        int dist = LevenshteinDistanceH(pattern, word);
        if(dist < best)
        {
            best = dist;
            r = w;
        }
    }

    return r;
}
#endif // __CPU__

__host__ int LevenshteinDistanceH(const char *const A, const char *const B)
{
    int sa = 0,
        sb = 0;
    while(A[sa ++] > 0);
    while(B[sb ++] > 0);

    int temp[2][WORD_MAXLEN + 1] = {};
    int t = 1;
    for(int a = 0; a <= sb; ++ a)
        temp[0][a] = a;

    for(int a = 1; a <= sa; ++ a, t ^= 1)
    {
        temp[t][0] = a;
        for(int b = 1; b <= sb; ++ b)
            temp[t][b] = MIN(temp[t ^ 1][  b  ] + 1,
                         MIN(temp[  t  ][b - 1] + 1,
                             temp[t ^ 1][b - 1] + (A[a-1] != B[b-1])));
    }

    return temp[t^1][sb];
}

__device__ int LevenshteinDistanceD(const char *const A, const char *const B)
{
    int temp[2][WORD_MAXLEN + 1] = {};
    int t = 1;
#pragma unroll
    for(int a = 0; a <= WORD_MAXLEN; ++ a)
        temp[0][a] = a;

#pragma unroll
    for(int a = 1; a <= WORD_MAXLEN; ++ a, t ^= 1)
    {
        temp[t][0] = a;
#pragma unroll
        for(int b = 1; b <= WORD_MAXLEN; ++ b)
            temp[t][b] = MIN(temp[t ^ 1][  b  ] + 1,
                         MIN(temp[  t  ][b - 1] + 1,
                             temp[t ^ 1][b - 1] + (A[a-1] != B[b-1])));
    }

    return temp[t^1][WORD_MAXLEN];
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
__global__ void LevenshteinCUDA_STEP_0(const char *dictionary, const int words, const char *pattern, int *result)
{
    result[(blockIdx.x * STEP_0_THREADS + threadIdx.x) * 2] = LevenshteinDistanceD(pattern, dictionary + (blockIdx.x * STEP_0_THREADS + threadIdx.x) * WORD_MAXLEN);
    result[(blockIdx.x * STEP_0_THREADS + threadIdx.x) * 2 + 1] = blockIdx.x * STEP_0_THREADS + threadIdx.x;
}
#endif // __CUDA__

