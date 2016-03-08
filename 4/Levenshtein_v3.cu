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
#define STEP_0_THREADS  128
#define STEP_R_THREADS  128
#define __CUDA__
#define __CPU__

__device__ __host__ inline int MIN(const int a, const int b) { return a<b?a:b; }
__device__ __host__ int LevenshteinDistance(const char *const A, const char *const B);

char *loadDictionary(const char *const file, int &words, int &size);
void printHead(void);

#ifdef __CUDA__
    __global__ void LevenshteinCUDA_STEP_0(const char *const dictionary, const int words, const char *const pattern, int *result);

    __global__ void LevenshteinCUDA_STEP_R(const int *from, int *to, const int words);
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

    int  *cudaResult = NULL;

    int alignedDictionarySize = 1;
    while(alignedDictionarySize < dictionarySize)
        alignedDictionarySize <<= 1;

    cudaMalloc(&cudaDictionary, alignedDictionarySize * sizeof(char));
    cudaMemcpy(cudaDictionary, dictionary, dictionarySize * sizeof(char), cudaMemcpyHostToDevice);

    cudaMalloc(&cudaPattern, WORD_MAXLEN * sizeof(char));

    cudaMalloc(&cudaResult, alignedDictionarySize * 2 * sizeof(int));
#endif // __CUDA__

    printHead();
    for(int a = 2; a < argc; ++ a)
    {
        int result[2] = {1 << 30, 1 << 30};
        char pattern[WORD_MAXLEN + 2] = {};
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
            for(int size = STEP_R_THREADS; size < dictionaryWords; size <<= 1)
                LevenshteinCUDA_STEP_R<<<(dictionaryWords + size - 1) / size, STEP_R_THREADS>>> (cudaResult, cudaResult, dictionaryWords);

            cudaMemcpy(result, cudaResult, 2 * sizeof(int), cudaMemcpyDeviceToHost);

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

        printf("%d\n", LevenshteinDistance(pattern, dictionary + result[0] * WORD_MAXLEN));
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
        int dist = LevenshteinDistance(pattern, word);
        if(dist < best)
        {
            best = dist;
            r = w;
        }
    }

    return r;
}
#endif // __CPU__

__device__ __host__ int LevenshteinDistance(const char *const A, const char *const B)
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
    int word = blockIdx.x * STEP_0_THREADS + threadIdx.x;
    if(word >= words)
        return;

    result[word * 2] = word;
    result[word * 2 + 1] = LevenshteinDistance(pattern, dictionary + word * WORD_MAXLEN);
}

__global__ void LevenshteinCUDA_STEP_R(const int *from, int *to, const int words)
{
    __shared__ int local_data[STEP_R_THREADS * 2];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    local_data[tid * 2] = from[i * 2];
    local_data[tid * 2 + 1] = from[i * 2 + 1];

    __syncthreads();
    for(int s = 1; s < blockDim.x && tid + s < words; s <<= 1)
        if(tid % (2 * s) == 0 && local_data[tid * 2 + 1] > local_data[(tid + s) * 2 + 1])
        {
            local_data[tid * 2] = local_data[(tid + s) * 2];
            local_data[tid * 2 + 1] = local_data[(tid + s) * 2 + 1];
        }

    if(tid == 0)
    {
        to[blockIdx.x * 2] = local_data[0];
        to[blockIdx.x * 2 + 1] = local_data[1];
    }
}
#endif // __CUDA__

