/* 2013
 * Maciej Szeptuch
 * IIUWr
 */
#include <cstdlib>
#include <cstdio>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>

#define EPS     0.00001

#define cudaErr(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline
void cudaAssert(cudaError_t code, char *file, int line)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr,"%s:%d CUDA: %s(%d)\n", file, line, cudaGetErrorString(code), code);
        exit(code);
    }
}

inline int divup(int a, int b) { return (a + b - 1) / b; }

// RANDOM
__device__ unsigned int TausStep(unsigned int &z, unsigned int S1, unsigned int S2, unsigned int S3, unsigned int M);
__device__ unsigned int LCGStep(unsigned int &z, unsigned int A, unsigned int C);
__device__ float HybridTaus(unsigned int &z1, unsigned int &z2, unsigned int &z3, unsigned int &z4);
__device__ unsigned int HybridTausInt(unsigned int &z1, unsigned int &z2, unsigned int &z3, unsigned int &z4);
__device__ unsigned int rand(unsigned int salt);

// GLUT and drawing stuff
void menuDraw(void);
void cudaDraw(void);
void cudaInit(void);
void glutDisplayCallback(void);
void glutKeyboardCallback(unsigned char key, int, int);
void glutReshapeCallback(int w, int h);
void cleanup(void);
__global__ void draw(int *picture, int width, int height, float scale, int steps, int posX, int posY, float *_matrix);
__device__ __host__ void multiply(float &dx, float &dy, float sx, float sy, float *_matrix);

int     width   = 800,
        height  = 600,
        steps   = 50,
        posX    = 0,
        posY    = 0;
float   scale   = 10;
float   matrix[12] = {
    -0.40,  0.00, -1.00,
     0.00, -0.40,  0.10,

     0.76, -0.40,  0.00,
     0.40,  0.76,  0.00,
};


int     *picture;
GLuint  data;
int     *cudaData;
float   *cudaMatrix;
struct ActEdit
{
    int m;
    int x;
    int y;
}       actEdit;

dim3    blockSize(16,16);
dim3    gridSize;

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA GL IFS");
    glutDisplayFunc(glutDisplayCallback);
    glutKeyboardFunc(glutKeyboardCallback);
    glutReshapeFunc(glutReshapeCallback);

    glewInit();
    if(!glewIsSupported("GL_VERSION_2_1"))
    {
        fprintf(stderr, "OpenGL >= 2.1 required\n");
        return 2;
    }

    cudaInit();
    atexit(cleanup);
    glutMainLoop();
    return 0;
}

void cleanup(void)
{
    cudaGLUnregisterBufferObject(data);
    glDeleteBuffers(1, &data);
    delete[] picture;
    cudaFree(cudaMatrix);
}

void glutReshapeCallback(int w, int h)
{
    width = w; height = h;
    cudaInit();
    glViewport(0, 0, w, h);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void glutKeyboardCallback(unsigned char key, int, int)
{
    switch(key)
    {
        case '\e':
        case 'q':
        case 'Q':
            exit(3);
            break;

        case '\t':
            ++ actEdit.x;
            if(actEdit.x == 3)
            {
                ++ actEdit.y;
                actEdit.x = 0;
            }

            if(actEdit.y == 2)
            {
                ++ actEdit.m;
                actEdit.y = 0;
            }

            if(actEdit.m == 2)
                actEdit.m = 0;

            break;

        case '+':
            matrix[actEdit.m * 6 + actEdit.y * 3 + actEdit.x] += 0.01;
            cudaErr(cudaMemcpy(cudaMatrix, matrix, 2 * 2 * 3 * sizeof(float), cudaMemcpyHostToDevice));
            break;

        case '-':
            matrix[actEdit.m * 6 + actEdit.y * 3 + actEdit.x] -= 0.01;
            cudaErr(cudaMemcpy(cudaMatrix, matrix, 2 * 2 * 3 * sizeof(float), cudaMemcpyHostToDevice));
            break;

        case '[':
            scale += 0.1;
            break;

        case ']':
            scale -= 0.1;
            break;

        case ',':
            steps -= 1;
            break;

        case '.':
            steps += 1;
            break;

        case 'w':
            posY += 5;
            break;

        case 's':
            posY -= 5;
            break;

        case 'a':
            posX -= 5;
            break;

        case 'd':
            posX += 5;
            break;
    }

    menuDraw();
    glutPostRedisplay();
}

void glutDisplayCallback(void)
{
    menuDraw();
    cudaDraw();
    cudaThreadSynchronize();

    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glutSwapBuffers();
    glutReportErrors();
}

void cudaInit(void)
{
    if(data)
    {
        cudaGLUnregisterBufferObject(data);
        glDeleteBuffers(1, &data);
        delete[] picture;
        cudaFree(cudaMatrix);
    }

    glGenBuffers(1, &data);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, data);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW);
    picture = new int[width * height];
    memset(picture, 0, width * height * sizeof(int));
    cudaErr(cudaGLRegisterBufferObject(data));

    gridSize = dim3(divup(width, blockSize.x), divup(height, blockSize.y));
    cudaErr(cudaMalloc(&cudaMatrix, 2 * 2 * 3 * sizeof(float)));
    cudaErr(cudaMemcpy(cudaMatrix, matrix, 2 * 2 * 3 * sizeof(float), cudaMemcpyHostToDevice));
}

void cudaDraw(void)
{
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, NULL);

    cudaErr(cudaGLMapBufferObject((void **) &cudaData, data));
    cudaErr(cudaMemcpy(cudaData, picture, width * height * sizeof(int), cudaMemcpyHostToDevice));
    draw<<<gridSize, blockSize>>>(cudaData, width, height, scale, steps, posX, posY, cudaMatrix);
    cudaErr(cudaPeekAtLastError());
    cudaErr(cudaDeviceSynchronize());
    cudaErr(cudaGLUnmapBufferObject(data));

    cudaEventRecord(end, NULL);
    cudaEventSynchronize(end);
    float gputotal = 0;
    cudaEventElapsedTime(&gputotal, start, end);

    printf("========== ][ Kernel took: %5.2f ][ ==========\n", gputotal);
}

__device__ __host__ void multiply(float &dx, float &dy, float sx, float sy, float *_matrix)
{
    dx = sx * _matrix[0] + sy * _matrix[1] + _matrix[2];
    dy = sx * _matrix[3] + sy * _matrix[4] + _matrix[5];
}

__global__ void draw(int *picture, int width, int height, float scale, int steps, int posX, int posY, float *_matrix)
{
    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    int id = y * width + x;
    int salt = rand(id);

    if(x >= width || y >= height)
        return;

    float px = x - width / 2,
          py = y - height / 2,
          lx = 0.0, ly = 0.0;

    for(int t = 0; t < 32; ++ t)
    {
        multiply(px, py, px, py, _matrix + (salt < 0) * 6);
        salt = rand(salt);
    }

    for(int t = 0; t < steps; ++ t)
    {
        multiply(px, py, px, py, _matrix + (salt < 0) * 6);
        salt = rand(salt);

        if(abs(px - lx) < EPS && abs(py - ly) < EPS)
            break;

        int _x = px / scale * width + width / 2 - posX;
        int _y = py / scale * height + height / 2 - posY;
        if(0 <= _x && _x < width && 0 <= _y && _y < height)
            picture[_y * width + _x] = 0xFFFFFF;

        lx = px;
        ly = py;
    }
}

void menuDraw(void)
{
    system("clear");
    puts("========== ][ CUDA IFS ][ ==========");
    printf("Resolution: %dx%d | Position (%d, %d)\n", width, height, posX, posY);
    printf("Scale: %4.1f | Steps: %3d\n", 10. / scale, steps);
    puts("Matrices: ");
    for(int m = 0; m < 2; ++ m)
    {
        puts("");
        for(int y = 0; y < 2; ++ y)
        {
            printf("|");
            for(int x = 0; x < 3; ++ x)
            {
                if(actEdit.m == m && actEdit.y == y && actEdit.x == x)
                    printf("*%5.2f*", matrix[m * 6 + y * 3 + x]);

                else
                    printf(" %5.2f ", matrix[m * 6 + y * 3 + x]);

                if(x == 1)
                    printf("| |");
            }

            puts("|");
        }
    }

    puts("");
}

__device__ unsigned int TausStep(unsigned int &z, unsigned int S1, unsigned int S2, unsigned int S3, unsigned int M)
{
    unsigned int b = (((z << S1) ^ z) >> S2);
    return z = (((z & M) << S3) ^ b);
}

__device__ unsigned int LCGStep(unsigned int &z, unsigned int A, unsigned int C)
{
    return z = (A * z + C);
}

__device__ float HybridTaus(unsigned int &z1, unsigned int &z2, unsigned int &z3, unsigned int &z4)
{
    return 2.3283064365387e-10 * (
        TausStep(z1, 13, 19, 12, 4294967294UL) ^
        TausStep(z2,  2, 25,  4, 4294967288UL) ^
        TausStep(z3,  3, 11, 17, 4294967280UL) ^
        LCGStep( z4,    1664525, 1013904223UL)
    );
}

__device__ unsigned int HybridTausInt(unsigned int &z1, unsigned int &z2, unsigned int &z3, unsigned int &z4)
{
    return (
       TausStep(z1, 13, 19, 12, 4294967294UL) ^
       LCGStep( z4,    1664525, 1013904223UL)
   );
}

__device__ unsigned int rand(unsigned int salt)
{
    return HybridTausInt(salt, salt, salt, salt);
}
