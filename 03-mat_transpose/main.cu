#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <corecrt_math.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const int TILE_DIM = 32;

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

void timing(const real* d_A, real* d_B, const int N, const int task);
__global__ void copy(const real* A, real *B, const int N);
__global__ void transpose1(const real *A, real* B, const int N);
__global__ void transpose2(const real* A, real* B, const int N);
__global__ void transpose_shared(const real *A, real* B, const int N);

void print_matrix(const int N, const real* A);

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: %s N\n", argv[0]);
        return 1;
    }
    const int N = atoi(argv[1]);
    const int N2 = N * N;
    const int M = sizeof(real) * N2;
    // alloc host memory
    real* h_A = (real*) malloc(M);
    real* h_B = (real*) malloc(M);
    for (int n = 0; n < N2; n++) {
        h_A[n] = n;
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = h_A[i * N + j];
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    printf("cpu copy cost: %lld ms.\n", std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
    // alloc device memory
    real *d_A, *d_B;
    CHECK(cudaMalloc(&d_A, M));
    CHECK(cudaMalloc(&d_B, M));
    // copy host data to device
    CHECK(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice));

    printf("\ncopy:\n");
    timing(d_A, d_B, N, 0);
    printf("\ntranspose with coalesced read:\n");
    timing(d_A, d_B, N, 1);
    printf("\ntranspose with coalesced write:\n");
    timing(d_A, d_B, N, 2);
    printf("\ntranspose with shared memory:\n");
    timing(d_A, d_B, N, 3);

    return 0;
}

void timing(const real* d_A, real* d_B, const int N, const int task)
{
    const int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;
    const int grid_size_y = grid_size_x;
    const dim3 block_size(TILE_DIM, TILE_DIM);
    const dim3 grid_size(grid_size_x, grid_size_y);

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; repeat++) 
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));

        switch(task) 
        {
            case 0:
                copy<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 1:
                transpose1<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 2:
                transpose2<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 3:
                transpose_shared<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            default:
                printf("Error: wrong task\n");
                break;
        }
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms. \n", elapsed_time);
        if (repeat > 0) {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
}

__global__ void copy(const real* A, real* B, const int N)
{
    const int nx = blockIdx.x * TILE_DIM + threadIdx.x;
    const int ny = blockIdx.y * TILE_DIM + threadIdx.y;
    const int index = ny * N + nx;
    if (nx < N && ny < N)
    {
        B[index] = A[index];
    }
}

__global__ void transpose1(const real *A, real* B, const int N)
{
    // 读取矩阵A是顺序的，写入矩阵B不是顺序的
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N) 
    {
        B[nx * N + ny] = A[ny * N + nx];
    }
}

/*
    合并访问指的是一个warp对全局内存的一次访问请求导致最少数量的数据传输；
    在一次数据传输中，转移的一片内存的首地址一定是32的整数倍，例如一次数据传输只能从全局内存读取地址为0到31字节、32到63字节等片段的数据；
    使用CUDA运行时API分配的内存的首地址至少是256的整数倍；
    如果编译器能够判断一个全局内存变量在整个核函数的范围都只可读，自动会用函数__ldg() 读取全局内存，从而对数据进行缓存，缓解非合并访问带来的影响。
    通常不能满足读取和写入都是合并的情况下，一般来说应当尽量做到合并地写入。
*/
__global__ void transpose2(const real *A, real* B, const int N) 
{
    // 读取矩阵A不是顺序的，写入矩阵B是顺序的
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N) 
    {
        B[ny * N + nx] = A[nx * N + ny];
    }
}

__global__ void transpose_shared(const real *A, real* B, const int N)
{
    // 读取矩阵A是顺序的，写入矩阵B不是顺序的
    __shared__ real S[TILE_DIM][TILE_DIM + 1];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    if (nx1 < N && ny1 < N)
    {
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N)
    {
        B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}
