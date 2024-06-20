#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <corecrt_math.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <chrono>
#include "../00-common/common.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <vcruntime.h>
#include <corecrt_math_defines.h>

// turn on bf16 as default, done up here for now
#define ENABLE_BF16

#if defined(ENABLE_BF16)
typedef __nv_bfloat16 floatX;
typedef __nv_bfloat16 floatN;
#elif defined(ENABLE_FP16)
typedef half floatX;
typedef half floatN;
#else
typedef float floatX;
typedef float floatN;
#endif

typedef Packed128<floatX> x128;

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

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

// CPU code reference
void gelu_forward_cpu(float* out, const float* inp, int N)
{
    for (int i = 0; i < N; i++)
    {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

__global__ void gelu_forward_kernel1(floatX* out,
                                     const floatX* inp, 
                                     int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < N)
    {
       float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}

void gelu_forward1(floatX* out,
                   const floatX* inp,
                   int N,
                   const int block_size)
{
    const int grid_size = ceil_div(N, block_size);
    gelu_forward_kernel1<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

__global__ void gelu_forward_kernel2(floatX* out,
                                     const floatX* inp, 
                                     int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (i < N)
    {
        x128 packed_out;
        x128 packed_inp = load128cs(inp + i);
        for (int k = 0; k < packed_inp.size; ++k)
        {
            float xi = (float)packed_inp[k];
            float cube =  0.044715f * xi * xi * xi;
            packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
        }
        store128(out + i, packed_out);
    }
}

void gelu_forward2(floatX* out,
                   const floatX* inp,
                   int N,
                   const int block_size)
{
    const int grid_size = ceil_div(N, block_size * x128::size);
    gelu_forward_kernel2<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

__global__ void gelu_forward_kernel3(floatX* out,
                                     const floatX* inp, 
                                     int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i < N)
    {
        for (int k = 0; k < 8; k++) {
            if (i + k >= N) break;
            float xi = inp[i + k];
            float cube = 0.044715f * xi * xi * xi;
            out[i + k] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
        }
    }
}

void gelu_forward3(floatX* out,
                   const floatX* inp,
                   int N,
                   const int block_size)
{
    const int grid_size = ceil_div(N, block_size * 8);
    gelu_forward_kernel3<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

void gelu_forward(int kernel_num,
                     floatX* out,
                     const floatX* inp,
                     int B, int T, int C,
                     const int block_size)
{
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    switch(kernel_num)
    {
        case 1:
            gelu_forward1(out, inp, B * T * C, block_size);
            break;
        case 2:
            gelu_forward2(out, inp, B * T * C, block_size);
            break;
        case 3:
            gelu_forward3(out, inp, B * T * C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
    cudaCheck(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("[kernel%d]Checking block size = %d,elapsed_time = %g ms.\n", kernel_num, block_size, elapsed_time);
}

int main(int argc, char **argv)
{
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;

    float* out = (float*) malloc(B * T * C * sizeof(float));
    float* inp = make_random_float(B * T * C);

    // first check the correctness of the kernel
    gelu_forward_cpu(out, inp, B * T * C);

    floatX* d_out;
    floatX* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++)
    {
        cudaCheck(cudaMemset(d_out, 0, B * T * C * sizeof(floatX)));
        gelu_forward(1, d_out, d_inp, B, T, C, block_sizes[i]);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        // validate_result(d_out, out, "out", B * T * C, tol);
    }

    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++)
    {
        cudaCheck(cudaMemset(d_out, 0, B * T * C * sizeof(floatX)));
        gelu_forward(2, d_out, d_inp, B, T, C, block_sizes[i]);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        // validate_result(d_out, out, "out", B * T * C, tol);
    }
    
    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++)
    {
        cudaCheck(cudaMemset(d_out, 0, B * T * C * sizeof(floatX)));
        gelu_forward(3, d_out, d_inp, B, T, C, block_sizes[i]);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        // validate_result(d_out, out, "out", B * T * C, tol);
    }

    // free memory
    free(out);
    free(inp);

    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    return 0;
}

