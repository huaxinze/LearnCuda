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

// CPU code reference
void crossentropy_forward_cpu(float* losses, 
                              const float* probs, 
                              const int* targets, 
                              int B,
                              int T,
                              int C)
{
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            const float* probs_bt = probs + b * T * C + t * C;
            const int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

__global__ void crossentropy_forward_kernel1(float* losses,
                                             const float* probs,
                                             const int* targets,
                                             int B,
                                             int T, 
                                             int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T)
    {
        int b = i / T;
        int t = i % T;
        const float* probs_bt = probs + b * T * C + t * C;
        // int ix = targets[i];
        // losses[i] = -logf(probs_bt[ix]);
        int ix = targets[b * T + t];
        losses[b * T + t] = -logf(probs_bt[ix]); 
    }

}

void crossentropy_forward1(float* losses,
                           const float* probs,
                           const int* targets,
                           int B,
                           int T,
                           int C,
                           const int block_size)
{
    const int grid_size = ceil_div(B * T, block_size);
    crossentropy_forward_kernel1<<<grid_size, block_size>>>(losses, probs, targets, B, T, C);
    cudaCheck(cudaGetLastError());
}

void crossentropy_forward(int kernel_num,
                          float* losses,
                          const float* probs,
                          const int* targets,
                          int B,
                          int T,
                          int C,
                          const int block_size)
{
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    switch (kernel_num)
    {
        case 1:
            crossentropy_forward1(losses, probs, targets, B, T, C, block_size);
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
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 50257;

    float* out = (float*) malloc(B * T * sizeof(float));
    float* probs = make_random_float_01(B * T * C);
    int* targets = make_random_int(B * T, C); // 标注的label,真实的结果
 
    // first check the correctness of the kernel
    crossentropy_forward_cpu(out, probs, targets, B, T, C);

    float* d_out;
    float* d_probs;
    int* d_targets;
    cudaCheck(cudaMalloc(&d_out, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_probs, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_targets, B * T * sizeof(int)));
    cudaCheck(cudaMemcpy(d_probs, probs, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++)
    {
        cudaCheck(cudaMemset(d_out, 0, B * T * sizeof(float)));
        crossentropy_forward(1, d_out, d_probs, d_targets, B, T, C, block_sizes[i]);
        float tol = 1e-5;
        validate_result(d_out, out, "d_out", B * T, tol);
    }

    // free memory
    free(out);
    free(probs);
    free(targets);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_probs));
    cudaCheck(cudaFree(d_targets));
    return 0;
}

