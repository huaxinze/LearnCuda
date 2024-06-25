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
void crossentropy_softmax_backward_cpu(float* dlogits,
                                       const float* dlosses,
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
            float* dlogits_bt = dlogits + b * T * C + t * C;
            const float* probs_bt = probs + b * T * C + t * C;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            for (int i = 0; i < C; i++)
            {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

__global__ void crossentropy_softmax_backward_kernel1(float* dlogits,
                                                      const float* dlosses,
                                                      const float* probs,
                                                      const int* targets,
                                                      int B,
                                                      int T,
                                                      int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T * C)
    {
        int b = i / (T * C);
        int t = (i / C) % T;
        int c = i % C;
        float* dlogits_bt = dlogits + b * T * C + t * C;
        const float* probs_bt = probs + b * T * C + t * C;
        float dloss = dlosses[b * T + t];
        int ix = targets[b * T + t];
        float p = probs_bt[c];
        float indicator = c == ix ? 1.0f : 0.0f;
        dlogits_bt[c] += (p - indicator) * dloss;
    }

}

void crossentropy_softmax_backward1(float* dlogits,
                                    const float* dlosses,
                                    const float* probs,
                                    const int* targets,
                                    int B,
                                    int T,
                                    int C,
                                    const int block_size)
{
    const int grid_size = ceil_div(B * T * C, block_size);
    crossentropy_softmax_backward_kernel1<<<grid_size, block_size>>>(dlogits, dlosses, probs, targets, B, T, C);
    cudaCheck(cudaGetLastError());
}

void crossentropy_softmax_backward(int kernel_num,
                                   float* dlogits,
                                   const float* dlosses,
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
            crossentropy_softmax_backward1(dlogits, dlosses, probs, targets, B, T, C, block_size);
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
    int V = 50257;

    float* probs = make_random_float(B * T * V);
    int* targets = make_random_int(B * T, V);
    float* dlosses = make_random_float(B * T);
    float* dlogits = make_zeros_float(B * T * V);

    float* d_probs;
    int* d_targets;
    float* d_dlosses;
    float* d_dlogits;

    cudaCheck(cudaMalloc(&d_probs, B * T * V * sizeof(float)));
    cudaCheck(cudaMalloc(&d_targets, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(&d_dlosses, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dlogits, B * T * V * sizeof(float)));
    cudaCheck(cudaMemcpy(d_probs, probs, B * T * V * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dlosses, dlosses, B * T * sizeof(float), cudaMemcpyHostToDevice));

    // cpu
    crossentropy_softmax_backward_cpu(dlogits, dlosses, probs, targets, B, T, V);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++)
    {
        cudaCheck(cudaMemset(d_dlogits, 0, B * T * V * sizeof(float)));
        crossentropy_softmax_backward(1, d_dlogits, d_dlosses, d_probs, d_targets, B, T, V, block_sizes[i]);
        float tol = 1e-5;
        validate_result(d_dlogits, dlogits, "d_dlogits", B * T * V, tol);
    }

    // free memory
    free(probs);
    free(targets);
    free(dlosses);
    free(dlogits);
    cudaCheck(cudaFree(d_probs));
    cudaCheck(cudaFree(d_targets));
    cudaCheck(cudaFree(d_dlosses));
    cudaCheck(cudaFree(d_dlogits));

    return 0;
}

