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
void gelu_backward_cpu(float* dinp, 
                       const float* inp, 
                       const float* dout, 
                       int N)
{
    for (int i = 0; i < N; i++)
    {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + 0.5f * x * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (floatX)(local_grad * (float)dout[i]);
    }
}

__global__ void gelu_backward_kernel1(floatX* dinp,
                                      const floatX* inp, 
                                      const floatX* dout,
                                      int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = (float)inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (floatX)(local_grad * (float)dout[i]);
    }
}

void gelu_backward1(floatX* dinp,
                    const floatX* inp,
                    const floatX* dout,
                    int N,
                    const int block_size)
{
    const int grid_size = ceil_div(N, block_size);
    gelu_backward_kernel1<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

__global__ void gelu_backward_kernel2(floatX* dinp,
                                      const floatX* inp, 
                                      const floatX* dout,
                                      int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (i < N)
    {
        x128 packed_dinp;
        x128 packed_inp = load128cs(inp + i);
        x128 packed_dout = load128cs(dout + i);
        for (int k = 0; k < packed_inp.size; k++) 
        {
            float x = (float)packed_inp[k];
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = tanhf(tanh_arg);
            float coshf_out = coshf(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
        }
        store128(dinp + i, packed_dinp);
    }
}

void gelu_backward2(floatX* dinp,
                    const floatX* inp, 
                    const floatX* dout,
                    int N,
                    const int block_size)
{
    const int grid_size = ceil_div(N, block_size * x128::size);
    gelu_backward_kernel2<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

void gelu_backward(int kernel_num,
                   floatX* dinp,
                   const floatX* inp,
                   const floatX* dout,
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
            gelu_backward1(dinp, inp, dout, B * T * C, block_size);
            break;
        case 2:
            gelu_backward2(dinp, inp, dout, B * T * C, block_size);
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

    float* dinp = (float*) malloc(B * T * C * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* dout = make_random_float(B * T * C);

    // first check the correctness of the kernel
    gelu_backward_cpu(dinp, inp, dout, B * T * C);

    floatX* d_dinp;
    floatX* d_inp;
    floatX* d_dout;
    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(floatX)));
    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));
    cudaCheck(memcpy_convert(d_dout, dout, B * T * C));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++)
    {
        cudaCheck(cudaMemset(d_dinp, 0, B * T * C * sizeof(floatX)));
        gelu_backward(1, d_dinp, d_inp, d_dout, B, T, C, block_sizes[i]);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_dinp, dinp, "dinp", B * T * C, tol);
    }

    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++)
    {
        cudaCheck(cudaMemset(d_dinp, 0, B * T * C * sizeof(floatX)));
        gelu_backward(2, d_dinp, d_inp, d_dout, B, T, C, block_sizes[i]);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_dinp, dinp, "dinp", B * T * C, tol);
    }

    // free memory
    free(dinp);
    free(inp);
    free(dout);
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_dout));
    return 0;
}

