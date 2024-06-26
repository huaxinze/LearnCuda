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
void encoder_forward_cpu(float* out, 
                         const int* inp, 
                         const float* wte,
                         const float* wpe, 
                         int B,
                         int T,
                         int C)
{
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            float* out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            const float* wte_ix = wte + ix * C;
            const float* wpe_ix = wpe + t * C;
            for (int i = 0; i < C; i++)
            {
                out_bt[i] = wte_ix[i] + wpe_ix[i];
            }
        }
    }
}

__global__ void encoder_forward_kernel1(floatX* out,
                                        const int* inp,
                                        const floatX* wte,
                                        const floatX* wpe,
                                        int B,
                                        int T,
                                        int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T;
    if (idx < N)
    {
        int b = idx / T;
        int t = idx % T;
        floatX* out_bt = out + b * T * C + t * C;
        int ix = inp[b * T + t];
        const floatX* wte_ix = wte + ix * C;
        const floatX* wpe_t = wpe + t * C;
        for (int i = 0; i < C; i++)
        {
            out_bt[i] = (floatX)((float)wte_ix[i] + (float)wpe_t[i]);
        }
    }
}

void encoder_forward1(floatX* out,
                      const int* inp,
                      const floatX* wte,
                      const floatX* wpe,
                      int B,
                      int T, 
                      int C,
                      const int block_size)
{
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    encoder_forward_kernel1<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

__global__ void encoder_forward_kernel2(floatX* out,
                                        const int* inp,
                                        const floatX* wte,
                                        const floatX* wpe,
                                        int B,
                                        int T,
                                        int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (idx < N)
    {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        floatX* out_btc = out + b * T * C + t * C + c;
        const floatX* wte_ix = wte + ix * C + c;
        const floatX* wpe_tc = wpe + t * C + c;
        *out_btc = (floatX)((float)*wte_ix + (float)*wpe_tc);
    }
}

void encoder_forward2(floatX* out,
                      const int* inp,
                      const floatX* wte,
                      const floatX* wpe,
                      int B, int T, int C,
                      const int block_size)
{
    const int N = B * T * C;
    const int grid_size = ceil_div(N, block_size);
    encoder_forward_kernel2<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

__global__ void encoder_forward_kernel3(floatX* out,
                                        const int* inp,
                                        const floatX* wte,
                                        const floatX* wpe,
                                        int B,
                                        int T,
                                        int C)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int N = B * T * C;
    if (idx < N)
    {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        floatX* out_btc = out + b * T * C + t * C + c;
        const floatX* wte_ix = wte + ix * C + c;
        const floatX* wpe_tc = wpe + t * C + c;

        x128 packed_out;
        x128 wte = load128cs(wte_ix);
        x128 wpe = load128cs(wpe_tc);
        #pragma unroll
        for (int k = 0; k < wte.size; k++) {
            packed_out[k] = (floatX)((float)wte[k] + (float)wpe[k]);
        }
        store128(out_btc, packed_out);
    }
}

void encoder_forward3(floatX* out,
                      const int* inp,
                      const floatX* wte,
                      const floatX* wpe,
                      int B, int T, int C,
                      const int block_size)
{
    const int N = B * T * C;
    const int grid_size = ceil_div(N, (int)block_size * x128::size);
    encoder_forward_kernel3<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

void encoder_forward(int kernel_num,
                     floatX* out,
                     const int* inp,
                     const floatX* wte,
                     const floatX* wpe,
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
            encoder_forward1(out, inp, wte, wpe, B, T, C, block_size);
            break;
        case 2:
            encoder_forward2(out, inp, wte, wpe, B, T, C, block_size);
            break;
        case 3:
            encoder_forward3(out, inp, wte, wpe, B, T, C, block_size);
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
    int V = 50257;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    float* out = (float*) malloc(B * T * C * sizeof(float));
    int* inp = make_random_int(B * T, V); // 词id
    float* wte = make_random_float(V * C * sizeof(float)); // 词表
    float* wpe = make_random_float(T * C * sizeof(float)); // 位置信息

    floatX* d_out;
    int* d_inp;
    floatX* d_wte;
    floatX* d_wpe;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(&d_wte, V * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_wpe, T * C * sizeof(floatX)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(memcpy_convert(d_wte, wte, V * C));
    cudaCheck(memcpy_convert(d_wpe, wpe, T * C));

    encoder_forward_cpu(out, inp, wte, wpe, B, T, C);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++)
    {
        cudaCheck(cudaMemset(d_out, 0, B * T * C * sizeof(float)));
        encoder_forward(1, d_out, d_inp, d_wte, d_wpe, B, T, C, block_sizes[i]);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_out, out, "d_out", B * T * C, tol);
    }

    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++)
    {
        cudaCheck(cudaMemset(d_out, 0, B * T * C * sizeof(float)));
        encoder_forward(2, d_out, d_inp, d_wte, d_wpe, B, T, C, block_sizes[i]);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_out, out, "d_out", B * T * C, tol);
    }

    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++)
    {
        cudaCheck(cudaMemset(d_out, 0, B * T * C * sizeof(float)));
        encoder_forward(3, d_out, d_inp, d_wte, d_wpe, B, T, C, block_sizes[i]);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_out, out, "d_out", B * T * C, tol);
    }

    // free memory
    free(out);
    free(inp);
    free(wte);
    free(wpe);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_wte));
    cudaCheck(cudaFree(d_wpe));
    return 0;
}

