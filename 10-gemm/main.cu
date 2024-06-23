#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <corecrt_math.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <utility>
#include <chrono>
#include "../00-common/common.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <vcruntime.h>
#include <corecrt_math_defines.h>
#include <vector_types.h>

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

#define OFFSET(row, col, ld) ((row) * ld + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

void cpuSgemm(float* a, 
              float* b, 
              float* c, 
              const int M,
              const int N,
              const int K)
{
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++) 
        {
            float psum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                psum += (a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)]);
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

__global__ void nativaSgem_kernel(float* __restrict__ a,
                                  float* __restrict__ b,
                                  float* __restrict__ c,
                                  const int M,
                                  const int N,
                                  const int K)
{
     // 使用restrict关键字告诉编译器，你放心优化，传入的指针a和指针b一定不指向同一个地址。
    int n = blockIdx.x * blockDim.x + threadIdx.x; // col
    int m = blockIdx.y * blockDim.y + threadIdx.y; // row
    if (m < M && n < N)
    {
        float psum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; k++)
        {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
}

void nativaSgem(float* __restrict__ a,
                float* __restrict__ b,
                float* __restrict__ c,
                const int M,
                const int N,
                const int K)
{
    const int BM = 32, BN = 32;
    dim3 blockDim(BN, BM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    nativaSgem_kernel<<<gridDim, blockDim>>>(a, b, c, M, N, K);
    cudaCheck(cudaGetLastError());
}

__global__ void sgemm_V1_kernel(float* __restrict__ a,
                                float* __restrict__ b,
                                float* __restrict__ c,
                                const int M,
                                const int N,
                                const int K)
{
    // 每个block计算[BM, BN]个的元素,每个线程计算[TM, TN]个的元素
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0};

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx; // block内的线程id
    // 将a数据load到共享内存s_a中, 每个线程负责load(BM * BK) / (blockDim.x * blockDim.y)个数据
    // 因为FLOAT4函数可以一次load 4个float数据,而BK为8,所以将线程按照2列的形式排布即可
    int load_a_smem_m = tid >> 1; // tid/2, row of s_a
    int load_a_smem_k = (tid & 1) << 2; // (tid % 2 == 0) ? 0 : 4, col of s_a
    // 将b数据load到共享内存s_b中, 每个线程负责load(BK * BN) / (blockDim.x * blockDim.y)个数据
    // 因为FLOAT4函数可以一次load 4个float数据,而BK为128,所以将线程按照32列的形式排布即可
    int load_b_smem_k = tid >> 5; // tid/32, row of s_b
    int load_b_smem_n = (tid & 31) << 2; // (tid % 32) * 4, col of s_b

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++)
    {
        // load数据
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; k++) 
        {
            #pragma unroll
            for (int m = 0; m < TM; m++)
            {
                #pragma unroll
                for (int n = 0; n < TN; n++)
                {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++)
    {
        int store_c_gmem_m = by * BM + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4)
        {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
        }
    }
}

void sgemm_V1(float* __restrict__ a,
                float* __restrict__ b,
                float* __restrict__ c,
                const int M,
                const int N,
                const int K)
{
    const int BM = 128, BN = 128, TN = 8, TM = 8;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_V1_kernel<<<gridDim, blockDim>>>(a, b, c, M, N, K);
    cudaCheck(cudaGetLastError());
}

__global__ void sgemm_V2_kernel(float* __restrict__ a,
                                float* __restrict__ b,
                                float* __restrict__ c,
                                const int M,
                                const int N,
                                const int K)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BK][BM];
    __shared__ float s_b[BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    for (int bk = 0; bk < (K + BK - 1) / BK; bk++)
    {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        s_a[load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < BK; tk++)
        {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; tm++)
            {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++)
                {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++)
    {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++)
    {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}

void sgemm_V2(float* __restrict__ a,
              float* __restrict__ b,
              float* __restrict__ c,
              const int M,
              const int N,
              const int K)
{
    const int BM = 128, BN = 128, TN = 8, TM = 8;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_V2_kernel<<<gridDim, blockDim>>>(a, b, c, M, N, K);
    cudaCheck(cudaGetLastError());
}

__global__ void sgemm_V3_kernel(float* __restrict__ a,
                                float* __restrict__ b,
                                float* __restrict__ c,
                                const int M,
                                const int N,
                                const int K)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        s_a[0][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
    }

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {

        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2         ]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2         ]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        s_a[smem_sel_next][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();
    }

    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2         ]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2         ]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}

void sgemm_V3(float* __restrict__ a,
              float* __restrict__ b,
              float* __restrict__ c,
              const int M,
              const int N,
              const int K)
{
    const int BM = 128, BN = 128, TN = 8, TM = 8;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_V3_kernel<<<gridDim, blockDim>>>(a, b, c, M, N, K);
    cudaCheck(cudaGetLastError());
}

void sgemm_cublas(float* __restrict__ a,
                  float* __restrict__ b,
                  float* __restrict__ c,
                  const int M,
                  const int N,
                  const int K)
{
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;
    cublasSgemm(cublas_handle, 
                CUBLAS_OP_N, 
                CUBLAS_OP_N,
                N,
                M,
                K,
                &cublas_alpha,
                b,
                N,
                a,
                K,
                &cublas_beta,
                c,
                N);

    // cublasSgemm(cublas_handle, 
    //             CUBLAS_OP_T, 
    //             CUBLAS_OP_T, 
    //             M, 
    //             N, 
    //             K, 
    //             &cublas_alpha, 
    //             a, 
    //             K, 
    //             b, 
    //             N, 
    //             &cublas_beta, 
    //             c, 
    //             M);
    cublasDestroy(cublas_handle);
}

void gemm(const int method, 
          float* __restrict__ a,
          float* __restrict__ b,
          float* __restrict__ c,
          const int M,
          const int N,
          const int K)
{
    switch (method) {
        case 0:
            nativaSgem(a, b, c, M, N, K);
            break;
        case 1:
            sgemm_V1(a, b, c, M, N, K);
            break;
        case 2:
            sgemm_V2(a, b, c, M, N, K);
            break;
        case 3:
            sgemm_V3(a, b, c, M, N, K);
            break;
        case 4:
            sgemm_cublas(a, b, c, M, N, K);
            break;
        default:
            printf("invalid method number!\n");
            break;
    }
}

float testPerformance(const int method, 
                      const int M, 
                      const int N, 
                      const int K, 
                      const int repeat)
{
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float* d_a, *d_b, *d_c;
    cudaCheck(cudaMalloc(&d_a, size_a));
    cudaCheck(cudaMalloc(&d_b, size_b));
    cudaCheck(cudaMalloc(&d_c, size_c));

    cudaEvent_t start, end;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++)
    {
        gemm(method, d_a, d_b, d_c, M, N, K);
    }
    cudaCheck(cudaEventRecord(end));
    cudaEventSynchronize(end);

    float msec, sec;
    cudaCheck(cudaEventElapsedTime(&msec, start, end));
    sec = msec / 1000.0 / repeat;

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(end));
    cudaCheck(cudaFree(d_a));
    cudaCheck(cudaFree(d_b));
    cudaCheck(cudaFree(d_c));
    return sec;
}

void testError(const int method, float tol = 1e-5f)
{
    const int M = 512, N = 512, K = 512;
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
    h_a = make_random_float(size_a);
    h_b = make_random_float(size_b);
    h_c = (float *)malloc(size_c);
    cudaCheck(cudaMalloc(&d_a, size_a));
    cudaCheck(cudaMalloc(&d_b, size_b));
    cudaCheck(cudaMalloc(&d_c, size_c));

    // cpu
    cpuSgemm(h_a, h_b, h_c, M, N, K);

    // gpu
    cudaCheck(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    gemm(method, d_a, d_b, d_c, M, N, K);

    validate_result<float, float>(d_c, h_c, "gemm", M * N, tol);

    free(h_a);
    free(h_b);
    free(h_c);
    cudaCheck(cudaFree(d_a));
    cudaCheck(cudaFree(d_b));
    cudaCheck(cudaFree(d_c));
}

int main(int argc, char **argv)
{
    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
   
    const int outer_repeat = 10, inner_repeat = 1;
    const int TESTNUM = 15;
    assert(TESTNUM == (sizeof(M_list) / sizeof(int)));
    assert(TESTNUM == (sizeof(N_list) / sizeof(int)));
    assert(TESTNUM == (sizeof(K_list) / sizeof(int)));

    // nativaSgem
    printf("\nKernal = naiveSgemm\n");
    int method = 0;
    testError(method);
    for (int i = 0; i < TESTNUM; i++)
    {
        const int M = M_list[i], N = N_list[i], K = K_list[i];
        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;
        for (int j = 0; j < outer_repeat; j++)
        {
            double this_sec = testPerformance(method, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }
        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;
        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }

    // sgemm_V1
    printf("\nKernal = sgemm_V1\n");
    method = 1;
    testError(method);
    for (int i = 0; i < TESTNUM; i++)
    {
        const int M = M_list[i], N = N_list[i], K = K_list[i];
        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;
        for (int j = 0; j < outer_repeat; j++)
        {
            double this_sec = testPerformance(method, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }
        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;
        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }

    // sgemm_V2
    printf("\nKernal = sgemm_V2\n");
    method = 2;
    testError(method);
    for (int i = 0; i < TESTNUM; i++)
    {
        const int M = M_list[i], N = N_list[i], K = K_list[i];
        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;
        for (int j = 0; j < outer_repeat; j++)
        {
            double this_sec = testPerformance(method, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }
        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;
        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }

    // sgemm_V3
    printf("\nKernal = sgemm_V3\n");
    method = 3;
    testError(method, 1e1f);
    for (int i = 0; i < TESTNUM; i++)
    {
        const int M = M_list[i], N = N_list[i], K = K_list[i];
        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;
        for (int j = 0; j < outer_repeat; j++)
        {
            double this_sec = testPerformance(method, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }
        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;
        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }

    // sgemm_cublas
    printf("\nKernal = sgemm_cublas\n");
    method = 4;
    testError(method, 1e-3f);
    for (int i = 0; i < TESTNUM; i++)
    {
        const int M = M_list[i], N = N_list[i], K = K_list[i];
        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;
        for (int j = 0; j < outer_repeat; j++)
        {
            double this_sec = testPerformance(method, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }
        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;
        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }
    return 0;
}
