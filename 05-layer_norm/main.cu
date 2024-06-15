#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <corecrt_math.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../00-common/common.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

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

void layernorm_forward_cpu(float* out,
                           float* mean, 
                           float* rstd, 
                           const float* inp, 
                           const float* weight, 
                           const float* bias, 
                           int B, 
                           int T, 
                           int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* x = inp + b * T * C + t * C;
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            float s = 1.0f / sqrtf(v + eps);
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m));
                float o = n * weight[i] + bias[i];
                out_bt[i] = o;
            }
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

__global__ void layernorm_forward_kernel1(float* out,
                                          float* mean,
                                          float* rstd,
                                          const float* inp,
                                          const float* weight,
                                          const float* bias,
                                          int N,
                                          int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;
    if (idx < N) {
        const float* x = inp + idx * C;
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += x[i];
        }
        m = m / C;
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        float s = 1.0f / sqrt(v + eps);
        float* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * (x[i] - m));
            float o = n * weight[i] + bias[i];
            out_idx[i] = o;
        }
        mean[idx] = m;
        rstd[idx] = s;
    }
}

void layernorm_forward1(float* out,
                        float* mean,
                        float* rstd,
                        const float* inp,
                        const float* weight,
                        const float* bias,
                        int B,
                        int T,
                        int C,
                        const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    layernorm_forward_kernel1<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    CHECK(cudaGetLastError());
}

__global__ void mean_kernel(float* mean, const float* inp, int N, int C, int block_size)
{
    extern __shared__ float shared[];
    int idx = blockIdx.x; // range[0, B * T] 一个block负责计算一个C列的平均值
    int tid = threadIdx.x;
    const float* x = inp + idx * C;
    float sum = 0.0f;
    // 一个线程需要计算C / block_size个元素
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    shared[tid] = sum;
    __syncthreads();
    // reduce
    for (int offset = block_size >> 1; offset >= 1; offset >>= 1)
    {
        __syncthreads();
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
    }
    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}

__global__ void rstd_kernel(float* rstd, const float* inp, const float* mean, int N, int C, int block_size)
{
    extern __shared__ float shared[];
    int idx = blockIdx.x; // range[0, B * T] 一个block负责计算一个C列的平均值
    int tid = threadIdx.x;
    const float* x = inp + idx * C;
    float m = mean[idx];
    float sum = 0.0f;
    // 一个线程需要计算C / block_size个元素
    for (int i = tid; i < C; i += block_size) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    // reduce
    for (int offset = block_size >> 1; offset >= 1; offset >>= 1)
    {
        __syncthreads();
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
    }
    if (tid == 0) {
        rstd[idx] = 1.0f / sqrt(shared[0] / C + 1e-5f);
    }
}

__global__ void normalization_kernel(float* out,
                                     const float* inp,
                                     float* mean,
                                     float* rstd,
                                     const float* weight,
                                     const float* bias,
                                     int B,
                                     int T,
                                     int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bt = idx / C;
    int c = idx % C;
    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];
    out[idx] = o;
}

void layernorm_forward2(float* out,
                        float* mean,
                        float* rstd,
                        const float* inp,
                        const float* weight,
                        const float* bias,
                        int B,
                        int T,
                        int C,
                        const int block_size) {
    int N = B * T;
    mean_kernel<<<B * T, block_size, block_size * sizeof(float)>>>(mean, inp, N, C, block_size);                       
    CHECK(cudaGetLastError());
    rstd_kernel<<<B * T, block_size, block_size * sizeof(float)>>>(rstd, inp, mean, N, C, block_size);
    CHECK(cudaGetLastError());
    const int block_size2 = 256;
    const int grid_size = ceil_div(B * T * C, block_size2);
    normalization_kernel<<<grid_size, block_size2>>>(out, inp, mean, rstd, weight, bias, B, T, C);
    CHECK(cudaGetLastError());
}

__global__ void layernorm_forward_kernel3(float* __restrict__ out,
                                          float* __restrict__ mean,
                                          float* __restrict__ rstd,
                                          const float* __restrict__ inp,
                                          const float* __restrict__ weight,
                                          const float* __restrict__ bias,
                                          int N,
                                          int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }
    const float* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        sum += x[i];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float m = sum / C;
    if (warp.thread_rank() == 0 && mean != nullptr)
    {
        __stcs(mean + idx, m);
    }

    // rstd
    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float s = rsqrtf(sum / C + 1e-5f);
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // norm
    float *o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size())
    {
        float n = s * (__ldcs(x + c) - m);
        __stcs(o + c, n * weight[c] + bias[c]);
    }
}

void layernorm_forward3(float* out,
                        float* mean,
                        float* rstd,
                        const float* inp,
                        const float* weight,
                        const float* bias,
                        int B,
                        int T,
                        int C,
                        const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size); // (32 / block_size)个block 计算一个C，也即1个wrap计算一个C
    layernorm_forward_kernel3<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    CHECK(cudaGetLastError());
}

// same as kernel 3 but uses var(x) == mean(x**2) - mean(x)**2
__global__ void layernorm_forward_kernel4(float* __restrict__ out,
                                          float* __restrict__ mean,
                                          float* __restrict__ rstd,
                                          const float* __restrict__ inp,
                                          const float* __restrict__ weight,
                                          const float* __restrict__ bias,
                                          int N,
                                          int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }
    const float* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    float sum2 = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        float xi = x[i];
        sum += xi;
        sum2 += xi * xi;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    sum2 = cg::reduce(warp, sum2, cg::plus<float>{});
    sum = sum / C; // mean(x)
    sum2 = sum2 / C; // mean(x ** 2);
    float m = sum;
    float var = sum2 - sum * sum;
    float s = rsqrtf(var + 1e-5f);
    if (warp.thread_rank() == 0 && mean != nullptr)
    {
        __stcs(mean + idx, m);
    }
    // rstd
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // norm
    float *o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size())
    {
        float n = s * (__ldcs(x + c) - m);
        __stcs(o + c, n * weight[c] + bias[c]);
    }
}

// like 4, but in kernel 5 we have each block doing one row, not just a single warp
__global__ void layernorm_forward_kernel5(float* __restrict__ out,
                                          float* __restrict__ mean,
                                          float* __restrict__ rstd,
                                          const float* __restrict__ inp,
                                          const float* __restrict__ weight,
                                          const float* __restrict__ bias,
                                          int N,
                                          int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float shared_sum[32];
    __shared__ float shared_sum2[32];
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int idx = blockIdx.x; // simpoy one block per row
    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;
    float thread_sum = 0.0f;
    float thread_sum2 = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xi = x[i];
        thread_sum += xi;
        thread_sum2 += xi * xi;
    }
    // wrap-level reduction
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>{});
    float warp_sum2 = cg::reduce(warp, thread_sum2, cg::plus<float>{});
    // store the warp-level reduction in shared memory (we could have lane_id == 0 guard but not needed)
    shared_sum[warp_id] = warp_sum;
    shared_sum2[warp_id] = warp_sum2;
    __syncthreads();
    // load results from shared memory to threads, pad with zeros for threads that are out of bounds
    // 得到的结果赋值到每个wrap的前num_warps线程上，比如wrap1的第一个线程存shared_sum[0], 第二个线程存shared_sum[1]的值，这样就可以继续使用wrap reduce
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    // now reduce the warp-level reductions
    float block_sum = cg::reduce(warp, warp_sum, cg::plus<float>{}); // sum(x)
    float block_sum2 = cg::reduce(warp, warp_sum2, cg::plus<float>{}); // sum(x**2)
    // mean, var, rstd
    block_sum /= C; // mean(x)
    block_sum2 /= C; // mean(x**2)
    float m = block_sum;
    float var = block_sum2 - m * m;
    float s = rsqrtf(var + 1e-5f);
    // store the mean, no need to cache it
    if(threadIdx.x == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }
    // store the rstd, no need to cache it
    if(threadIdx.x == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float n = s * (__ldcs(x+i) - m);
        __stcs(o+i, n * weight[i] + bias[i]);
    }
}

void layernorm_forward4(float* out,
                        float* mean,
                        float* rstd,
                        const float* inp,
                        const float* weight,
                        const float* bias,
                        int B,
                        int T,
                        int C,
                        const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size); // (32 / block_size)个block 计算一个C，也即1个wrap计算一个C
    layernorm_forward_kernel4<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    CHECK(cudaGetLastError());
}

void layernorm_forward5(float* out,
                        float* mean,
                        float* rstd,
                        const float* inp,
                        const float* weight,
                        const float* bias,
                        int B,
                        int T,
                        int C,
                        const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = N; // 1个block 计算一个C
    layernorm_forward_kernel5<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    CHECK(cudaGetLastError());
}

void layernorm_forward(int kernel_num,
                       float* out,
                       float* mean,
                       float* rstd,
                       const float* inp,
                       const float* weight,
                       const float* bias,
                       int B,
                       int T,
                       int C,
                       const int block_size) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    switch (kernel_num) {
        case 1:
            layernorm_forward1(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 2:
            layernorm_forward2(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 3:
            layernorm_forward3(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 4:
            layernorm_forward4(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 5:
            layernorm_forward5(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("[kernel%d]Checking block size = %d,elapsed_time = %g ms.\n", kernel_num, block_size, elapsed_time);
}

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;

    int deviceIdx = 0;
    CHECK(cudaSetDevice(deviceIdx));

    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    float* d_out;
    float* d_mean;
    float* d_rstd;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    CHECK(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    CHECK(cudaMalloc(&d_mean, B * T * sizeof(float)));
    CHECK(cudaMalloc(&d_rstd, B * T * sizeof(float)));
    CHECK(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    CHECK(cudaMalloc(&d_weight, C * sizeof(float)));
    CHECK(cudaMalloc(&d_bias, C * sizeof(float)));
    CHECK(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);
    printf("layernorm_forward_cpu mean[%d][%d] = %f \n", B / 2, T / 2, mean[(B / 2) * T + (T / 2)]);
    printf("layernorm_forward_cpu rstd[%d][%d] = %f \n", B / 2, T / 2, rstd[(B / 2) * T + (T / 2)]);

    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++)
    {
        int block_size = block_sizes[i];
        layernorm_forward(1, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);
        CHECK(cudaMemcpy(mean, d_mean, sizeof(float) * T * B, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(rstd, d_rstd, sizeof(float) * T * B, cudaMemcpyDeviceToHost));
        // printf("layernorm_forward1_gpu mean[%d][%d] = %f \n", B / 2, T / 2, mean[(B / 2) * T + (T / 2)]);
        // printf("layernorm_forward1_gpu rstd[%d][%d] = %f \n", B / 2, T / 2, rstd[(B / 2) * T + (T / 2)]);
    }

    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++)
    {
        int block_size = block_sizes[i];
        layernorm_forward(2, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);
        CHECK(cudaMemcpy(mean, d_mean, sizeof(float) * T * B, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(rstd, d_rstd, sizeof(float) * T * B, cudaMemcpyDeviceToHost));
        // printf("layernorm_forward2_gpu mean[%d][%d] = %f \n", B / 2, T / 2, mean[(B / 2) * T + (T / 2)]);
        // printf("layernorm_forward2_gpu rstd[%d][%d] = %f \n", B / 2, T / 2, rstd[(B / 2) * T + (T / 2)]);
    }

    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++)
    {
        int block_size = block_sizes[i];
        layernorm_forward(3, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);
        CHECK(cudaMemcpy(mean, d_mean, sizeof(float) * T * B, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(rstd, d_rstd, sizeof(float) * T * B, cudaMemcpyDeviceToHost));
        // printf("layernorm_forward2_gpu mean[%d][%d] = %f \n", B / 2, T / 2, mean[(B / 2) * T + (T / 2)]);
        // printf("layernorm_forward2_gpu rstd[%d][%d] = %f \n", B / 2, T / 2, rstd[(B / 2) * T + (T / 2)]);
    }

    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++)
    {
        int block_size = block_sizes[i];
        layernorm_forward(4, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);
        CHECK(cudaMemcpy(mean, d_mean, sizeof(float) * T * B, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(rstd, d_rstd, sizeof(float) * T * B, cudaMemcpyDeviceToHost));
        // printf("layernorm_forward2_gpu mean[%d][%d] = %f \n", B / 2, T / 2, mean[(B / 2) * T + (T / 2)]);
        // printf("layernorm_forward2_gpu rstd[%d][%d] = %f \n", B / 2, T / 2, rstd[(B / 2) * T + (T / 2)]);
    }

    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++)
    {
        int block_size = block_sizes[i];
        layernorm_forward(5, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);
        CHECK(cudaMemcpy(mean, d_mean, sizeof(float) * T * B, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(rstd, d_rstd, sizeof(float) * T * B, cudaMemcpyDeviceToHost));
        printf("layernorm_forward2_gpu mean[%d][%d] = %f \n", B / 2, T / 2, mean[(B / 2) * T + (T / 2)]);
        printf("layernorm_forward2_gpu rstd[%d][%d] = %f \n", B / 2, T / 2, rstd[(B / 2) * T + (T / 2)]);
    }

    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_mean));
    CHECK(cudaFree(d_rstd));
    CHECK(cudaFree(d_inp));
    CHECK(cudaFree(d_weight));
    CHECK(cudaFree(d_bias));
    return 0;
}

