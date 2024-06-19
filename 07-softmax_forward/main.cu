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

void softmax_forward_cpu(float* out, const float* inp, int N, int C)
{
    for (int i = 0; i < N; i++) 
    {
        const float* inp_row = inp + i * C;
        float *out_row = out + i * C;
        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) 
        {
            if (inp_row[j] > maxval)
            {
                maxval = inp_row[j];
            }
        }
        double sum = 0.0f;
        for (int j = 0; j < C; j++)
        {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        float norm = 1.0f / (float)sum;
        for (int j = 0; j < C; j++)
        {
            out_row[j] *= norm;
        }
    }
}

void softmax_forward_online_cpu(float* out, const float* inp, int N, int C)
{
    for (int i = 0; i < N; i++)
    {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;
        float sum = 0.0f;
        float maxval = -INFINITY;

        for (int j = 0; j < C; j++) 
        {
            float maxval_prev = maxval;
            if (inp_row[j] > maxval)
            {
                maxval = inp_row[j];
                sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
            } else
            {
                sum += expf(inp_row[j] - maxval); 
            }
        }
        for (int j = 0; j < C; j++)
        {
            out_row[j] = expf(inp_row[j] - maxval) / sum;
        }
    }
}

// kernel launcher

__global__ void softmax_forward_kernel1(float* out, 
                                        const float* inp, 
                                        int N, 
                                        int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;
        float maxval = -INFINITY;
        for (int j = 0; j < C; j++)
        {
            if (inp_row[j] > maxval)
            {
                maxval = inp_row[j];
            }
        }
        double sum = 0.0f;
        for (int j = 0; j < C; j++)
        {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++)
        {
            out_row[j] /= (float)sum;
        }
    }
}

void softmax_forward1(float* out, 
                      const float* inp, 
                      int N, 
                      int C, 
                      int block_size)
{
    const int grid_size = ceil_div(N, block_size);
    softmax_forward_kernel1<<<grid_size, block_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

__global__ void softmax_forward_kernel2(float* out, 
                                        const float* inp, 
                                        int N, 
                                        int C)
{
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const float* x = inp + C * idx;
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += block_size)
    {
        maxval = fmaxf(maxval, x[i]);
    }
    shared[tid] = maxval;
    for (int stride = block_size >> 1; stride >= 1; stride >>= 1)
    {
        __syncthreads();
        if (tid < stride)
        {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }
    __syncthreads();
    float offset = shared[0];
    for (int i = tid; i < C; i += block_size)
    {
        out[idx * C + i] = expf(x[i] - offset);
    }
    __syncthreads();
    
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += block_size)
    {
        sumval += x[i];
    }
    shared[tid] = sumval;
    __syncthreads();

    for (int stride = block_size >> 1; stride >= 1; stride >>= 1)
    {
        __syncthreads();
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
    }
    __syncthreads();
    float sum = shared[0];
    for (int i = tid; i < C; i += block_size)
    {
        out[idx * C + i] = x[i] / sum;
    }

}

void softmax_forward2(float* out, 
                      const float* inp, 
                      int N, 
                      int C, 
                      int block_size)
{
    const int grid_size = N;
    size_t shared_mem_size = block_size * sizeof(float);
    softmax_forward_kernel2<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

//__shfl_down_sync的作用是平行广播一个线程的值到另外一个线程,比如offset为16的话,则是依次将[16, 17, ..., 31] -> [0, 1, ...., 15]
__device__ float warpReduceMax(float val)
{
    for (int offset = 16; offset >= 1; offset >>= 1)
    {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val)
{
    for (int offset = 16; offset >= 1; offset >>= 1)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softmax_forward_kernel3(float* out, 
                                        const float* inp, 
                                        int N, 
                                        int C)
{
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = inp + idx * C;

    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x)
    {
        maxval = fmaxf(x[i], maxval);
    }
    maxval = warpReduceMax(maxval);

    // Broadcast maxval within the warp
    float offset = __shfl_sync(0xFFFFFFFF, maxval, 0);
    for (int i = tid; i < C; i += blockDim.x)
    {
        out[idx * C + i] = expf(x[i] - offset);
    }
    float sumval = 0.0f;
    x = out + idx * C;
    for (int i = tid; i < C; i += blockDim.x)
    {
        sumval += x[i];
    }
    sumval = warpReduceSum(sumval);
    float sum = __shfl_sync(0xFFFFFFFF, sumval, 0);
    for (int i = tid; i < C; i += blockDim.x)
    {
        out[idx * C + i] = x[i] / sum;
    }
}

void softmax_forward3(float* out, 
                      const float* inp, 
                      int N, 
                      int C, 
                      int block_size)
{
    block_size = 32;
    int grid_size = N;
    size_t shared_mem_size = block_size * sizeof(float);
    softmax_forward_kernel3<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

__global__ void softmax_forward_kernel4(float* out, 
                                        const float* inp, 
                                        int N, 
                                        int C)
{
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    int warpsPerBlock = blockDim.x / 32;

    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    const float* x = inp + idx * C;

    float maxval = -INFINITY;
    for (int i = 0; i < C; i += blockDim.x)
    {
        maxval = fmaxf(maxval, x[i]);
    }
    maxval = warpReduceMax(maxval); // reduce a warp
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    if (tid == 0)
    {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++)
        {
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();

    float offset = maxvals[0];
    for (int i = tid; i < C; i += blockDim.x)
    {
        out[idx * C + i] = expf(x[i] - offset);
    }

    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x)
    {
        sumval += x[i];
    }
    sumval = warpReduceSum(sumval);
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    if (tid == 0)
    {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i)
        {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();

    float sum = sumvals[0];
    for (int i = tid; i < C; i += blockDim.x)
    {
        out[idx * C + i] = x[i] / sum;
    }
}

void softmax_forward4(float* out, 
                      const float* inp, 
                      int N, 
                      int C, 
                      int block_size)
{
    int grid_size = N;
    size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
    softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

__global__ void softmax_forward_online_kernel1(float* out, 
                                               const float* inp, 
                                               int N, 
                                               int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        double sum = 0.0f;
        for (int j = 0; j < C; j++)
        {
            float maxval_prev = maxval;
            if (inp_row[j] > maxval)
            {
                maxval = inp_row[j];
                sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
            }
            else
            {
                sum = sum + expf(inp_row[j] - maxval);
            }
        }
        for (int j = 0; j < C; j++)
        {
            out_row[j] = expf(inp_row[j] - maxval) / sum;
        }
    }
}

void softmax_forward_online1(float* out, 
                             const float* inp, 
                             int N, 
                             int C, 
                             int block_size)
{
    int grid_size = ceil_div(N, block_size);
    softmax_forward_online_kernel1<<<grid_size, block_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

struct __align__(8) SumMax
{
    float maxval;
    float sum;
};

__device__ __forceinline__ SumMax reduce_sum_max_op(SumMax a, SumMax b)
{
    bool a_bigger = (a.maxval > b.maxval);
    SumMax bigger_m = a_bigger ? a : b;
    SumMax smaller_m = a_bigger ? b : a;
    SumMax res;
    res.maxval = bigger_m.maxval;
    res.sum = bigger_m.sum + smaller_m.sum * expf(smaller_m.maxval - bigger_m.maxval);
    return res;
}

__global__ void softmax_forward_online_kernel2(float* out, 
                                               const float* inp, 
                                               int N,
                                               int C)
{
	namespace cg = cooperative_groups;
	cg::thread_block block = cg::this_thread_block();
	cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= N)
    {
        return;
    }
    const float* x = inp + idx * C;
    // base case for the reduction
    SumMax sm_partial;
	sm_partial.maxval = -INFINITY;
	sm_partial.sum = 0.0f;

	// first, thread coarsening by directly accessing global memory in series
	for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sm_partial = reduce_sum_max_op(sm_partial, { x[i], 1.0f });
	}
    // second, the reduction
	SumMax sm_total = cg::reduce(warp, sm_partial, reduce_sum_max_op);
	// divide the whole row by the sum
	for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        // the below is equivalent to
        // out[idx * C + i] = expf(x[i] - sm_total.maxval) / sm_total.sum;
        // but uses special instruction that bypasses the cache
        __stcs(out + idx * C + i, expf(x[i] - sm_total.maxval) / sm_total.sum);
	}
}

void softmax_forward_online2(float* out, 
                             const float* inp, 
                             int N, 
                             int C, 
                             int block_size)
{
    int grid_size = ceil_div(N * 32, block_size); // N / (block_size / 32) 个grid即可
    softmax_forward_online_kernel2<<<grid_size, block_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

__global__ void softmax_forward_kernel7(float* out, const float* inp, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel4, but optimised for very large Cs with advanced unrolling

    // The trick is to read into a register array (all indices known at compile time)
    // and always read UNROLL_FACTOR values to maximise memory level parallelism
    // even if we would be out of bounds, we set the index to min(C-1, idx)
    // so we just do some unnecessary reads (obviously bad for small C)
    // the writes are in a separate loop with a conditional check for out of bounds
    // making it separate is necessary to convince the compiler to do the right thing
    const int UNROLL_FACTOR = 8;
    const int warpsPerBlock = blockDim.x / 32;

    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    if (tid >= C) {
        maxvals[warpId] = -INFINITY;
        sumvals[warpId] = 0.0f;
        return;
    }

    const float* x = inp + idx * C; // input
    float* y = out + idx * C; // output

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            maxval = fmaxf(maxval, x[min(C - 1, i + u*blockDim.x)]);
        }
    }

    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);
    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();
    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    // + thread coarsening for sum
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = __ldcs(&x[min(C - 1, i + u*blockDim.x)]);
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
                float output = expf(reg_array[u] - offset);
                y[min(C - 1, i + u*blockDim.x)] = output; // compiler likes redundant min()?!
                sumval += output; // combined into the same loop unlike kernel3
            }
        }
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);
    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();
    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = y[min(C - 1, i + u*blockDim.x)];
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
                y[i + u*blockDim.x] = reg_array[u] / sum;
            }
        }
    }
}

void softmax_forward7(float* out, 
                      const float* inp, 
                      int N, 
                      int C, 
                      int block_size)
{
    int grid_size = N;
    size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
    softmax_forward_kernel7<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_forward(int kernel_num,
                     float* out,
                     const float* inp,
                     int N,
                     int C,
                     const int block_size)
{
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    switch(kernel_num)
    {
        case 1:
            softmax_forward1(out, inp, N, C, block_size);
            break;
        case 2:
            softmax_forward2(out, inp, N, C, block_size);
            break;
        case 3:
            softmax_forward3(out, inp, N, C, block_size);
            break;
        case 4:
            softmax_forward4(out, inp, N, C, block_size);
            break;
        case 5:
            softmax_forward_online1(out, inp, N, C, block_size);
            break;
        case 6:
            softmax_forward_online2(out, inp, N, C, block_size);
            break;
        case 7:
            softmax_forward7(out, inp, N, C, block_size);
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

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    float* inp = make_random_float(B * T * V);
    float* out = (float*) malloc(B * T * V * sizeof(float));
    
    // make the input less uniformly random: Otherwise, all probabilities will be basically zero,
    // and the tests are not actually meaningful.
    const int* outliers = make_random_int(B * T * 3, V);
    // 随机选取3 * B * T个元素*20
    for (int k = 0; k < 3; k++)
    {
        for (int j = 0; j < B * T; j++)
        {
            inp[j * V + outliers[j * 3 + k]] *= 20;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    softmax_forward_cpu(out, inp, B * T, V);
    {
        float max_el = -INFINITY;
        for(int i = 0; i < B * T * V; ++i) 
        {
            max_el = max(max_el, out[i]);
        }
        assert(max_el > 1e-4);
        printf("Largest output is: %f\n", max_el);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    printf("softmax_forward_cpu cost: %lld ms.\n", std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());

    // start = std::chrono::high_resolution_clock::now();
    // memset(out, 0, B * T * V * sizeof(float));
    // softmax_forward_online_cpu(out, inp, B * T, V);
    // {
    //     float max_el = -INFINITY;
    //     for(int i = 0; i < B * T * V; ++i)
    //     {
    //         max_el = max(max_el, out[i]);
    //     }
    //     assert(max_el > 1e-4);
    //     printf("Largest output is: %f\n", max_el);
    // }
    // stop = std::chrono::high_resolution_clock::now();
    // printf("softmax_forward_online_cpu cost: %lld ms.\n", std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());

    float* d_out;
    float* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * V * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * V * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * V * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        cudaCheck(cudaMemset(d_out, 0, B * T * V * sizeof(float)));
        softmax_forward(1, d_out, d_inp, B * T, V, block_size);
        // validate_result(d_out, out, "out", B * T * V, 1e-4f);
    }

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        cudaCheck(cudaMemset(d_out, 0, B * T * V * sizeof(float)));
        softmax_forward(2, d_out, d_inp, B * T, V, block_size);
        // validate_result(d_out, out, "out", B * T * V, 1e-4f);
    }

    {
        cudaCheck(cudaMemset(d_out, 0, B * T * V * sizeof(float)));
        softmax_forward(3, d_out, d_inp, B * T, V, 32);
        // validate_result(d_out, out, "out", B * T * V, 1e-4f);
    }

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        cudaCheck(cudaMemset(d_out, 0, B * T * V * sizeof(float)));
        softmax_forward(4, d_out, d_inp, B * T, V, block_size);
        // validate_result(d_out, out, "out", B * T * V, 1e-4f);
    }

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        cudaCheck(cudaMemset(d_out, 0, B * T * V * sizeof(float)));
        softmax_forward(5, d_out, d_inp, B * T, V, block_size);
        // validate_result(d_out, out, "out", B * T * V, 1e-4f);
    }

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        cudaCheck(cudaMemset(d_out, 0, B * T * V * sizeof(float)));
        softmax_forward(6, d_out, d_inp, B * T, V, block_size);
        // validate_result(d_out, out, "out", B * T * V, 1e-4f);
    }

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        cudaCheck(cudaMemset(d_out, 0, B * T * V * sizeof(float)));
        softmax_forward(7, d_out, d_inp, B * T, V, block_size);
        // validate_result(d_out, out, "out", B * T * V, 1e-4f);
    }
    // free memory
    free(out);
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    return 0;
}

