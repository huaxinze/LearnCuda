#include <stdio.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;
const unsigned FULL_MASK = 0xffffffff;

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

void timing_cpu(const real *x);
real reduce_cpu(const real *x);

void timing_gpu(real* h_x, real* d_x, const int task);
real reduce_gpu(real* h_x, real* d_x, const int task);

void timing_parallel_dynamic(real* h_x, real* d_x, const int task);
real reduce_parallel_dynamic(real* h_x, real* d_x, const int task);

void timing_parallel_static(real* h_x, real* d_x, const int task);
real reduce_parallel_static(real* h_x, real* d_x, const int task);

void timing_wrap(real* h_x, real* d_x, const int task);
real reduce_wrap(real* h_x, real* d_x, const int task);

int main(void)
{
    real* h_x = (real*) malloc(M);
    for (int n = 0; n < N; n++)
    {
        h_x[n] = 1.23;
    }

    real *d_x;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    printf("\nreduce with cpu:\n");
    timing_cpu(h_x);

    printf("\nreduce with gpu[global memory]:\n");
    timing_gpu(h_x, d_x, 0);

    printf("\nreduce with gpu[shared memory]:\n");
    timing_gpu(h_x, d_x, 1);

    printf("\nreduce with gpu[wrap syncwarp]:\n");
    timing_wrap(h_x, d_x, 0);

    printf("\nreduce with gpu[wrap shfl]:\n");
    timing_wrap(h_x, d_x, 1);

    printf("\nreduce with gpu[wrap cp]:\n");
    timing_wrap(h_x, d_x, 2);

    printf("\nreduce with gpu[parallel dynamic]:\n");
    timing_parallel_dynamic(h_x, d_x, 0);

    printf("\nreduce with gpu[parallel static]:\n");
    timing_parallel_static(h_x, d_x, 0);

    return 0;
}

// reduce cpu
real reduce_cpu(const real *x)
{
    real sum = 0.0;
    for (int n = 0; n < N; ++n)
    {
        sum += x[n];
    }
    return sum;
}

void timing_cpu(const real *x)
{
    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; repeat++) 
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));

        real result = reduce_cpu(x);
    
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms, result = %g. \n", elapsed_time, result);
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

// reduce gpu
void timing_gpu(real* h_x, real* d_x, const int task)
{
    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; repeat++) 
    {
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        real result = reduce_gpu(h_x,d_x, task);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms, result = %g. \n", elapsed_time, result);
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

void __global__ reduce_gpu_global(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    real *x = d_x + blockIdx.x * blockDim.x;
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) 
    {
        d_y[blockIdx.x] = x[0];
    }
}

void __global__ reduce_gpu_shared(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ real s_y[128];
    s_y[tid] = (idx < N) ? d_x[idx] : 0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) 
    {
        d_y[blockIdx.x] = s_y[0];
    }
}

real reduce_gpu(real* h_x, real* d_x, const int task)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ymem = sizeof(real) * grid_size;
    const int smem = sizeof(real) * BLOCK_SIZE;
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));
    real *h_y = (real *) malloc(ymem);

    switch (task)
    {
        case 0:
            reduce_gpu_global<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 1:
            reduce_gpu_shared<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        default:
            break;
    }

    CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));
    real result = 0.0;
    for (int n = 0; n < grid_size; ++n)
    {
        result += h_y[n];
    }

    free(h_y);
    CHECK(cudaFree(d_y));
    return result;
}

// reduce parallel dynamic
void timing_parallel_dynamic(real* h_x, real* d_x, const int task)
{
    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; repeat++) 
    {
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        real result = reduce_parallel_dynamic(h_x, d_x, task);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms, result = %g. \n", elapsed_time, result);
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

void __global__ reduce_parallel_kernel(const real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ real s_y[];

    real y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    // 每个线程需要计算的元素个数是 N / stride
    for (int n = bid *  blockDim.x + tid; n < N; n += stride)
    {
        y += d_x[n];
    }
    s_y[tid] = y;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    
    y = s_y[tid];
    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1)
    {
        y += g.shfl_down(y, i);
    }

    if (tid == 0)
    {
        d_y[bid] = y;
    }
}

real reduce_parallel_dynamic(real* h_x, real* d_x, const int task)
{
    const int ymem = sizeof(real) * 10240;
    const int smem = sizeof(real) * BLOCK_SIZE;
    real h_y[1] = {0};
    real* d_y;
    CHECK(cudaMalloc(&d_y, ymem));
    // reduce N size array to 12240 size array
    reduce_parallel_kernel<<<10240, BLOCK_SIZE, smem>>>(d_x, d_y, N);
    reduce_parallel_kernel<<<1, 1024, sizeof(real) * 1024>>>(d_y, d_y, 10240);

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));
    return h_y[0];
}

// reduce parallel dynamic
void timing_parallel_static(real* h_x, real* d_x, const int task)
{
    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; repeat++) 
    {
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        real result = reduce_parallel_static(h_x, d_x, task);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms, result = %g. \n", elapsed_time, result);
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

__device__ real static_y[10240];

real reduce_parallel_static(real* h_x, real* d_x, const int task)
{
    real *d_y;
    CHECK(cudaGetSymbolAddress((void**)&d_y, static_y));

    const int smem = sizeof(real) * BLOCK_SIZE;

    reduce_parallel_kernel<<<10240, BLOCK_SIZE, smem>>>(d_x, d_y, N);
    reduce_parallel_kernel<<<1, 1024, sizeof(real) * 1024>>>(d_y, d_y, 10240);

    real h_y[1] = {0};
    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));

    return h_y[0];
}

// reduce wrap  
void timing_wrap(real* h_x, real* d_x, const int task)
{
    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; repeat++) 
    {
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        real result = reduce_wrap(h_x, d_x, task);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms, result = %g. \n", elapsed_time, result);
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

void __global__ reduce_wrap_syncwarp(real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncwarp();
    }
    if (tid == 0)
    {
        atomicAdd(d_y, s_y[0]);
    }
}

void __global__ reduce_wrap_shfl(real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    real y = s_y[tid];

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        y += __shfl_down_sync(FULL_MASK, y, offset);
    }

    if (tid == 0)
    {
        atomicAdd(d_y, y);
    }
}

void __global__ reduce_wrap_cp(real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    real y = s_y[tid];

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1)
    {
        y += g.shfl_down(y, i);
    }

    if (tid == 0)
    {
        atomicAdd(d_y, y);
    }
}

real reduce_wrap(real* h_x, real* d_x, const int task)
{
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;

    real h_y[1] = {0};
    real *d_y;
    CHECK(cudaMalloc(&d_y, sizeof(real)));
    CHECK(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));

    switch (task)
    {
        case 0:
            reduce_wrap_syncwarp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
            break;
        case 1:
            reduce_wrap_shfl<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
            break;
        case 2:
            reduce_wrap_cp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
            break;
        default:
            printf("Wrong method.\n");
            exit(1);
    }

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}
