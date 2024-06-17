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

#if defined(ENABLE_BF16)
typedef __nv_bfloat16 floatX;
typedef __nv_bfloat16 floatN;
#elif defined(ENABLE_FP16)
typedef half floatX;
typedef hale floatN;
#else
typedef float floatX;
typedef float floatN;
#endif

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

void layernorm_forward_cpu(float* out, 
                           float* mean, 
                           float* rstd,
                           const float* inp, 
                           const float* weight, 
                           const float* bias,
                           int B, 
                           int T, 
                           int C)
{
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

void layernorm_backward_cpu(float* dinp,
                            float* dweight,
                            float* dbias,
                            const float* dout,
                            const float* inp,
                            const float* weight,
                            const float* mean,
                            const float* rstd,
                            int B,
                            int T,
                            int C)
{
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * C + t * C;
            const float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            const float mean_bt = mean[b * T + t];
            const float rstd_bt = rstd[b * T + t];
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dbias[i] += dout_bt[i];
                dweight[i] += norm_bti * dout_bt[i];
                float dval = 0.0f;
                dval += dnorm_i;
                dval -= dnorm_mean;
                dval -= norm_bti * dnorm_norm_mean;
                dval *= rstd_bt;
                dinp_bt[i] += dval;
            }
        }
    }
}

__device__ void atomicAddX(float* addr, float val) {
    atomicAdd(addr, val);
}

// super naive kernel that just parallelizes over B,T and loops over C
__global__ void layernorm_backward_kernel1(float* dinp,
                                           float* dweight,
                                           float* dbias,
                                           const float* dout,
                                           const float* inp,
                                           const float* weight,
                                           const float* mean,
                                           const float* rstd,
                                           int B,
                                           int T,
                                           int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) {
        return;
    }
    int b = idx / T;
    int t = idx % T;
    const float* dout_bt = dout + b * T * C + t * C;
    const float* inp_bt = inp + b * T * C + t * C;
    float* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = mean[b * T + t];
    const float rstd_bt = rstd[b * T + t];

    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        atomicAdd(&dbias[i], dout_bt[i]);
        atomicAdd(&dweight[i], norm_bti * dout_bt[i]);
        float dval = 0.0f;
        dval += dnorm_i;
        dval -= dnorm_mean;
        dval -= norm_bti * dnorm_norm_mean;
        dval *= rstd_bt;
        dinp_bt[i] += dval;
    }
}

void layernorm_backward1(float* dinp,
                         float* dweight,
                         float* dbias,
                         const float* dout,
                         const float* inp,
                         const float* weight,
                         const float* mean,
                         const float* rstd,
                         int B,
                         int T,
                         int C,
                         const int block_size)
{
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    layernorm_backward_kernel1<<<grid_size, block_size>>>(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
}

// uses shared memory instead for the reduces
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
__global__ void layernorm_backward_kernel2(Tdinp* dinp,
                                           Tparams* dweight,
                                           Tparams* dbias,
                                           const Tdout* dout,
                                           const Trest* inp,
                                           const Tparams* weight,
                                           const Trest* mean,
                                           const Trest* rstd,
                                           int B,
                                           int T,
                                           int C)
{
    extern __shared__ float shared[]; // size = 2 * C
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    int N = B * T;
    if (idx >= N) {
        return;
    }
    int b = idx / T;
    int t = idx % T;
    const Tdout* dout_bt = dout + b * T * C + t * C;
    const Trest* inp_bt = inp + b * T * C + t * C;
    Tdinp* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = (float)mean[b * T + t];
    const float rstd_bt = (float)rstd[b * T + t];
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    #pragma unroll
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        dbias_shared[i] = 0.0f;
        dweight_shared[i] = 0.0f;
    }
    __syncthreads();

    // reduction
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = (float)weight[i] * (float) dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
    dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>{});
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // per thread need compute C / warp.size() times
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = (float)weight[i] * (float)dout_bt[i];
        // gradient contribution to bias
        atomicAdd(&dbias_shared[i], (float)dout_bt[i]);
        // gradient contribution to weight
        atomicAdd(&dweight_shared[i], norm_bti * (float)dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < C; i += blockDim.x)
    {
        atomicAddX(&dbias[i], (Tparams)dbias_shared[i]);
        atomicAddX(&dweight[i], (Tparams)dweight_shared[i]);
    }
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward2(Tdinp* dinp,
                         Tparams* dweight,
                         Tparams* dbias,
                         const Tdout* dout,
                         const Trest* inp,
                         const Tparams* weight,
                         const Trest* mean,
                         const Trest* rstd,
                         int B,
                         int T,
                         int C,
                         const int block_size)
{
    const int N = B * T;
    const int grid_size = ceil_div(32 * N, block_size);
    size_t shared_mem_size = 2 * C * sizeof(float);
    layernorm_backward_kernel2<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
}

// one wrap compute (BTs / wraps) C
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
__global__ void layernorm_backward_kernel3(Tdinp* dinp,
                                           Tparams* dweight,
                                           Tparams* dbias,
                                           const Tdout* dout,
                                           const Trest* inp,
                                           const Tparams* weight,
                                           const Trest* mean,
                                           const Trest* rstd,
                                           int B,
                                           int T,
                                           int C)
{
    extern __shared__ float shared[]; // size = 2 * C
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // 计算warp id
    int base_idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    #pragma unroll 4
    for (int i = threadIdx.x; i < C; i += blockDim.x)
    {
        dbias_shared[i] = 0.0f;
        dweight_shared[i] = 0.0f;
    }
    __syncthreads();

    // wraps_in_grid代表总的wraps = blocks * wraps_per_block
    int wraps_in_grid = gridDim.x * warp.meta_group_size();
    // 每个wrap需要计算BTs / wraps_in_grid 个C
    for (int idx = base_idx; idx < B * T; idx += wraps_in_grid)
    {
        // ony C by one warp 单个C的计算过程
        int b = idx / T;
        int t = idx % T;
        const Tdout* dout_bt = dout + b * T * C + t * C;
        const Trest* inp_bt = inp + b * T * C + t * C;
        Tdinp* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;

        // 由one warp中的32个线程进行规约计算
        for (int i = warp.thread_rank(); i < C; i += warp.size())
        {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
        dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>{});
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        for (int i = warp.thread_rank(); i < C; i += warp.size())
        {
            float dout_i = (float)__ldcs(&dout_bt[i]);
            float norm_bti = ((float)__ldcs(&inp_bt[i]) - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            atomicAdd(&dbias_shared[i], dout_i);
            // gradient contribution to weight
            atomicAdd(&dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < C; i += blockDim.x)
    {
        atomicAddX(&dbias[i], (Tparams)dbias_shared[i]);
        atomicAddX(&dweight[i], (Tparams)dweight_shared[i]);
    }
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward3(Tdinp* dinp,
                         Tparams* dweight,
                         Tparams* dbias,
                         const Tdout* dout,
                         const Trest* inp,
                         const Tparams* weight,
                         const Trest* mean,
                         const Trest* rstd,
                         int B,
                         int T,
                         int C,
                         const int block_size)
{
    const int grid_size = (1024 / block_size) * cuda_num_SMs;
    size_t shared_mem_size = 2 * C * sizeof(float);
    layernorm_backward_kernel3<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
}

void layernorm_backward(int kernel_num,
                        floatX* dinp,
                        floatX* dweight,
                        floatX* dbias,
                        const floatX* dout,
                        const floatX* inp,
                        const floatX* weight,
                        const floatX* mean,
                        const floatX* rstd,
                        int B,
                        int T,
                        int C,
                        const int block_size)
{
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    switch(kernel_num) {
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        case 1:
            layernorm_backward1(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
#endif
        case 2:
            layernorm_backward2(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 3:
            layernorm_backward3(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
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

    // first do the forward pass in CPU
    float* out = (float*) malloc(B * T * C * sizeof(float));
    float* mean = (float*) malloc(B * T * sizeof(float));
    float* rstd = (float*) malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);
    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // now do the backward pass, again on CPU
    float* dout = make_random_float(B * T * C); // 损失函数输出
    float* dinp = make_zeros_float(B * T * C); // x梯度
    float* dweight = make_zeros_float(C); // w梯度
    float* dbias = make_zeros_float(C); // b梯度
    auto start = std::chrono::high_resolution_clock::now();
    layernorm_backward_cpu(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
    auto stop = std::chrono::high_resolution_clock::now();
    printf("layernorm_backward_cpu cost: %lld ms.\n", std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
    printf("layernorm_backward_cpu dweight[%d] = %f \n", C / 2, dweight[(C / 2)]);
    printf("layernorm_backward_cpu dbias[%d] = %f \n", C / 2, dbias[(C / 2)]);

    floatX* meanX = (floatX*)malloc(B * T * sizeof(floatX));
    floatX* rstdX = (floatX*)malloc(B * T * sizeof(floatX));
    floatX* doutX = (floatX*)malloc(B * T * C * sizeof(floatX));
    floatX* inpX = (floatX*)malloc(B * T * C * sizeof(floatX));
    floatX* weightX = (floatX*)malloc(C * sizeof(floatX));

    for (int i = 0; i < B * T; i++) {
        meanX[i] = (floatX)mean[i];
        rstdX[i] = (floatX)rstd[i];
    }
    for (int i = 0; i < B * T * C; i++) {
        doutX[i] = (floatX)dout[i];
        inpX[i] = (floatX)inp[i];
    }
    for (int i = 0; i < C; i++) {
        weightX[i] = (floatX)weight[i];
    }

    const int block_size = 256;
    floatX* d_dinp;
    floatX* d_dweight;
    floatX* d_dbias;
    floatX* d_dout;
    floatX* d_inp;
    floatX* d_weight;
    floatX* d_mean;
    floatX* d_rstd;
    float* d_scratch;
    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dweight, C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dbias, C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_mean, B * T * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_rstd, B * T * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_scratch, cuda_num_SMs * (2 * C + 1) * sizeof(float)));

    // copy over the "inputs" to the backward call
    cudaCheck(cudaMemcpy(d_dout, doutX, B * T * C * sizeof(floatX), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_inp, inpX, B * T * C * sizeof(floatX), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weightX, C * sizeof(floatX), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_mean, meanX, B * T * sizeof(floatX), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_rstd, rstdX, B * T * sizeof(floatX), cudaMemcpyHostToDevice));
   
    float error_threshold_dinp = sizeof(floatX) == 4 ? 1e-3f : 1e-1f;
    float error_threshold_dparams = sizeof(floatX) == 4 ? 1e-3f : 20.0f;

    for (int i = 1; i <= 3; i++) {
        cudaCheck(cudaMemset(d_dinp, 0, B * T * C * sizeof(floatX)));  // x梯度
        cudaCheck(cudaMemset(d_dweight, 0, C * sizeof(floatX))); // w梯度
        cudaCheck(cudaMemset(d_dbias, 0, C * sizeof(floatX))); // b梯度
        layernorm_backward(i, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_mean, d_rstd, B, T, C, block_size);
        printf("Checking correctness for layernorm_backward%d\n", i);
        printf("dinp:\n");
        validate_result(d_dinp, dinp, "dinp", B * T * C, error_threshold_dinp);
        printf("dweight:\n");
        validate_result(d_dweight, dweight, "dweight", C, error_threshold_dparams);
        printf("dbias:\n");
        validate_result(d_dbias, dbias, "dbias", C, error_threshold_dparams);
    }

    // cleanups
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);
    free(dout);
    free(dinp);
    free(dweight);
    free(dbias);
    free(meanX);
    free(rstdX);
    free(doutX);
    free(inpX);
    free(weightX);
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dweight));
    cudaCheck(cudaFree(d_dbias));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_mean));
    cudaCheck(cudaFree(d_rstd));
    cudaCheck(cudaFree(d_scratch));
    return 0;
}

