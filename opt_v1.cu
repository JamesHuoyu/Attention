#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <math.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
#define TILE 32

const char* version_name = "Optimized implementation.";


__global__ void SgemmNaiveTransKernel(float* A, float* B, float* C, int n){
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < n && col < n){
        float sum = 0.0f;
        for(int i = 0; i < n; i++){
            sum += A[row * n + i] * B[col * n + i];
        }
        C[row * n + col] += sum;
    }
}

__global__ void SgemmNaiveKernel(float* A, float* B, float* C, int n){
    __shared__ float smemA[TILE][TILE + 1];
    __shared__ float smemB[TILE][TILE + 1];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float c_reg = 0.0f;

    for (int k = 0; k < n; k += TILE){
        if (row < n && k + tx < n){
            smemA[ty][tx] = A[row * n + k + tx];
        } else {
            smemA[ty][tx] = 0.0f;
        }

        if (col < n && k + ty < n){
            smemB[ty][tx] = B[(k + ty) * n + col];
        } else {
            smemB[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE; i++){
            c_reg += smemA[ty][i] * smemB[i][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n){
        C[row * n + col] += c_reg;
    }
    
    // const int row = blockIdx.y * blockDim.y + threadIdx.y;
    // const int col = blockIdx.x * blockDim.x + threadIdx.x;

    // if(row < n && col < n){
    //     float sum = 0.0f;
    //     for(int i = 0; i < n; i++){
    //         sum += A[row * n + i] * B[i * n + col];
    //     }
    //     C[row * n + col] += sum;
    // }
}

__global__ void scale_kernel(float* matrix, int n, float scale){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n * n){
        matrix[idx] *= scale;
    }
}

__global__ void softmax_kernel(float* matrix, int n){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float shared_data[];
    
    float max_val = -INFINITY;
    for (int j = tid; j < n; j += blockDim.x) {
        max_val = fmaxf(max_val, matrix[row * n + j]);
    }
    shared_data[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + s]);
        }
        __syncthreads();
    }
    max_val = shared_data[0];

    float sum = 0.0f;
    for (int j = tid; j < n; j += blockDim.x){
        float val = expf(matrix[row * n + j] - max_val);
        matrix[row * n + j] = val;
        sum += val;
    }
    shared_data[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    sum = shared_data[0];

    for (int j = tid; j < n; j += blockDim.x){
        matrix[row * n + j] /= sum;
    }
}

void square_attention (int n, float* gpu_Q, float* gpu_K, float* gpu_V, float* gpu_Y){

    float* gpu_QK_T;
    size_t size = n * n * sizeof(float);

    CHECK_CUDA(cudaMalloc((void**)&gpu_QK_T, size));

    dim3 block_multrans(32, 32);
    dim3 grid_multrans((n + block_multrans.x - 1) / block_multrans.x, (n + block_multrans.y - 1) / block_multrans.y);
    SgemmNaiveTransKernel<<<grid_multrans, block_multrans>>>(gpu_Q, gpu_K, gpu_QK_T, n);
    CHECK_CUDA(cudaGetLastError());

    dim3 block_scale(256);
    dim3 grid_scale((n*n + block_scale.x - 1) / block_scale.x);
    scale_kernel<<<grid_scale, block_scale>>>(gpu_QK_T, n, 1.0f / sqrtf(n));
    CHECK_CUDA(cudaGetLastError());

    dim3 block_softmax(256);
    dim3 grid_softmax(n);
    softmax_kernel<<<grid_softmax, block_softmax, block_softmax.x * sizeof(float)>>>(gpu_QK_T, n);
    CHECK_CUDA(cudaGetLastError());

    dim3 block_mul(32, 32);
    dim3 grid((n + block_mul.x - 1) / block_mul.x, (n + block_mul.y - 1) / block_mul.y);
    size_t smem = sizeof(float) * (TILE + 1) * TILE * 2;
    SgemmNaiveKernel<<<grid, block_mul, smem>>>(gpu_QK_T, gpu_V, gpu_Y, n);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(gpu_QK_T));
}
