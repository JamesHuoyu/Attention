#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define CHECK_CUDA(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
#define TILE 32

const char* version_name = "Optimized implementation.";

__global__ void SgemmNaiveTransKernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int n){
    __shared__ float smemA[TILE][TILE + 1];
    __shared__ float smemB[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int blockRow = blockIdx.y * TILE;
    const int blockCol = blockIdx.x * TILE;

    const int threadRow = ty * 4;
    const int threadCol = tx * 4;

    for (int k = 0; k < n; k += TILE) {
        float c_reg[4][4] = {{0}};
        // 加载 A 的块到 smemA（行主序）
        for (int load = 0; load < 4; load ++) {
            const int arow = blockRow + threadRow + load;
            const int acol = k + threadCol;

            if (arow < n && acol < n) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    smemA[threadRow + load][threadCol + i] = (acol + i < n) ? A[arow * n + (acol + i)] : 0.0f;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    smemA[threadRow + load][threadCol + i] = 0.0f;
                }
            }

            // 加载 B 的块到 smemB (转置形式)
            const int brow = k + threadRow + load;
            const int bcol = blockCol + threadCol;

            if (brow < n && bcol < n) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    smemB[threadRow + load][threadCol + i] = (bcol + i < n) ? B[(bcol + i) * n + brow] : 0.0f;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    smemB[threadRow + load][threadCol + i] = 0.0f;
                }
            }
        }
        __syncthreads();

        // 乘累加计算
        for (int i = 0; i < TILE; i++) {
            float a_reg[4], b_reg[4];
            #pragma unroll
            for (int x = 0; x < 4; x++) {
                a_reg[x] = smemA[threadRow + x][i];
            }
            #pragma unroll
            for (int y = 0; y < 4; y++) {
                b_reg[y] = smemB[i][threadCol + y];
            }
            #pragma unroll
            for (int x = 0; x < 4; x++) {
                #pragma unroll
                for (int y = 0; y < 4; y++) {
                    c_reg[x][y] += a_reg[x] * b_reg[y];
                }
            }
        }
        __syncthreads();

        // 写回结果（行主序）
        const int write_row_start = blockRow + threadRow;
        const int write_col_start = blockCol + threadCol;

        for (int x = 0; x < 4; x++) {
            const int write_row = write_row_start + x;
            if (write_row >= n) break;
            for (int y = 0; y < 4; y++) {
                const int write_col = write_col_start + y;
                if (write_col < n) {
                    atomicAdd(&C[write_row * n + write_col], c_reg[x][y]);
                }
            }
        }
    }
}
__global__ void SgemmNaiveKernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int n) {
    __shared__ float smemA[TILE][TILE + 1];
    __shared__ float smemB[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int blockRow = blockIdx.y * TILE;
    const int blockCol = blockIdx.x * TILE;

    const int threadRow = ty * 4;
    const int threadCol = tx * 4;

    for (int k = 0; k < n; k += TILE) {
        float c_reg[4][4] = {{0}};
        // 加载 A 的块到 smemA（行主序）
        for (int load = 0; load < 4; load ++) {
            const int arow = blockRow + threadRow + load;
            const int acol = k + threadCol;

            if (arow < n && acol < n) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    smemA[threadRow + load][threadCol + i] = (acol + i < n) ? A[arow * n + (acol + i)] : 0.0f;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    smemA[threadRow + load][threadCol + i] = 0.0f;
                }
            }

            // 加载 B 的块到 smemB
            const int brow = k + threadRow + load;
            const int bcol = blockCol + threadCol; // 使用 ty 而非 threadIdx.y

            if (brow < n && bcol < n) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    smemB[threadRow + load][threadCol + i] = (bcol + i < n) ? B[brow * n + (bcol + i)] : 0.0f;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    smemB[threadRow + load][threadCol + i] = 0.0f;
                }
            }
        }
        __syncthreads();

        // 乘累加计算
        for (int i = 0; i < TILE; i++) {
            float a_reg[4], b_reg[4];
            #pragma unroll
            for (int x = 0; x < 4; x++) {
                a_reg[x] = smemA[threadRow + x][i];
            }
            #pragma unroll
            for (int y = 0; y < 4; y++) {
                b_reg[y] = smemB[i][threadCol + y];
            }
            #pragma unroll
            for (int x = 0; x < 4; x++) {
                #pragma unroll
                for (int y = 0; y < 4; y++) {
                    c_reg[x][y] += a_reg[x] * b_reg[y];
                }
            }
        }
        __syncthreads();

        // 写回结果（行主序）
        const int write_row_start = blockRow + threadRow;
        const int write_col_start = blockCol + threadCol;

        for (int x = 0; x < 4; x++) {
            const int write_row = write_row_start + x;
            if (write_row >= n) break;
            for (int y = 0; y < 4; y++) {
                const int write_col = write_col_start + y;
                if (write_col < n) {
                    atomicAdd(&C[write_row * n + write_col], c_reg[x][y]);
                }
            }
        }
    }
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

    dim3 block_trans(TILE/4, TILE/4);
    dim3 grid_trans((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    size_t smem_trans = sizeof(float) * (TILE + 1) * TILE * 2;
    SgemmNaiveTransKernel<<<grid_trans, block_trans, smem_trans>>>(gpu_Q, gpu_K, gpu_QK_T, n);
    CHECK_CUDA(cudaGetLastError());

    dim3 block_scale(256);
    dim3 grid_scale((n*n + block_scale.x - 1) / block_scale.x);
    scale_kernel<<<grid_scale, block_scale>>>(gpu_QK_T, n, 1.0f / sqrtf(n));
    CHECK_CUDA(cudaGetLastError());

    dim3 block_softmax(256);
    dim3 grid_softmax(n);
    softmax_kernel<<<grid_softmax, block_softmax, block_softmax.x * sizeof(float)>>>(gpu_QK_T, n);
    CHECK_CUDA(cudaGetLastError());

    dim3 block_mul(TILE/4, TILE/4);
    dim3 grid_mul((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    size_t smem_mul = sizeof(float) * (TILE + 1) * TILE * 2;
    SgemmNaiveKernel<<<grid_mul, block_mul, smem_mul>>>(gpu_QK_T, gpu_V, gpu_Y, n);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(gpu_QK_T));
}
