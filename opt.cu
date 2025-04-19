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

const char* version_name = "Optimized implementation.";

__device__  float4 load_float4(float* A, int row, int col, int n){
    float4 tmp_vec = {0.0f, 0.0f, 0.0f, 0.0f};
    int index = row * n + col;
    if (index % 4 == 0 && row < n && col + 3 < n){
        tmp_vec = reinterpret_cast<float4*>(&A[index])[0];
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++){
            bool valid = (row < n) && (col + i < n);
            ((float*)&tmp_vec)[i] = valid ? A[index + i] : 0.0f;
        }
    }
    return tmp_vec;
}

// template <int TILE>
// __global__ void SgemmVecKernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int n) {
//     __shared__ float4 smemA[2][TILE][TILE / 8 + 2];
//     __shared__ float4 smemB[2][TILE][TILE / 8 + 2];

//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;

//     const int blockRow = blockIdx.y * TILE;
//     const int blockCol = blockIdx.x * TILE;

//     const int threadRow = ty * 2;
//     const int threadCol = tx * 8;

//     float c_reg[4][4] = {{0}};

//     // 预加载 A 和 B 的块到 smemA 和 smemB
//     int curr = 0, next = 1;
//     smemA[curr][threadRow][tx] = load_float4(A, blockRow + threadRow, threadCol, n);
//     smemA[curr][threadRow + 1][tx] = load_float4(A, blockRow + threadRow + 1, threadCol, n);
//     smemB[curr][threadRow][tx] = load_float4(B, blockRow + threadRow, threadCol, n);
//     smemB[curr][threadRow + 1][tx] = load_float4(B, blockRow + threadRow + 1, threadCol, n);
//     __syncthreads();

//     for (int k = TILE; k < n; k += TILE) {
//         /*加载 A 的块到 smemA（行主序), 要进行转置才能使用 float4 连续加载到 reg中，需要保证arow 是 4的倍数
//         A, B的封装是独立的，理论上最好的方法应该对A，B单独设计封装核*/ 

//         for (int load = 0; load < 2; load ++) {
//             const int arow = blockRow + threadRow + load;
//             const int acol = k + threadCol;
//             smemA[next][threadRow + load][tx] = load_float4(A, arow, acol, n);
//             // 加载 B 的块到 smemB
//             const int brow = k + threadRow + load;
//             const int bcol = blockCol + threadCol;
//             smemB[next][threadRow + load][tx] = load_float4(B, brow, bcol, n);
//         }
//         __syncthreads();

//         // 乘累加计算
//         for (int i_group = 0; i_group < TILE / 4; i_group ++){
//             float4 b_reg = smemB[curr][i_group][tx];
//             float4 a_reg = smemA[curr][i_group][ty];
//             #pragma unroll
//             for (int x = 0; x < 4; x++) {
//                 float a_val = ((float*) & a_reg)[x];
//                 #pragma unroll
//                 for (int y = 0; y < 4; y++) {
//                     float b_val = ((float*) & b_reg)[y];
//                     c_reg[x][y] += a_val * b_val;
//                 }
//             }
//         }
//         __syncthreads();

//         // 交换 smemA 和 smemB 的指针
//         curr = 1 - curr;
//         next = 1 - next;
//     }

//     // 写回结果（行主序）
//     const int write_row_start = blockRow + threadRow;
//     const int write_col_start = blockCol + threadCol;

//     for (int x = 0; x < 4; x++) {
//         const int write_row = write_row_start + x;
//         if (write_row >= n) break;
//         for (int y = 0; y < 4; y++) {
//             const int write_col = write_col_start + y;
//             if (write_col < n) {
//                 C[write_row * n + write_col] = c_reg[x][y];
//             }
//         }
//     }
// }
// template <int TILE>
// __global__ void SgemmVecTransKernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int n) {
//     __shared__ float4 smemA[2][TILE][TILE / 8 + 2];
//     __shared__ float4 smemB[2][TILE][TILE / 8 + 2];

//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;

//     const int blockRow = blockIdx.y * TILE;
//     const int blockCol = blockIdx.x * TILE;

//     const int threadRow = ty * 2;
//     const int threadCol = tx * 8;
    
//     float c_reg[4][4] = {{0}};

//     // 预加载 A 和 B 的块到 smemA 和 smemB
//     int curr = 0, next = 1;
//     smemA[curr][threadRow][tx] = load_float4(A, blockRow + threadRow, threadCol, n);
//     smemA[curr][threadRow + 1][tx] = load_float4(A, blockRow + threadRow + 1, threadCol, n);
//     smemB[curr][threadRow][tx] = load_float4(B, blockRow + threadRow, threadCol, n);
//     smemB[curr][threadRow + 1][tx] = load_float4(B, blockRow + threadRow + 1, threadCol, n);
//     __syncthreads();

//     for (int k = TILE; k < n; k += TILE) {
//         /*加载 A 的块到 smemA（行主序), 要进行转置才能使用 float4 连续加载到 reg中，需要保证arow 是 4的倍数，
//          方案是展开load，一次性读取4个load的值*/ 

//         for (int load = 0; load < 2; load ++) {
//             const int arow = blockRow + threadRow + load;
//             const int acol = k + threadCol;
//             smemA[next][threadRow + load][tx] = load_float4(A, arow, acol, n);
//             // 加载 B 的块到 smemB
//             const int brow = k + threadRow + load;
//             const int bcol = blockCol + threadCol;
//             smemB[next][threadRow + load][tx] = load_float4(B, brow, bcol, n);
//         }
//         __syncthreads();

//         // 乘累加计算
//         for (int i_group = 0; i_group < TILE / 4; i_group ++){
//             float4 b_reg = smemB[curr][i_group][ty];
//             float4 a_reg = smemA[curr][i_group][ty];
//             #pragma unroll
//             for (int x = 0; x < 4; x++) {
//                 float a_val = ((float*) & a_reg)[x];
//                 #pragma unroll
//                 for (int y = 0; y < 4; y++) {
//                     float b_val = ((float*) & b_reg)[y];
//                     c_reg[x][y] += a_val * b_val;
//                 }
//             }
//         }
//         __syncthreads();
//         // 交换 smemA 和 smemB 的指针
//         curr = 1 - curr;
//         next = 1 - next;
//     }
//     // 写回结果（行主序）
//     const int write_row_start = blockRow + threadRow;
//     const int write_col_start = blockCol + threadCol;

//     for (int x = 0; x < 4; x++) {
//         const int write_row = write_row_start + x;
//         if (write_row >= n) break;
//         for (int y = 0; y < 4; y++) {
//             const int write_col = write_col_start + y;
//             if (write_col < n) {
//                 C[write_row * n + write_col] = c_reg[x][y];
//             }
//         }
//     }
// }
template <int TILE>
__global__ void SgemmOptimizedLargeKernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int n, bool trans) {
    __shared__ float4 smemA[2][TILE][TILE / 8 + 2];
    __shared__ float4 smemB[2][TILE][TILE / 8 + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int blockRow = blockIdx.y * TILE;
    const int blockCol = blockIdx.x * TILE;

    const int threadRow = ty * 2;
    const int threadCol = tx * 8;

    float c_reg[4][4] = {{0}};
    int true_index = (trans) ? ty : tx;

    // 预加载 A 和 B 的块到 smemA 和 smemB
    int curr = 0, next = 1;
    smemA[curr][threadRow][tx] = load_float4(A, blockRow + threadRow, threadCol, n);
    smemA[curr][threadRow + 1][tx] = load_float4(A, blockRow + threadRow + 1, threadCol, n);
    smemB[curr][threadRow][tx] = load_float4(B, blockRow + threadRow, threadCol, n);
    smemB[curr][threadRow + 1][tx] = load_float4(B, blockRow + threadRow + 1, threadCol, n);
    __syncthreads();

    for (int k = TILE; k < n; k += TILE) {
        /*加载 A 的块到 smemA（行主序), 要进行转置才能使用 float4 连续加载到 reg中，需要保证arow 是 4的倍数
        A, B的封装是独立的，理论上最好的方法应该对A，B单独设计封装核*/ 

        for (int load = 0; load < 2; load ++) {
            const int arow = blockRow + threadRow + load;
            const int acol = k + threadCol;
            smemA[next][threadRow + load][tx] = load_float4(A, arow, acol, n);
            // 加载 B 的块到 smemB
            const int brow = k + threadRow + load;
            const int bcol = blockCol + threadCol;
            smemB[next][threadRow + load][tx] = load_float4(B, brow, bcol, n);
        }
        __syncthreads();

        // 乘累加计算
        for (int i_group = 0; i_group < TILE / 4; i_group ++){
            float4 b_reg = smemB[curr][i_group][true_index];
            float4 a_reg = smemA[curr][i_group][ty];
            #pragma unroll
            for (int x = 0; x < 4; x++) {
                float a_val = ((float*) & a_reg)[x];
                #pragma unroll
                for (int y = 0; y < 4; y++) {
                    float b_val = ((float*) & b_reg)[y];
                    c_reg[x][y] += a_val * b_val;
                }
            }
        }
        __syncthreads();

        // 交换 smemA 和 smemB 的指针
        curr = 1 - curr;
        next = 1 - next;
    }

    // 写回结果（行主序）
    const int write_row_start = blockRow + threadRow;
    const int write_col_start = blockCol + threadCol;

    for (int x = 0; x < 4; x++) {
        const int write_row = write_row_start + x;
        if (write_row >= n) break;
        for (int y = 0; y < 4; y++) {
            const int write_col = write_col_start + y;
            if (write_col < n) {
                C[write_row * n + write_col] = c_reg[x][y];
            }
        }
    }
}
template <int TILE>
__global__ void SgemmOptimizedSmallKernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int n, bool trans) {
    __shared__ float4 smemA[TILE][TILE / 8 + 2];
    __shared__ float4 smemB[TILE][TILE / 8 + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int blockRow = blockIdx.y * TILE;
    const int blockCol = blockIdx.x * TILE;

    const int threadRow = ty * 2;
    const int threadCol = tx * 8;

    float c_reg[4][4] = {{0}};
    int true_index = (trans) ? ty : tx;

    for (int k = 0; k < n; k += TILE) {
        /*加载 A 的块到 smemA（行主序), 要进行转置才能使用 float4 连续加载到 reg中，需要保证arow 是 4的倍数
        A, B的封装是独立的，理论上最好的方法应该对A，B单独设计封装核*/ 

        for (int load = 0; load < 2; load ++) {
            const int arow = blockRow + threadRow + load;
            const int acol = k + threadCol;
            smemA[threadRow + load][tx] = load_float4(A, arow, acol, n);
            // 加载 B 的块到 smemB
            const int brow = k + threadRow + load;
            const int bcol = blockCol + threadCol;
            smemB[threadRow + load][tx] = load_float4(B, brow, bcol, n);
        }
        __syncthreads();

        // 乘累加计算
        for (int i_group = 0; i_group < TILE / 4; i_group ++){
            float4 b_reg = smemB[i_group][true_index];
            float4 a_reg = smemA[i_group][ty];
            #pragma unroll
            for (int x = 0; x < 4; x++) {
                float a_val = ((float*) & a_reg)[x];
                #pragma unroll
                for (int y = 0; y < 4; y++) {
                    float b_val = ((float*) & b_reg)[y];
                    c_reg[x][y] += a_val * b_val;
                }
            }
        }
        __syncthreads();
    }

    // 写回结果（行主序）
    const int write_row_start = blockRow + threadRow;
    const int write_col_start = blockCol + threadCol;

    for (int x = 0; x < 4; x++) {
        const int write_row = write_row_start + x;
        if (write_row >= n) break;
        for (int y = 0; y < 4; y++) {
            const int write_col = write_col_start + y;
            if (write_col < n) {
                C[write_row * n + write_col] = c_reg[x][y];
            }
        }
    }
}

// template <int TILE>
// __global__ void SgemmDynamicTransKernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int n){
//     __shared__ float smemA[TILE][TILE + 1];
//     __shared__ float smemB[TILE][TILE + 1];

//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;

//     const int blockRow = blockIdx.y * TILE;
//     const int blockCol = blockIdx.x * TILE;

//     const int threadRow = ty * 4;
//     const int threadCol = tx * 4;

//     float c_reg[4][4] = {{0}};
//     for (int k = 0; k < n; k += TILE) {
//         // 加载 A 的块到 smemA（行主序）
//         for (int load = 0; load < 4; load ++) {
//             const int arow = blockRow + threadRow + load;
//             const int acol = k + threadCol;

//             #pragma unroll
//             for (int i = 0; i < 4; i++) {
//                 bool valid = (arow < n) && (acol + i < n);
//                 smemA[threadRow + load][threadCol + i] = valid ? A[arow * n + (acol + i)] : 0.0f;
//             }

//             // 加载 B 的块到 smemB (转置形式)
//             const int brow = k + threadRow + load;
//             const int bcol = blockCol + threadCol;

//             #pragma unroll
//             for (int i = 0; i < 4; i++) {
//                 bool valid = (brow < n) && (bcol + i < n);
//                 smemB[threadRow + load][threadCol + i] = valid ? B[(bcol + i) * n + brow] : 0.0f;
//             }
//         }
//         __syncthreads();

//         // 乘累加计算
//         for (int i = 0; i < TILE; i++) {
//             float a_reg[4], b_reg[4];
//             #pragma unroll
//             for (int x = 0; x < 4; x++) {
//                 a_reg[x] = smemA[threadRow + x][i];
//             }
//             #pragma unroll
//             for (int y = 0; y < 4; y++) {
//                 b_reg[y] = smemB[i][threadCol + y];
//             }
//             #pragma unroll
//             for (int x = 0; x < 4; x++) {
//                 #pragma unroll
//                 for (int y = 0; y < 4; y++) {
//                     c_reg[x][y] += a_reg[x] * b_reg[y];
//                 }
//             }
//         }
//         __syncthreads();
//     }

//     // 写回结果（行主序）
//     const int write_row_start = blockRow + threadRow;
//     const int write_col_start = blockCol + threadCol;

//     for (int x = 0; x < 4; x++) {
//         const int write_row = write_row_start + x;
//         if (write_row >= n) break;
//         for (int y = 0; y < 4; y++) {
//             const int write_col = write_col_start + y;
//             if (write_col < n) {
//                 C[write_row * n + write_col] = c_reg[x][y];
//             }
//         }
//     }
// }
// template <int TILE>
// __global__ void SgemmDynamicKernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int n) {
//     __shared__ float smemA[TILE][TILE + 1];
//     __shared__ float smemB[TILE][TILE + 1];

//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;

//     const int blockRow = blockIdx.y * TILE;
//     const int blockCol = blockIdx.x * TILE;

//     const int threadRow = ty * 4;
//     const int threadCol = tx * 4;

//     float c_reg[4][4] = {{0}};

//     for (int k = 0; k < n; k += TILE) {
//         // 加载 A 的块到 smemA（行主序）
//         for (int load = 0; load < 4; load ++) {
//             const int arow = blockRow + threadRow + load;
//             const int acol = k + threadCol;

//             #pragma unroll
//             for (int i = 0; i < 4; i++) {
//                 bool valid = (arow < n) && (acol + i < n);
//                 smemA[threadRow + load][threadCol + i] = valid ? A[arow * n + (acol + i)] : 0.0f;
//             }

//             // 加载 B 的块到 smemB
//             const int brow = k + threadRow + load;
//             const int bcol = blockCol + threadCol; 
//             #pragma unroll
//             for (int i = 0; i < 4; i++) {
//                 bool valid = (brow < n) && (bcol + i < n);
//                 smemB[threadRow + load][threadCol + i] = valid ? B[brow * n + (bcol + i)] : 0.0f;
//             }
//         }
//         __syncthreads();

//         // 乘累加计算
//         for (int i = 0; i < TILE; i++) {
//             float a_reg[4], b_reg[4];
//             #pragma unroll
//             for (int x = 0; x < 4; x++) {
//                 a_reg[x] = smemA[threadRow + x][i];
//             }
//             #pragma unroll
//             for (int y = 0; y < 4; y++) {
//                 b_reg[y] = smemB[i][threadCol + y];
//             }
//             #pragma unroll
//             for (int x = 0; x < 4; x++) {
//                 #pragma unroll
//                 for (int y = 0; y < 4; y++) {
//                     c_reg[x][y] += a_reg[x] * b_reg[y];
//                 }
//             }
//         }
//         __syncthreads();
//     }
//     // 写回结果（行主序）
//     const int write_row_start = blockRow + threadRow;
//     const int write_col_start = blockCol + threadCol;

//     for (int x = 0; x < 4; x++) {
//         const int write_row = write_row_start + x;
//         if (write_row >= n) break;
//         for (int y = 0; y < 4; y++) {
//             const int write_col = write_col_start + y;
//             if (write_col < n) {
//                 C[write_row * n + write_col] = c_reg[x][y];
//             }
//         }
//     }
// }
__global__ void scale_kernel(float* matrix, int n, float scale){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n * n){
        matrix[idx] *= scale;
    }
}

// __global__ void softmax_kernel(float* matrix, int n){
//     int row = blockIdx.x;
//     int tid = threadIdx.x;
//     extern __shared__ float shared_data[];
    
//     float max_val = -INFINITY;
//     for (int j = tid; j < n; j += blockDim.x) {
//         max_val = fmaxf(max_val, matrix[row * n + j]);
//     }
//     shared_data[tid] = max_val;
//     __syncthreads();

//     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + s]);
//         }
//         __syncthreads();
//     }
//     max_val = shared_data[0];

//     float sum = 0.0f;
//     for (int j = tid; j < n; j += blockDim.x){
//         float val = expf(matrix[row * n + j] - max_val);
//         matrix[row * n + j] = val;
//         sum += val;
//     }
//     shared_data[tid] = sum;
//     __syncthreads();

//     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             shared_data[tid] += shared_data[tid + s];
//         }
//         __syncthreads();
//     }
//     sum = shared_data[0];

//     for (int j = tid; j < n; j += blockDim.x){
//         matrix[row * n + j] /= sum;
//     }
// }

// online softmax kernel
__global__ void OnlineSoftmaxKernel(float* __restrict__ matrix, int n){
    // 每个线程块处理一行
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float shared_data[];
    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;

    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float old_max = 0.0f;

    // 每个线程计算一部分最大值和指数和
    for (int j = 0; j < n; j += blockDim.x){
        float val = matrix[row * n + j];
        old_max = max_val;
        max_val = fmaxf(max_val, val);
        sum_exp = sum_exp * expf(old_max - max_val) + expf(val - max_val);
    }

    // warp内进行归约最大值和指数和
    for (int offset = warpSize / 2; offset > 0; offset >>= 1){
        float tmp_max = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        float tmp_sum = __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
        if (lane_id < offset){
            // for tid < offset, sum_exp->tid, max_val->tid, tmp_sum->tid + offset, tmp_max->tid + offset
            old_max = max_val;
            max_val = fmaxf(max_val, tmp_max);
            sum_exp = sum_exp * expf(old_max - max_val) + tmp_sum * expf(tmp_max - max_val);
        }
    }

    if(lane_id == 0){
        shared_data[warp_id * 2] = max_val;
        shared_data[warp_id * 2 + 1] = sum_exp;
    }
    __syncthreads();

    // 全局归约
    if (warp_id == 0){
        if (tid < blockDim.x / warpSize){
            max_val = shared_data[tid * 2];
            sum_exp = shared_data[tid * 2 + 1];
        } else {
            max_val = INFINITY;
            sum_exp = 0.0f;
        }
        for (int offset = 1; offset < (blockDim.x + warpSize - 1) / warpSize; offset <<= 1){
            float tmp_max = shared_data[offset * 2];
            float tmp_sum = shared_data[offset * 2 + 1];
            old_max = max_val;
            max_val = fmaxf(max_val, tmp_max);
            sum_exp = sum_exp * expf(old_max - max_val) + tmp_sum * expf(tmp_max - max_val);
        }
    
        shared_data[0] = max_val;
        shared_data[1] = sum_exp;
    }
    __syncthreads();
    const float global_max = shared_data[0];
    const float global_sum = shared_data[1];
    // 计算softmax
    for (int j = tid; j < n; j += blockDim.x){
        float val = expf(matrix[row * n + j] - global_max);
        matrix[row * n + j] = val / global_sum;
    }
}

template <int TILE>
void launch_kernel(int n, float* gpu_Q, float* gpu_K, float* gpu_V, float* gpu_Y, float* gpu_QK_T){
    dim3 block_trans(TILE/8, TILE/2);
    dim3 grid_trans((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    // size_t smem_trans = sizeof(float) * (TILE + 1) * TILE * 2;
    SgemmOptimizedSmallKernel<TILE><<<grid_trans, block_trans>>>(gpu_Q, gpu_K, gpu_QK_T, n, true);
    CHECK_CUDA(cudaGetLastError());

    dim3 block_scale(256);
    dim3 grid_scale((n*n + block_scale.x - 1) / block_scale.x);
    scale_kernel<<<grid_scale, block_scale>>>(gpu_QK_T, n, 1.0f / sqrtf(n));
    CHECK_CUDA(cudaGetLastError());

    dim3 block_softmax(256);
    dim3 grid_softmax(n);
    // softmax_kernel<<<grid_softmax, block_softmax, block_softmax.x * sizeof(float)>>>(gpu_QK_T, n);
    OnlineSoftmaxKernel<<<grid_softmax, block_softmax, block_softmax.x / 32 * 2 * sizeof(float)>>>(gpu_QK_T, n);
    CHECK_CUDA(cudaGetLastError());

    dim3 block_mul(TILE/8, TILE/2);
    dim3 grid_mul((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    // size_t smem_mul = sizeof(float) * (TILE + 1) * TILE * 2;
    SgemmOptimizedSmallKernel<TILE><<<grid_mul, block_mul>>>(gpu_QK_T, gpu_V, gpu_Y, n, false);
    CHECK_CUDA(cudaGetLastError());
}

void square_attention (int n, float* gpu_Q, float* gpu_K, float* gpu_V, float* gpu_Y){

    float* gpu_QK_T;
    size_t size = n * n * sizeof(float);

    CHECK_CUDA(cudaMalloc((void**)&gpu_QK_T, size));

    if (n <= 128) {
        launch_kernel<32>(n, gpu_Q, gpu_K, gpu_V, gpu_Y, gpu_QK_T);
    } else if (n <= 4000){
        launch_kernel<64>(n, gpu_Q, gpu_K, gpu_V, gpu_Y, gpu_QK_T);
    } else {
        int TILE = 64;
        dim3 block_trans(TILE/8, TILE/2);
        dim3 grid_trans((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        // size_t smem_trans = sizeof(float) * (TILE + 1) * TILE * 2;
        SgemmOptimizedLargeKernel<64><<<grid_trans, block_trans>>>(gpu_Q, gpu_K, gpu_QK_T, n, true);
        CHECK_CUDA(cudaGetLastError());
    
        dim3 block_scale(256);
        dim3 grid_scale((n*n + block_scale.x - 1) / block_scale.x);
        scale_kernel<<<grid_scale, block_scale>>>(gpu_QK_T, n, 1.0f / sqrtf(n));
        CHECK_CUDA(cudaGetLastError());
    
        dim3 block_softmax(256);
        dim3 grid_softmax(n);
        // softmax_kernel<<<grid_softmax, block_softmax, block_softmax.x * sizeof(float)>>>(gpu_QK_T, n);
        OnlineSoftmaxKernel<<<grid_softmax, block_softmax, block_softmax.x / 32 * 2 * sizeof(float)>>>(gpu_QK_T, n);
        CHECK_CUDA(cudaGetLastError());
    
        dim3 block_mul(TILE/8, TILE/2);
        dim3 grid_mul((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        // size_t smem_mul = sizeof(float) * (TILE + 1) * TILE * 2;
        SgemmOptimizedLargeKernel<64><<<grid_mul, block_mul>>>(gpu_QK_T, gpu_V, gpu_Y, n, false);
        CHECK_CUDA(cudaGetLastError()); 
    }
    CHECK_CUDA(cudaFree(gpu_QK_T));
}
