#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include "utils/check_error.cuh"
#include "utils/tensor.cuh"

template <typename AT, typename BT, typename OT>
static void ensure_mm_shape_device(const Tensor<AT> &a, const Tensor<BT> &b, const Tensor<OT> &out)
{
    if (a.h != out.h || b.w != out.w || a.w != b.h)
        throw std::runtime_error("a,b,out tensor shape mismatch a:" +
            a.repr() + ", b:" + b.repr() + ", out:" + out.repr());

    if (a.on_device != b.on_device || a.on_device != out.on_device)
        throw std::runtime_error("a,b,out tensor device mismatch a:" + 
            a.repr() + ", b:" + b.repr() + ", out:" + out.repr());
}

template <typename T>
__global__ void mm_kernel(const Tensor<T> A, const Tensor<T> B, Tensor<T> C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A.h) {
        if (col < B.w) {
            T sum = 0;
            for (int k = 0 ; k < A.w ; k++) {
                sum += Index(A, row, k) * Index(B, k, col);
            }
            Index(C, row, col) = sum;
        }
            
    }
}


//compute C = A@B
template <typename T>
void op_mm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
{
    ensure_mm_shape_device(A,B,C);
    //Lab-1: please complete this
    //You need to define separate kernel function(s) and launch them her.
    // Configure block & grid sizes
    // int k = A.w
    // if (k < 1024) {
    //     const int BLOCK_DIM = std::sqrt(k);

    // } else {
    // }
    const int BLOCK_DIM = 16;

    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((C.w + BLOCK_DIM - 1) / BLOCK_DIM,
                 (C.h + BLOCK_DIM - 1) / BLOCK_DIM);

    // Launch kernel
    mm_kernel<<<gridDim, blockDim>>>(A, B, C);
    cudaDeviceSynchronize();
}

// Helper to get pointer offset for specific batch
// We assume inputs are flattened 2D tensors: [Batch_Count * M, K]
template <typename T>
__device__ T* get_batch_ptr(Tensor<T>& t, int batch_idx, int rows_per_batch) {
    // Skip 'rows_per_batch' rows for every batch index
    // offset = batch_idx * (rows_per_batch * stride_h)
    return t.rawp + t.offset + (batch_idx * rows_per_batch * t.stride_h);
}

// ops/op_bmm.cuh

template <typename T>
__global__ void bmm_kernel(Tensor<T> A, Tensor<T> B, Tensor<T> C, int M, int N, int K) 
{
    // batch_idx corresponds to (Batch * Heads) flattened
    int batch_idx = blockIdx.z; 

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // Calculate Base Pointers for this specific matrix in the stack
        // We use the "true" 2D strides (stride_h, stride_w) for inner matrix access
        // We use manual offset math for the batch jump
        
        // Offset = batch_idx * (Matrix_Size)
        // Matrix_Size for A = M * K
        long long offset_A = batch_idx * (M * A.stride_h); 
        long long offset_B = batch_idx * (K * B.stride_h);
        long long offset_C = batch_idx * (M * C.stride_h);

        // Pointers to the start of the current matrices
        T* A_ptr = A.rawp + A.offset + offset_A;
        T* B_ptr = B.rawp + B.offset + offset_B;
        T* C_ptr = C.rawp + C.offset + offset_C;

        T sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A_ptr[row * A.stride_h + k * A.stride_w] * B_ptr[k * B.stride_h + col * B.stride_w];
        }
        C_ptr[row * C.stride_h + col * C.stride_w] = sum;
    }
}

template <typename T>
void op_bmm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
{
    // 1. Validation using Internal Metadata
    // A: [b, d, h, w] -> [Batch, Heads, Seq, HeadDim]
    // B: [b, d, h, w] -> [Batch, Heads, HeadDim, Seq] (Transposed logic usually)
    
    if (A.b != B.b || A.d != B.d) {
        throw std::runtime_error("Batch/Head dimension mismatch in BMM");
    }
    
    // 2. Auto-Extract Dimensions
    int batch_count = A.b * A.d; // Collapse Batch and Heads for the Z-grid
    int M = A.true_h;            // Seq Len
    int K = A.true_w;            // Head Dim
    int N = B.true_w;            // Seq Len (Output width)

    if (A.true_w != B.true_h && A.true_w != B.true_w) { 
        // Note: Check logic depends on if B is pre-transposed or not. 
        // Assuming standard [M, K] @ [K, N] logic on the "true" dimensions.
    }

    // 3. Launch Configuration
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16, batch_count);

    // 4. Launch Kernel
    // Note: We pass the Tensors directly. The kernel must handle the pointer math.
    bmm_kernel<<<grid, block>>>(A, B, C, M, N, K);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("BMM Error: %s\n", cudaGetErrorString(err));
}