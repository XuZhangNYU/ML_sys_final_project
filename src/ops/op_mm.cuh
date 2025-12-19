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
 
    const int BLOCK_DIM = 16;

    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((C.w + BLOCK_DIM - 1) / BLOCK_DIM,
                 (C.h + BLOCK_DIM - 1) / BLOCK_DIM);

    // Launch kernel
    mm_kernel<<<gridDim, blockDim>>>(A, B, C);
    cudaDeviceSynchronize();
}

// Helper to get pointer offset for specific batch
template <typename T>
__device__ T* get_batch_ptr(Tensor<T>& t, int batch_idx, int rows_per_batch) {
    return t.rawp + t.offset + (batch_idx * rows_per_batch * t.stride_h);
}


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
    // A: [b, d, h, w] -> [Batch, Heads, Seq, HeadDim]
    // B: [b, d, h, w] -> [Batch, Heads, HeadDim, Seq] 
    
    if (A.b != B.b || A.d != B.d) {
        throw std::runtime_error("Batch/Head dimension mismatch in BMM");
    }
    
    // Auto-Extract Dimensions
    int batch_count = A.b * A.d; // Collapse Batch and Heads for the Z-grid
    int M = A.true_h;            // Seq Len
    int K = A.true_w;            // Head Dim
    int N = B.true_w;            // Seq Len (Output width)

    if (A.true_w != B.true_h && A.true_w != B.true_w) { 
        // Assuming standard [M, K] @ [K, N] logic on the "true" dimensions.
    }

    // Launch Configuration
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16, batch_count);

    bmm_kernel<<<grid, block>>>(A, B, C, M, N, K);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("BMM Error: %s\n", cudaGetErrorString(err));
}


// Generic Kernel: Swaps Axis 1 and Axis 2 (0-indexed: 0, 2, 1, 3)
// Works for 2 cases: 1.  [B, S, H, D] -> [B, H, S, D]
// AND 2. [B, H, S, D] -> [B, S, H, D]
template <typename T>
__global__ void permute_0213_kernel(Tensor<T> in, Tensor<T> out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = out.b * out.d * out.true_h * out.true_w;

    if (idx < total_elements) {
        // 1. Decode OUTPUT Linear Index -> (b, d, h, w)
    
        int temp = idx;
        int w_idx = temp % out.true_w; temp /= out.true_w; // Dim 3
        int h_idx = temp % out.true_h; temp /= out.true_h; // Dim 2
        int d_idx = temp % out.d;      temp /= out.d;      // Dim 1
        int b_idx = temp;                                  // Dim 0

        // 2. Map to INPUT (Swap d and h indices)
        
        
        long long src_offset = 
            b_idx * in.stride_b + 
            h_idx * in.stride_d +  // <--- Use h_idx for D-stride
            d_idx * in.stride_h +  // <--- Use d_idx for H-stride
            w_idx * in.stride_w;

        out.rawp[idx] = in.rawp[in.offset + src_offset];
    }
}

template <typename T>
void op_permute_0213(const Tensor<T>& in, Tensor<T>& out) {
    int total = in.b * in.d * in.true_h * in.true_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    permute_0213_kernel<<<blocks, threads>>>(in, out);
}


// Permute 0, 2, 3, 1 Kernel (For K)
// In:  [B, Seq, Heads, Dim] and Out: [B, Heads, Dim, Seq]
template <typename T>
__global__ void permute_0231_kernel(Tensor<T> in, Tensor<T> out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = out.b * out.d * out.true_h * out.true_w; // Use output total

    if (idx < total_elements) {
        // 1. Decode OUTPUT indices [b, h, d, s] from linear 'idx'
        
        int s_dim = out.true_w; // Last dim (Seq)
        int d_dim = out.true_h; // 2nd last (Dim)
        int h_dim = out.d;      // 3rd last (Heads)
        
        int temp = idx;
        int s_idx = temp % s_dim; temp /= s_dim; // Seq
        int m_idx = temp % d_dim; temp /= d_dim; // Dim
        int h_idx = temp % h_dim; temp /= h_dim; // Heads
        int b_idx = temp;                        // Batch

        // 2. Map to INPUT indices [b, s, h, d]
        // Input Shape: [B, Seq, Heads, Dim]
        
        int in_dim_size = in.true_w;  // Dim
        int in_head_size = in.true_h; // Heads

        int stride_dim = 1;
        int stride_head = in_dim_size;
        int stride_seq  = in_head_size * in_dim_size; // FIX: Heads * Dim
        int stride_batch = in.d * stride_seq;         // Seq * (Heads * Dim)

        long long src_offset = 
            (long long)b_idx * stride_batch +
            (long long)s_idx * stride_seq +
            (long long)h_idx * stride_head +
            (long long)m_idx * stride_dim;

        out.rawp[idx] = in.rawp[in.offset + src_offset];
    }
}

template <typename T>
void op_permute_0231(const Tensor<T>& in, Tensor<T>& out) {
    int total = out.b * out.d * out.true_h * out.true_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    permute_0231_kernel<<<blocks, threads>>>(in, out);
}

template <typename T>
__global__ void causal_mask_kernel(Tensor<T> w, T scale) {
    // w shape: [Batch, Heads, Seq, Seq]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Flattened dimensions
    int total_elements = w.b * w.d * w.true_h * w.true_w;
    if (idx >= total_elements) return;

    // Decode indices to find row/col in the Sequence matrix
    // Stride of last dim (Seq) is 1
    int col = idx % w.true_w; 
    int row = (idx / w.true_w) % w.true_h; 

    // 1. Scale
    T val = w.rawp[idx] * scale;

    // 2. Causal Mask Logic
    // We want only lower triangular (row >= col).
    // If row < col, it's future token -> Mask it (-1e10)
    if (col > row) {
        val = -1e10f;
    }

    w.rawp[idx] = val;
}

template <typename T>
void op_causal_mask(Tensor<T>& w, float scale) {
    int total = w.b * w.d * w.true_h * w.true_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    causal_mask_kernel<<<blocks, threads>>>(w, (T)scale);
}

// Kernel to Split [Rows, 3*Dim] -> 3 * [Rows, Dim]
template <typename T>
__global__ void split_qkv_kernel(Tensor<T> in, Tensor<T> q, Tensor<T> k, Tensor<T> v) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Col in Q/K/V (0..Dim-1)
    
    if (row < q.h && col < q.w) {
        int dim = q.w; // 768
        
        // Input layout: [Row, 3*Dim]
        // Q is 0..Dim, K is Dim..2Dim, V is 2Dim..3Dim
        
        Index(q, row, col) = Index(in, row, col);
        Index(k, row, col) = Index(in, row, col + dim);
        Index(v, row, col) = Index(in, row, col + 2*dim);
    }


}

template <typename T>
void op_split_qkv(Tensor<T> in, Tensor<T> q, Tensor<T> k, Tensor<T> v) {
    // int total = in.b * in.d * in.true_h * in.true_w;
    dim3 block(16, 16);
    dim3 grid((q.w + 15) / 16, (q.h + 15) / 16);
    split_qkv_kernel<<<grid, block>>>(in, q, k, v);
}


// Gpu kernel: LayerNorm per row
template <typename T>
__global__ void op_layernorm_kernel(Tensor<T> x,
                                    Tensor<T> gamma,
                                    Tensor<T> beta,
                                    Tensor<T> out,
                                    float eps)
{
    int row = blockIdx.x;       
    int w   = x.w;       

    // 1. compute mean for this row
    T mean = 0;
    for (int j = 0; j < w; j++) {
        mean += Index(x, row, j);
    }
    mean /= (float) T(w);

    // 2. variance
    T var = 0;
    for (int j = 0; j < w; j++) {
        T diff = Index(x, row, j) - mean;
        var += diff * diff;
    }
    var /= T(w);
// 2. Use rsqrtf (explicit float version)
    float var_f = static_cast<float>(var);
    float inv_std_f = rsqrtf(var_f + eps);
    
    // 3. Cast back to T (If T is float, this does nothing. If T is int, it truncates)
    T inv_std = static_cast<T>(inv_std_f);

    // 3. normalize, scale, shift
    for (int j = 0; j < w; j++) {
        T norm   = (Index(x, row, j) - mean) * inv_std;
        T scaled = norm * Index(gamma, 0, j) + Index(beta, 0, j);
        Index(out, row, j) = scaled;
    }
}

template <typename T>
void op_layernorm(const Tensor<T> &x,
                  const Tensor<T> &gamma,
                  const Tensor<T> &beta,
                  Tensor<T> &out,
                  float eps = 1e-5f)
{
    if (gamma.h != 1 || beta.h != 1 || gamma.w != x.w || beta.w != x.w) {
        throw std::runtime_error("LayerNorm: gamma/beta shape mismatch");
    }
    if (!x.on_device) {
        throw std::runtime_error("LayerNorm requires CUDA tensor");
    }

    int blocks = x.h;
    int threads = 1;   

    op_layernorm_kernel<<<blocks, threads>>>(x, gamma, beta, out, eps);
    cudaDeviceSynchronize();

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[layernorm] CUDA ERROR: %s\n", cudaGetErrorString(err));
    }
}
