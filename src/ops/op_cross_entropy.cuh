#pragma once
#include "utils/tensor.cuh"
#include <cuda_runtime.h>
#include <cfloat>

//This function calculates the cross_entropy loss from the "logits" tensor for a batch of training innput
//and the batch's corresponding "target" label tensor and returns the average loss of the batch.
//It also returns the gradient of the logits tensor.
template <typename T, typename S>
T op_cross_entropy_loss(const Tensor<T> &logits, const Tensor<S> &targets,
                               Tensor<T> &d_logits)
{
    if (logits.h != d_logits.h || logits.w != d_logits.w)
    {
        throw std::runtime_error("op_cross_entropy_loss: d_logits shape mismatch");
    }

    if (targets.h != logits.h || targets.w != 1)
    {
        throw std::runtime_error("op_cross_entropy_loss: targets shape mismatch");
    }
    if (logits.on_device != d_logits.on_device || logits.on_device != targets.on_device)
    {
        throw std::runtime_error("op_cross_entropy_loss: device mismatch");
    }
    int N = logits.h;
    int dim_logit = logits.w;
    int threads;
    if (dim_logit < 1024) {
        threads = dim_logit;
    } else {
        threads = 1024;}
        
    int blocks = (N + threads - 1) / threads;

    T zero = 0;
    T* d_loss_sum;
    cudaMalloc(&d_loss_sum, sizeof(T));
    cudaMemcpy(d_loss_sum, &zero, sizeof(T), cudaMemcpyHostToDevice);

    cross_entropy_kernel<<<blocks, threads>>>(
        logits,
        targets,
        d_logits,
        d_loss_sum
    );
    cudaDeviceSynchronize();

    // Copy back summed loss
    T total_loss = 0;
    cudaMemcpy(&total_loss, d_loss_sum, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_loss_sum);

    // Return average loss
    return total_loss / static_cast<T>(N);
    }
 
   


template <typename T, typename S>
__global__ void cross_entropy_kernel(
    const Tensor<T> logits,
    const Tensor<S> targets,
    Tensor<T> d_logits,
    T* loss_sum     // device scalar for accumulating total loss
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= logits.h) return;

    int C = logits.w;   // num classes
    int label = Index(targets, i, 0);

    T sum_exp = 0;
    for(int j = 0; j < C; j++) {
        sum_exp += exp(Index(logits, i, j));

    }

    // Compute probabilities and gradient
    T loss_i = 0;
    for(int j = 0; j < C; j++){
        T p = exp(Index(logits, i, j)) / sum_exp;

        T y = (j == label) ? 1.0f : 0.0f;
        Index(d_logits, i, j) = (p - y) / logits.h; // divide by batch size

        if(j == label){
            loss_i = -log(p);
        }
    }
    atomicAdd(loss_sum, loss_i);
}

// Computes safe-Softmax row-wise respecting Strides and Offsets
// recall the quiz-1 last question for reference
template <typename T>
__global__ void softmax_kernel(Tensor<T> input, Tensor<T> output) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < input.h) {
        // 1. Find Max (for numerical stability)
        T max_val = -999999;
        for (int c = 0; c < input.w; ++c) {
            T val = Index(input, row, c); 
            if (val > max_val) {
                max_val = val;
            }
        }

        T sum = 0.0f;
        for (int c = 0; c < input.w; ++c) {
            T val = expf(Index(input, row, c) - max_val);
            Index(output, row, c) = val; 
            sum += val;
        }

        for (int c = 0; c < input.w; ++c) {
            Index(output, row, c) /= sum;
        }
    }
}

// Wrapper of softmax kernel
template <typename T>
void op_softmax(const Tensor<T>& input, Tensor<T>& output) {

    if (input.h != output.h || input.w != output.w) {
        printf("Shape mismatch in softmax\n"); 
        return;
    }

    // Configure: 1 thread per row. 
    // Note: For very wide matrices (large input.w), this is slow. 
    int threads = 256;
    int blocks = (input.h + threads - 1) / threads;

    softmax_kernel<<<blocks, threads>>>(input, output);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Softmax Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
}