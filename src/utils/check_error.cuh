#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>

#define CUDA_OK(err) (cuda_check((err), __FILE__, __LINE__))
#define CURAND_OK(err) (curand_check((err), __FILE__, __LINE__))

void cuda_check(cudaError_t err, const char *filename, int lineno)
{
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)) + 
            " (" + std::to_string((int)err) + ") at " + filename + ":" + std::to_string(lineno));
  }
}

void curand_check(curandStatus_t err, const char *filename, int lineno)
{
    if (err != CURAND_STATUS_SUCCESS)
    {
        throw std::runtime_error("CURAND Error at " + std::string(filename) + 
            ":" + std::to_string(lineno));
    }
}