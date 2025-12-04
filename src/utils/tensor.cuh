#pragma once
#include <random>
#include <memory>
#include <sstream>
#include <string>
#include <iostream>
#include "utils/check_error.cuh"

#define ISCLOSE_RELTOL 1e-6
#define ISCLOSE_ABSTOL 1e-6

// LEGACY MACRO (Unchanged): Works because we adjust stride_h/h internally
#define Index(t, row, col) ((((t).rawp)[(t).offset + (row) * (t).stride_h + (col) * (t).stride_w]))

#define Index4D(t, b_idx, d_idx, row, col) \
    ((t).rawp[(t).offset + \
              (b_idx) * (t).stride_b + \
              (d_idx) * (t).stride_d + \
              (row)   * (t).stride_h + \
              (col)   * (t).stride_w])

template <typename T>
struct cudaDeleter {
    void operator()(T *p) const { if (p) cudaFree(p); }
};

template <typename T>
struct cpuDeleter {
    void operator()(T *p) const { if (p) free(p); }
};

template <typename T>
class Tensor {
public:
    // 2D View (Legacy)
    int32_t h; 
    int32_t w; 
    int32_t stride_h;
    int32_t stride_w;
    int32_t offset;

    // 4D Metadata
    int32_t b; 
    int32_t d; 
    int32_t true_h; 
    int32_t true_w;
    // 4D Strides
    int32_t stride_b; 
    int32_t stride_d;

    T *rawp;
    std::shared_ptr<T> ref;
    bool on_device;

    // Default Constructor
    Tensor() : h(0), w(0), stride_h(0), stride_w(0), offset(0), 
               b(1), d(1), true_h(0), true_w(0), 
               stride_b(0), stride_d(0), // Init new strides
               rawp(nullptr), on_device(false) 
    {
        ref = std::shared_ptr<T>(rawp, cpuDeleter<T>());
    }

    // Constructor 1: course hw sadffsf Legacy 2D ---
    Tensor(int32_t h_, int32_t w_, bool on_device_ = false)
        : h(h_), w(w_), stride_h(w_), stride_w(1), offset(0),
          b(1), d(1), true_h(h_), true_w(w_),
          on_device(on_device_)
    {
        // Even in 2D, calculate theoretical 4D strides for safety
        stride_d = h_ * w_; 
        stride_b = stride_d; // b=1, d=1
        allocate(h * w);
    }

    // --- Constructor 2: New 4D ---
    Tensor(int32_t b_, int32_t d_, int32_t h_, int32_t w_, bool on_device_ = false)
        : b(b_), d(d_), true_h(h_), true_w(w_),
          // Flatten 2D view:
          h(b_ * d_ * h_), w(w_), 
          stride_h(w_), stride_w(1), offset(0),
          on_device(on_device_)
    {
        // Calculate 4D Strides
        // To jump a head, skip (Rows * Cols)
        stride_d = h_ * w_; 
        // To jump a batch, skip (Heads * Rows * Cols)
        stride_b = d_ * stride_d; 
        
        allocate(b * d * h * w);
    }

private:
    void allocate(size_t total_elements) {
        if (on_device) {
            CUDA_OK(cudaMalloc(&rawp, sizeof(T) * total_elements));
            ref = std::shared_ptr<T>(rawp, cudaDeleter<T>());
        } else {
            rawp = (T *)malloc(sizeof(T) * total_elements);
            ref = std::shared_ptr<T>(rawp, cpuDeleter<T>());
        }
    }

public:
    // --- Helper for 4D Access (Used by BMM kernel) ---
    // Calculates the pointer offset for a specific batch/head
    __device__ __host__ T* ptr(int batch_idx, int head_idx) const {
        // We treat (Batch, Head) as the "stack" index
        // Stride for one full matrix (Seq * Dim) is (true_h * w)
        long long matrix_size = true_h * w;
        long long stack_idx = batch_idx * d + head_idx;
        return rawp + offset + (stack_idx * matrix_size);
    }

    // // --- Modified toHost to support 4D metadata preservation ---
    // Tensor<T> toHost() const {
    //     // Create a host tensor with the EXACT SAME structure (2D or 4D)
    //     Tensor<T> out;
    //     if (b > 1 || d > 1) {
    //          out = Tensor<T>(b, d, true_h, true_w, false);
    //     } else {
    //          out = Tensor<T>(h, w, false);
    //     }
        
    //     if (on_device) {
    //         out.toDevice(*this); // reuse existing copy logic (reversed)
    //         // Actually, let's just do the copy manually here to be safe
    //         CUDA_OK(cudaMemcpy(out.rawp, rawp, h * w * sizeof(T), cudaMemcpyDeviceToHost));
    //     }
    //     return out;
    // }
    void toHost(Tensor<T> &out) const{
    if (out.on_device) {
      throw std::runtime_error("Output tensor must be on host instead of: " + out.repr());
    }
    if (h!= out.h || w != out.w) {
      throw std::runtime_error("Output tensor shape mismatch: " + out.repr() + " vs " + this->repr());
    }

    if (!on_device) {
      out = *this;
      return;
    }
    out.offset = offset;
    out.stride_h = stride_h;
    out.stride_w = stride_w;
    CUDA_OK(cudaMemcpy(out.rawp, rawp, h * w * sizeof(T), cudaMemcpyDeviceToHost));
    }

    Tensor<T> toHost() const
    {
      Tensor<T> t{h, w};
      toHost(t);
      return t;
    }

    void toDevice(Tensor<T> &out) const
    {
      if (!out.on_device) {
        throw std::runtime_error("Output tensor must be on device instead of: " + out.repr());
      }
      if (h != out.h || w != out.w) {
        throw std::runtime_error("Output tensor shape mismatch: " + out.repr() + " vs " + this->repr());
      }

      if (on_device) {
        out = *this;
        return;
      } 

      out.offset = offset;
      out.stride_h = stride_h;
      out.stride_w = stride_w;
      CUDA_OK(cudaMemcpy(out.rawp, rawp, h * w * sizeof(T), cudaMemcpyHostToDevice));
    }

    Tensor<T> toDevice() const
    {
      Tensor<T> t{h, w, true};
      toDevice(t);
      return t;
    }

    Tensor<T> transpose() const
    {
      Tensor<T> t{};
      t.w = h;
      t.stride_w = stride_h;
      t.h = w;
      t.stride_h = stride_w;
      t.offset = offset;
      t.ref = ref;
      t.rawp = rawp;
      t.on_device = on_device;
      return t;
    }



    // Keep Slice (Legacy 2D slicing)
    // WARNING: Slicing a flattened 4D tensor is dangerous if crossing batch boundaries.
    Tensor<T> slice(int start_h, int end_h, int start_w, int end_w) const
    {
      // 1. Basic Validation (Same as before)
      if(start_h >= end_h || end_h > h || start_w >= end_w || end_w > w) {
        throw std::runtime_error("Slice index out of range");
      }

      // 2. Make a shallow copy
      Tensor<T> t = *this; 

      // 3. Update 2D View (Same as before)
      t.w = end_w - start_w;
      t.h = end_h - start_h;
      t.offset = offset + start_h * stride_h + start_w * stride_w;

      // ---------------------------------------------------------
      // 4. CRITICAL FIX: Invalidate 4D Metadata
      // ---------------------------------------------------------
      // Since we took an arbitrary slice, we can no longer guarantee 
      // that this represents a clean stack of Batches/Heads.
      // We "downgrade" it to a standard 2D tensor.
      
      t.b = 1;
      t.d = 1;
      t.true_h = t.h; // The new height is the only height
      t.true_w = t.w; // The new width is the only width
      
      t.stride_b = 0; // Disable 4D strides to be safe
      t.stride_d = 0;

      return t;
    }

    std::string repr() const {
        std::ostringstream oss;
        if (b > 1 || d > 1) {
            oss << "Tensor4D(" << b << "x" << d << "x" << true_h << "x" << true_w;
        } else {
            oss << "Tensor2D(" << h << "x" << w;
        }
        oss << ", device=" << (on_device ? "GPU" : "CPU") << ")";
        return oss.str();
    }
  std::string str() const
  {
    Tensor<T> t{};
    if (on_device)
    {
      t = toHost();
    }
    else
    {
      t = *this;
    }
    std::stringstream ss;
    for (int i = 0; i < h; i++)
    {
      for (int j = 0; j < w; j++)
      {
        if (std::is_same_v<T, char> || std::is_same_v<T, unsigned char>)
        {
          ss << (int)Index(t, i, j) << " ";
        }
        else
        {
         // std::cout << "haha " << Index(t, i, j) << std::endl;
          ss << Index(t, i, j) << " ";
        }
        ss << "";
      }
      ss << "\n";
    }
    return ss.str();
  }

// Check if all elements of this tensor is the "same" as the other tensor 
bool allclose(const Tensor<T> &other)
{
    if (h != other.h || w != other.w)
    {
        return false;
    }
    Tensor<T> me = this->toHost();
    Tensor<T> ot = other.toHost();
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            // Check if the numbers are close using relative and absolute tolerances
            T a = Index(me, i, j);
            T b = Index(ot, i, j);
            if (std::abs(a - b) >
                std::max(ISCLOSE_RELTOL * std::max(std::abs(a), std::abs(b)), ISCLOSE_ABSTOL))
            {
                std::cout << "(" << i << "," << j << ") this=" << a << " other=" << b << " diff=" << (a - b) << std::endl;
                return false;
            }
        }
    }
    return true;
}



};



