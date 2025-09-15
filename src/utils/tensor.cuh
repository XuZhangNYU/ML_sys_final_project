#pragma once
#include <random>
#include <memory>
#include <sstream>
#include <string>

#include "utils/check_error.cuh"

#define ISCLOSE_RELTOL 1e-6 // this would not work for precision lower than float
#define ISCLOSE_ABSTOL 1e-6

#define Index(t, row, col) ((((t).rawp)[(t).offset + (row) * (t).stride_h + (col) * (t).stride_w]))
#define IndexOutofBound(t, row, col) ((((row) >= (t).h) || ((col) >= (t).w)))

template <typename T>
struct cudaDeleter
{
  void operator()(T *p) const
  {
    if (p != nullptr)
    {
      // std::cout << "Free p=" << p << std::endl;
      cudaFree(p);
    }
  }
};

template <typename T>
struct cpuDeleter
{
  void operator()(T *p) const
  {
    if (p != nullptr)
    {
      // std::cout << "Free p=" << p << std::endl;
      free(p);
    }
  }
};

template <typename T>
class Tensor
{
public:
  int32_t h; // height
  int32_t w; // width
  int32_t stride_h;
  int32_t stride_w;
  int32_t offset;
  T *rawp;
  std::shared_ptr<T> ref; // refcounted pointer, for garbage collection use only
  bool on_device;

  Tensor() : h(0), w(0), stride_h(0), stride_w(0), offset(0), rawp(nullptr), on_device(false)
  {
    ref = std::shared_ptr<T>(rawp, cpuDeleter<T>());
  }

  Tensor(int32_t h_, int32_t w_, bool on_device_ = false)
      : h(h_), w(w_), stride_h(w_), stride_w(1), offset(0), on_device(on_device_)
  {
    if (on_device_)
    {
      CUDA_OK(cudaMalloc(&rawp, sizeof(T) * h * w));
      // std::cout << "cudaMalloc p=" << rawp << std::endl;
      ref = std::shared_ptr<T>(rawp, cudaDeleter<T>());
    }
    else
    {
      rawp = (T *)malloc(sizeof(T) * h * w);
      if (rawp == nullptr) {
        throw std::runtime_error("Failed to malloc cpu memory");
      }
      //std::cout << "malloc p=" << rawp << " h=" << h << " w=" << w << std::endl;
      ref = std::shared_ptr<T>(rawp, cpuDeleter<T>());
    }
  }


  void toHost(Tensor<T> &out) const
  {
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

  // slice the tensor, the returned tensor shares the same memory as this tensor
  // similar to numpy's tensor[start_h:end_h, start_w:end_w]
  Tensor<T> slice(int start_h, int end_h, int start_w, int end_w) const
  {
    Tensor<T> t{};
    if(start_h >= end_h || end_h > h || start_w >= end_w || end_w > w) {
      throw std::runtime_error("Slice index out of range: " + std::to_string(start_h) + 
        ":" + std::to_string(end_h) + "," + std::to_string(start_w) + 
        ":" + std::to_string(end_w) + " for tensor " + repr());
    }
    //Since the sliced tensor shares the same memory as this tensor, 
    //we just copy the ref, rawp, and on_device flag
    t.ref = ref;
    t.rawp = rawp;
    t.on_device = on_device;
    
    //set the other fields accordingly 
    t.w = end_w - start_w;
    t.h = end_h - start_h;
    t.stride_h = stride_h;
    t.stride_w = stride_w;
    t.offset = offset + start_h * stride_h + start_w * stride_w;

    return t;
  }

  std::string repr() const
  {
    std::ostringstream oss;
    oss << "Tensor(" << h << ", " << w << ", on_device=" << (on_device ? "true" : "false") << ")";
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



