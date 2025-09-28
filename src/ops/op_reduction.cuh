#pragma once

#include "utils/tensor.cuh"

template <typename T, typename IT>
class MaxAccumFunc
{
public:
    //This function compares input x with the current accumulated maximum value stored in accum
    //If x is bigger than accum, stores x in accum and stores x's index (ind_x) to ind_accum
    __host__ __device__ void operator()(const T &x, const IT &ind_x, T &accum, IT &ind_accum)
    {
      //Lab-1: add your code here
      if (x > accum) {
        // printf("update");
        // if (x > 10)
        //     printf("greater than 10 at ");
        ind_accum = ind_x;
        accum = x;
      }
    }
};

template <typename T, typename IT>
class SumAccumFunc
{
public:
    //This function adds input x to the current accumulated sum value stored in accum
    //The accumu's value is updated (to add x).  The ind_x and ind_accum arguments are not used.
    __host__ __device__ void operator()(const T &x, const IT &ind_x, T &accum, IT &ind_accum)
    {
        //Lab-1: add your code here
        accum += x;
    }
};

//This kernel function performs column-wise reduction of the "in" tensor and stores the result in "out" tensor.
//If "get_index" is true, then the index accumulator values are stored in "out_index" and "out" is not touched.
template <typename OpFunc, typename T, typename IT>
__global__ void op_reduction_kernel_colwise(OpFunc f, Tensor<T> in, Tensor<T> out, Tensor<IT> out_index, bool get_index)
{
    int cur_row = blockIdx.x * blockDim.x + threadIdx.x;
    // int cur_col = blockIdx.x * blockDim.x + threadIdx.x;
    // printf( "in op_elemwise_unary_kernel .  run kernel.." );

    if (cur_row >= in.h) {return;}

    IT accum_ind = 0;
    T accum = Index(in, cur_row, 0);
    for (int cur_col = 1; cur_col < in.w; cur_col++) {
        // Index(out, cur_row, cur_col) = f(Index(in, cur_row, cur_col));
        f(Index(in, cur_row, cur_col), cur_col, accum, accum_ind);

    }

    if (get_index) {
        Index(out_index, cur_row, 0) = accum_ind;
    }
    else {
        Index(out, cur_row, 0) = accum;
    }
    

    
}

//This kernel function performs row-wise reduction of the "in" tensor and stores the result in "out" tensor.
//If "get_index" is true, then the index accumulator values are stored in "out_index" and "out" is not touched.
template <typename OpFunc, typename T, typename IT>
__global__ void op_reduction_kernel_rowwise(OpFunc f, Tensor<T> in, Tensor<T> out, Tensor<IT> out_index, bool get_index)
{
    // int cur_row = blockIdx.y * blockDim.y + threadIdx.y;
    int cur_col = blockIdx.x * blockDim.x + threadIdx.x;
    // printf( "in op_elemwise_unary_kernel .  run kernel.." );
    if (cur_col >= in.w) {return;} 

    IT accum_ind = 0;
    T accum = Index(in, 0, cur_col);
    for (int cur_row = 1; cur_row < in.h; cur_row++) {
        // Index(out, cur_row, cur_col) = f(Index(in, cur_row, cur_col));
        f(Index(in, cur_row, cur_col), cur_row, accum, accum_ind);


    }

    if (get_index) {
        Index(out_index, 0, cur_col) = accum_ind;
    }
    else {
        Index(out, 0, cur_col) = accum;
    }


}

template <typename OpFunc, typename T, typename IT>
void op_reduction_gpu(OpFunc f, const Tensor<T> &in, Tensor<T> &out, Tensor<IT> &out_index, bool get_index = false)
{
    // printf("out.w %d", out.w);
    // printf("out.h %d", out.h);

  //Lab-1: add your code here. You need to launch either op_reduction_kernel_colwise or op_reduction_kernel_rowwise
  //depending on the output shape 
    int threads = ELEMWISE_BLOCK_DIM;

    // op_elemwise_unary_kernel<<<gridDim, blockDim>>>(f, t, out);
    // if ((out.w == 1) && (out.h == 1)) {
    //     Tensor<int> temp{in.h, 1, true};
    //     // const Tensor<int> rtemp = temp;

    //     int blocks = (in.h + threads - 1) / threads;
    //     op_reduction_kernel_colwise<<<blocks, threads>>>(f, in, temp, out_index, get_index);
    //     blocks = (in.w + threads - 1) / threads;
    //     op_reduction_kernel_rowwise<<<blocks, threads>>>(f, temp, out, out_index, get_index);
    if ((out.w <= 1) && (in.w > 1)) {
        int blocks = (in.h + threads - 1) / threads;
        // printf("doing column wise reduction");
        op_reduction_kernel_colwise<<<blocks, threads>>>(f, in, out, out_index, get_index);
    } else {
        int blocks = (in.w + threads - 1) / threads;
        // printf("doing row wise reduction");

        op_reduction_kernel_rowwise<<<blocks, threads>>>(f, in, out, out_index, get_index);

    }

    
    cudaDeviceSynchronize();

}

template <typename OpFunc, typename T, typename IT>
void op_reduction_cpu_rowwise(OpFunc f, const Tensor<T> &in, Tensor<T> &out, Tensor<IT> &out_index, bool get_index = false)
{    
    for (int j = 0; j < in.w; j++)
    {
        IT accum_ind = 0;
        T accum = Index(in, 0, j);
        for (int i = 1; i < in.h; i++)
        {
            f(Index(in, i, j), i, accum, accum_ind);
        }
        if (get_index)
            Index(out_index, 0, j) = accum_ind;
        else
            Index(out, 0, j) = accum;
    }
}

template <typename OpFunc, typename T, typename IT>
void op_reduction_cpu_colwise(OpFunc f, const Tensor<T> &in, Tensor<T> &out, Tensor<IT> &out_index, bool get_index = false)
{
    
    for (int i = 0; i < in.h; i++)
    {
        IT accum_ind = 0;
        T accum = Index(in, i, 0);
        for (int j = 1; j < in.w; j++)
        {
            f(Index(in, i, j), j, accum, accum_ind);
        }
        if (get_index)
            Index(out_index, i, 0) = accum_ind;
        else
            Index(out, i, 0) = accum;
    }
}

template <typename OpFunc, typename T, typename IT>
void op_reduction_cpu(OpFunc f, const Tensor<T> &in, Tensor<T> &out, Tensor<IT> &out_index, bool get_index = false)
{
    int out_h = get_index?out_index.h:out.h;
    if (in.h > out_h)
        op_reduction_cpu_rowwise(f, in, out, out_index, get_index);
    else
        op_reduction_cpu_colwise(f, in, out, out_index, get_index);
}

/*-----------------------------------------------------------*/
template <typename AT, typename OT>
static void ensure_reduction_shape_device(const Tensor<AT> &a, const Tensor<OT> &out)
{
    if (a.on_device != out.on_device)
    {
        throw std::runtime_error("ensure_reduction_shape_device2: device mismatch");
    }

    if (a.w == out.w && out.h == 1)
    {
    }
    else if (a.h == out.h && out.w == 1)
    {
    }
    else
    {
        throw std::runtime_error("ensure_reduction_shape_device2: output shape mismatch");
    }
}


template <typename T>
void op_sum(const Tensor<T> &in, Tensor<T> &out)
{
    Tensor<int> out_index;
    SumAccumFunc<T, int> f;
    ensure_reduction_shape_device(in, out);
    if (in.on_device)
    {
        op_reduction_gpu(f, in, out, out_index, false);
    }
    else
    {
        op_reduction_cpu(f, in, out, out_index, false);
    }
}

template <typename T, typename IT>
void op_argmax(const Tensor<T> &in, Tensor<IT> &out_index)
{
    Tensor<T> out;
    MaxAccumFunc<T, IT> f;
    ensure_reduction_shape_device(in, out_index);
    if (in.on_device)
    {
        op_reduction_gpu(f, in, out, out_index, true);
    }
    else 
    {
        op_reduction_cpu(f, in, out, out_index, true);
    }
}
