#pragma once
#include "utils/tensor.cuh"

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

    //Lab-1: please add your code here
    //You need to define separate GPU kernel function(s) and launch them here
    //In order to calculate d_logits, you should derive what its values should be 
    //symbolically.
    return 0;
    
}
