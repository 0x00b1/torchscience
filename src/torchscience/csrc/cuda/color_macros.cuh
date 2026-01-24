#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#define TORCHSCIENCE_CUDA_COLOR_CONVERSION(name)                              \
namespace torchscience::cuda::graphics::color {                               \
                                                                              \
template <typename scalar_t>                                                  \
__global__ void name##_kernel(                                                \
    const scalar_t* __restrict__ input,                                       \
    scalar_t* __restrict__ output,                                            \
    int64_t num_pixels                                                        \
) {                                                                           \
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;                      \
    if (idx < num_pixels) {                                                   \
        kernel::graphics::color::name##_scalar(                               \
            input + idx * 3,                                                  \
            output + idx * 3                                                  \
        );                                                                    \
    }                                                                         \
}                                                                             \
                                                                              \
template <typename scalar_t>                                                  \
__global__ void name##_backward_kernel(                                       \
    const scalar_t* __restrict__ grad_output,                                 \
    const scalar_t* __restrict__ input,                                       \
    scalar_t* __restrict__ grad_input,                                        \
    int64_t num_pixels                                                        \
) {                                                                           \
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;                      \
    if (idx < num_pixels) {                                                   \
        kernel::graphics::color::name##_backward_scalar(                      \
            grad_output + idx * 3,                                            \
            input + idx * 3,                                                  \
            grad_input + idx * 3                                              \
        );                                                                    \
    }                                                                         \
}                                                                             \
                                                                              \
inline at::Tensor name(const at::Tensor& input) {                             \
    TORCH_CHECK(input.size(-1) == 3, #name ": last dim must be 3");           \
    c10::cuda::CUDAGuard device_guard(input.device());                        \
    auto input_contig = input.contiguous();                                   \
    auto output = at::empty_like(input_contig);                               \
    const int64_t num_pixels = input.numel() / 3;                             \
    const int threads = 256;                                                  \
    const int blocks = (num_pixels + threads - 1) / threads;                  \
                                                                              \
    AT_DISPATCH_FLOATING_TYPES_AND2(                                          \
        at::kBFloat16, at::kHalf, input.scalar_type(), #name "_cuda", [&] {   \
            name##_kernel<scalar_t><<<blocks, threads>>>(                     \
                input_contig.data_ptr<scalar_t>(),                            \
                output.data_ptr<scalar_t>(),                                  \
                num_pixels);                                                  \
        });                                                                   \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                           \
    return output;                                                            \
}                                                                             \
                                                                              \
inline at::Tensor name##_backward(                                            \
    const at::Tensor& grad_output,                                            \
    const at::Tensor& input                                                   \
) {                                                                           \
    TORCH_CHECK(grad_output.size(-1) == 3);                                   \
    TORCH_CHECK(input.size(-1) == 3);                                         \
    c10::cuda::CUDAGuard device_guard(input.device());                        \
    auto grad_output_contig = grad_output.contiguous();                       \
    auto input_contig = input.contiguous();                                   \
    auto grad_input = at::empty_like(input_contig);                           \
    const int64_t num_pixels = input.numel() / 3;                             \
    const int threads = 256;                                                  \
    const int blocks = (num_pixels + threads - 1) / threads;                  \
                                                                              \
    AT_DISPATCH_FLOATING_TYPES_AND2(                                          \
        at::kBFloat16, at::kHalf, input.scalar_type(),                        \
        #name "_backward_cuda", [&] {                                         \
            name##_backward_kernel<scalar_t><<<blocks, threads>>>(            \
                grad_output_contig.data_ptr<scalar_t>(),                      \
                input_contig.data_ptr<scalar_t>(),                            \
                grad_input.data_ptr<scalar_t>(),                              \
                num_pixels);                                                  \
        });                                                                   \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                           \
    return grad_input;                                                        \
}                                                                             \
                                                                              \
} /* namespace torchscience::cuda::graphics::color */                         \
                                                                              \
TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {                                   \
    m.impl(#name, torchscience::cuda::graphics::color::name);                 \
    m.impl(#name "_backward",                                                 \
           torchscience::cuda::graphics::color::name##_backward);             \
}
