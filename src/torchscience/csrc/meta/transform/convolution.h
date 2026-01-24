#pragma once

#include <tuple>
#include <vector>

#include <torch/library.h>

namespace torchscience::meta::transform {

/**
 * Meta implementation of convolution for shape inference.
 */
inline at::Tensor convolution(
    const at::Tensor& input,
    const at::Tensor& kernel,
    int64_t dim,
    int64_t mode
) {
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    int64_t n_input = input.size(dim);
    int64_t n_kernel = kernel.dim() > 0 ? kernel.size(kernel.dim() == input.dim() ? dim : 0) : 1;

    int64_t n_output;
    if (mode == 0) {  // full
        n_output = n_input + n_kernel - 1;
    } else if (mode == 1) {  // same
        n_output = n_input;
    } else {  // valid
        n_output = n_input - n_kernel + 1;
    }

    std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());
    output_shape[dim] = n_output;

    return at::empty(output_shape, input.options());
}

inline std::tuple<at::Tensor, at::Tensor> convolution_backward(
    [[maybe_unused]] const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& kernel,
    [[maybe_unused]] int64_t dim,
    [[maybe_unused]] int64_t mode
) {
    return std::make_tuple(at::empty_like(input), at::empty_like(kernel));
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_backward(
    [[maybe_unused]] const at::Tensor& grad_grad_input,
    [[maybe_unused]] const at::Tensor& grad_grad_kernel,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& kernel,
    [[maybe_unused]] int64_t dim,
    [[maybe_unused]] int64_t mode
) {
    return std::make_tuple(
        at::empty_like(grad_output),
        at::empty_like(input),
        at::empty_like(kernel)
    );
}

}  // namespace torchscience::meta::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "convolution",
        &torchscience::meta::transform::convolution
    );

    module.impl(
        "convolution_backward",
        &torchscience::meta::transform::convolution_backward
    );

    module.impl(
        "convolution_backward_backward",
        &torchscience::meta::transform::convolution_backward_backward
    );
}
