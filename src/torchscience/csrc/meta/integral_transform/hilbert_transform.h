#pragma once

#include <tuple>
#include <vector>

#include <torch/library.h>

namespace torchscience::meta::integral_transform {

/**
 * Meta implementation of Hilbert transform for shape inference.
 *
 * The Hilbert transform preserves the shape of the input tensor,
 * unless n is specified, in which case the output size along dim is n.
 */
inline at::Tensor hilbert_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    [[maybe_unused]] int64_t padding_mode,
    [[maybe_unused]] double padding_value,
    [[maybe_unused]] const c10::optional<at::Tensor>& window
) {
    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    // Determine output size along dim
    int64_t n = (n_param > 0) ? n_param : input.size(dim);

    // Create output shape
    std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());
    output_shape[dim] = n;

    return at::empty(output_shape, input.options());
}

/**
 * Meta implementation of backward pass.
 */
inline at::Tensor hilbert_transform_backward(
    [[maybe_unused]] const at::Tensor& grad_output,
    const at::Tensor& input,
    [[maybe_unused]] int64_t n_param,
    [[maybe_unused]] int64_t dim,
    [[maybe_unused]] int64_t padding_mode,
    [[maybe_unused]] double padding_value,
    [[maybe_unused]] const c10::optional<at::Tensor>& window
) {
    // Output matches input shape (backward produces gradient for input)
    return at::empty_like(input);
}

/**
 * Meta implementation of double backward pass.
 */
inline std::tuple<at::Tensor, at::Tensor> hilbert_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    [[maybe_unused]] const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    [[maybe_unused]] int64_t padding_mode,
    [[maybe_unused]] double padding_value,
    [[maybe_unused]] const c10::optional<at::Tensor>& window
) {
    // Normalize dimension
    int64_t ndim = grad_grad_input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    // Determine output size along dim
    int64_t n = (n_param > 0) ? n_param : grad_grad_input.size(dim);

    // Create output shape for grad_grad_output
    std::vector<int64_t> output_shape(grad_grad_input.sizes().begin(), grad_grad_input.sizes().end());
    output_shape[dim] = n;

    at::Tensor grad_grad_output = at::empty(output_shape, grad_grad_input.options());
    at::Tensor new_grad_input = at::empty_like(input);

    return std::make_tuple(grad_grad_output, new_grad_input);
}

}  // namespace torchscience::meta::integral_transform

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "hilbert_transform",
        &torchscience::meta::integral_transform::hilbert_transform
    );

    module.impl(
        "hilbert_transform_backward",
        &torchscience::meta::integral_transform::hilbert_transform_backward
    );

    module.impl(
        "hilbert_transform_backward_backward",
        &torchscience::meta::integral_transform::hilbert_transform_backward_backward
    );
}
