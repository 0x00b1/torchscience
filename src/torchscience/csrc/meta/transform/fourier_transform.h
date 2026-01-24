#pragma once

#include <tuple>
#include <vector>

#include <torch/library.h>

namespace torchscience::meta::transform {

/**
 * Meta implementation of Fourier transform for shape inference.
 *
 * The Fourier transform preserves the shape of the input tensor,
 * unless n is specified, in which case the output size along dim is n.
 * Output is always complex.
 */
inline at::Tensor fourier_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    [[maybe_unused]] int64_t padding_mode,
    [[maybe_unused]] double padding_value,
    [[maybe_unused]] const c10::optional<at::Tensor>& window,
    [[maybe_unused]] int64_t norm
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

    // FFT output is always complex
    auto output_dtype = input.is_complex() ? input.scalar_type() :
        (input.scalar_type() == at::kFloat ? at::kComplexFloat : at::kComplexDouble);

    return at::empty(output_shape, input.options().dtype(output_dtype));
}

/**
 * Meta implementation of backward pass.
 */
inline at::Tensor fourier_transform_backward(
    [[maybe_unused]] const at::Tensor& grad_output,
    const at::Tensor& input,
    [[maybe_unused]] int64_t n_param,
    [[maybe_unused]] int64_t dim,
    [[maybe_unused]] int64_t padding_mode,
    [[maybe_unused]] double padding_value,
    [[maybe_unused]] const c10::optional<at::Tensor>& window,
    [[maybe_unused]] int64_t norm
) {
    // Output matches input shape (backward produces gradient for input)
    return at::empty_like(input);
}

/**
 * Meta implementation of double backward pass.
 */
inline std::tuple<at::Tensor, at::Tensor> fourier_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    [[maybe_unused]] const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    [[maybe_unused]] int64_t padding_mode,
    [[maybe_unused]] double padding_value,
    [[maybe_unused]] const c10::optional<at::Tensor>& window,
    [[maybe_unused]] int64_t norm
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

    // FFT output is complex
    auto output_dtype = grad_grad_input.is_complex() ? grad_grad_input.scalar_type() :
        (grad_grad_input.scalar_type() == at::kFloat ? at::kComplexFloat : at::kComplexDouble);

    at::Tensor grad_grad_output = at::empty(output_shape, grad_grad_input.options().dtype(output_dtype));
    at::Tensor new_grad_input = at::empty_like(input);

    return std::make_tuple(grad_grad_output, new_grad_input);
}

}  // namespace torchscience::meta::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "fourier_transform",
        &torchscience::meta::transform::fourier_transform
    );

    module.impl(
        "fourier_transform_backward",
        &torchscience::meta::transform::fourier_transform_backward
    );

    module.impl(
        "fourier_transform_backward_backward",
        &torchscience::meta::transform::fourier_transform_backward_backward
    );
}
