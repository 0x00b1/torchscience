#pragma once

#include <tuple>
#include <vector>

#include <torch/library.h>

namespace torchscience::meta::transform {

/**
 * Meta implementation of DST for shape inference.
 */
inline at::Tensor fourier_sine_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    [[maybe_unused]] int64_t type,
    [[maybe_unused]] int64_t norm
) {
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    int64_t n = (n_param > 0) ? n_param : input.size(dim);

    std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());
    output_shape[dim] = n;

    return at::empty(output_shape, input.options());
}

inline at::Tensor fourier_sine_transform_backward(
    [[maybe_unused]] const at::Tensor& grad_output,
    const at::Tensor& input,
    [[maybe_unused]] int64_t n_param,
    [[maybe_unused]] int64_t dim,
    [[maybe_unused]] int64_t type,
    [[maybe_unused]] int64_t norm
) {
    return at::empty_like(input);
}

inline std::tuple<at::Tensor, at::Tensor> fourier_sine_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    [[maybe_unused]] const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    [[maybe_unused]] int64_t type,
    [[maybe_unused]] int64_t norm
) {
    int64_t ndim = grad_grad_input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    int64_t n = (n_param > 0) ? n_param : grad_grad_input.size(dim);

    std::vector<int64_t> output_shape(grad_grad_input.sizes().begin(), grad_grad_input.sizes().end());
    output_shape[dim] = n;

    at::Tensor grad_grad_output = at::empty(output_shape, grad_grad_input.options());
    at::Tensor new_grad_input = at::empty_like(input);

    return std::make_tuple(grad_grad_output, new_grad_input);
}

inline at::Tensor inverse_fourier_sine_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    return fourier_sine_transform(input, n_param, dim, type, norm);
}

inline at::Tensor inverse_fourier_sine_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    return fourier_sine_transform_backward(grad_output, input, n_param, dim, type, norm);
}

inline std::tuple<at::Tensor, at::Tensor> inverse_fourier_sine_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    return fourier_sine_transform_backward_backward(grad_grad_input, grad_output, input, n_param, dim, type, norm);
}

}  // namespace torchscience::meta::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "fourier_sine_transform",
        &torchscience::meta::transform::fourier_sine_transform
    );

    module.impl(
        "fourier_sine_transform_backward",
        &torchscience::meta::transform::fourier_sine_transform_backward
    );

    module.impl(
        "fourier_sine_transform_backward_backward",
        &torchscience::meta::transform::fourier_sine_transform_backward_backward
    );

    module.impl(
        "inverse_fourier_sine_transform",
        &torchscience::meta::transform::inverse_fourier_sine_transform
    );

    module.impl(
        "inverse_fourier_sine_transform_backward",
        &torchscience::meta::transform::inverse_fourier_sine_transform_backward
    );

    module.impl(
        "inverse_fourier_sine_transform_backward_backward",
        &torchscience::meta::transform::inverse_fourier_sine_transform_backward_backward
    );
}
