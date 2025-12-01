#pragma once

#include <torch/library.h>

namespace torchscience::meta::filter {

/**
 * Meta kernel for butterworth_analog_bandpass_filter.
 * Computes output shape without actual computation.
 *
 * Input: n (int), omega_p1 (Tensor), omega_p2 (Tensor)
 * Output shape: (*broadcast_shape(omega_p1, omega_p2), n, 6)
 */
inline at::Tensor butterworth_analog_bandpass_filter(
    int64_t n,
    const at::Tensor& omega_p1,
    const at::Tensor& omega_p2
) {
    TORCH_CHECK(n > 0, "butterworth_analog_bandpass_filter: order n must be positive");
    TORCH_CHECK(n <= 64, "butterworth_analog_bandpass_filter: order n must be <= 64");

    // Compute broadcasted shape
    auto broadcasted = at::broadcast_tensors({omega_p1, omega_p2});
    auto batch_shape = broadcasted[0].sizes().vec();

    // Output shape: (*batch_shape, n, 6)
    std::vector<int64_t> output_shape = batch_shape;
    output_shape.push_back(n);
    output_shape.push_back(6);

    // Create output tensor with correct shape but no data
    auto options = omega_p1.options();
    return at::empty(output_shape, options);
}

/**
 * Meta kernel for backward pass.
 */
inline std::tuple<at::Tensor, at::Tensor> butterworth_analog_bandpass_filter_backward(
    const at::Tensor& grad_output,
    int64_t n,
    const at::Tensor& omega_p1,
    const at::Tensor& omega_p2
) {
    // Output gradients have same shape as inputs
    auto broadcasted = at::broadcast_tensors({omega_p1, omega_p2});
    auto batch_shape = broadcasted[0].sizes().vec();

    auto options = omega_p1.options();
    at::Tensor grad_omega_p1 = at::empty(batch_shape, options);
    at::Tensor grad_omega_p2 = at::empty(batch_shape, options);

    return std::make_tuple(grad_omega_p1, grad_omega_p2);
}

/**
 * Meta kernel for double-backward pass.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> butterworth_analog_bandpass_filter_backward_backward(
    const at::Tensor& grad_grad_omega_p1,
    const at::Tensor& grad_grad_omega_p2,
    const at::Tensor& grad_output,
    int64_t n,
    const at::Tensor& omega_p1,
    const at::Tensor& omega_p2
) {
    // Output has same shape as grad_output and inputs
    at::Tensor grad_grad_output = at::empty_like(grad_output);
    at::Tensor grad_omega_p1 = at::empty_like(omega_p1);
    at::Tensor grad_omega_p2 = at::empty_like(omega_p2);

    return std::make_tuple(grad_grad_output, grad_omega_p1, grad_omega_p2);
}

}  // namespace torchscience::meta::filter

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "butterworth_analog_bandpass_filter",
        &torchscience::meta::filter::butterworth_analog_bandpass_filter
    );

    module.impl(
        "butterworth_analog_bandpass_filter_backward",
        &torchscience::meta::filter::butterworth_analog_bandpass_filter_backward
    );

    module.impl(
        "butterworth_analog_bandpass_filter_backward_backward",
        &torchscience::meta::filter::butterworth_analog_bandpass_filter_backward_backward
    );
}
