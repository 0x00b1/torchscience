#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::quantized::cpu::integral_transform {

/**
 * Quantized CPU implementation of inverse Hilbert transform.
 */
inline at::Tensor inverse_hilbert_transform(
    const at::Tensor& input,
    [[maybe_unused]] int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    TORCH_CHECK(
        input.is_quantized(),
        "inverse_hilbert_transform (QuantizedCPU) expects quantized tensor"
    );

    at::Tensor input_dequant = input.dequantize();

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::inverse_hilbert_transform", "")
        .typed<at::Tensor(const at::Tensor&, int64_t, int64_t, int64_t, double, const c10::optional<at::Tensor>&)>()
        .call(input_dequant, n_param, dim, padding_mode, padding_value, window);
}

/**
 * Backward pass for quantized CPU inverse Hilbert transform.
 */
inline at::Tensor inverse_hilbert_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    [[maybe_unused]] int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    TORCH_CHECK(
        input.is_quantized(),
        "inverse_hilbert_transform_backward (QuantizedCPU) expects quantized tensor for input"
    );

    at::Tensor input_dequant = input.dequantize();

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::inverse_hilbert_transform_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, double, const c10::optional<at::Tensor>&)>()
        .call(grad_output, input_dequant, n_param, dim, padding_mode, padding_value, window);
}

/**
 * Double backward pass for quantized CPU inverse Hilbert transform.
 */
inline std::tuple<at::Tensor, at::Tensor> inverse_hilbert_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    [[maybe_unused]] int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    at::Tensor input_dequant = input.is_quantized() ? input.dequantize() : input;

    auto [gg_output, new_grad_input] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::inverse_hilbert_transform_backward_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, double, const c10::optional<at::Tensor>&
        )>()
        .call(grad_grad_input, grad_output, input_dequant, n_param, dim, padding_mode, padding_value, window);

    return {gg_output, new_grad_input};
}

}  // namespace torchscience::quantized::cpu::integral_transform

TORCH_LIBRARY_IMPL(torchscience, QuantizedCPU, module) {
    module.impl(
        "inverse_hilbert_transform",
        &torchscience::quantized::cpu::integral_transform::inverse_hilbert_transform
    );
    module.impl(
        "inverse_hilbert_transform_backward",
        &torchscience::quantized::cpu::integral_transform::inverse_hilbert_transform_backward
    );
    module.impl(
        "inverse_hilbert_transform_backward_backward",
        &torchscience::quantized::cpu::integral_transform::inverse_hilbert_transform_backward_backward
    );
}
