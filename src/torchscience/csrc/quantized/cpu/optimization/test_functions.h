#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::quantized::cpu::optimization::test_functions {

/**
 * Quantized CPU implementation of the Rosenbrock function.
 *
 * For quantized input x, we dequantize before computation.
 * The output is a dense float tensor (reduction result), not quantized,
 * since the function value range is unbounded and quantization would
 * introduce significant error.
 *
 * Parameters a and b can be either quantized or regular tensors.
 */
inline at::Tensor rosenbrock(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    TORCH_CHECK(
        x.is_quantized(),
        "rosenbrock (QuantizedCPU) expects quantized tensor for x"
    );

    // Dequantize x for computation
    at::Tensor x_dequant = x.dequantize();

    // Handle a and b - dequantize if quantized
    at::Tensor a_dequant = a.is_quantized() ? a.dequantize() : a;
    at::Tensor b_dequant = b.is_quantized() ? b.dequantize() : b;

    // Call the dense implementation
    // Result is a float tensor (reduction), not quantized
    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::rosenbrock", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(x_dequant, a_dequant, b_dequant);
}

/**
 * Backward pass for quantized CPU Rosenbrock.
 *
 * Returns dequantized gradients since:
 * 1. Quantized tensors don't support autograd in the traditional sense
 * 2. Training with quantized weights requires float gradients
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> rosenbrock_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    TORCH_CHECK(
        x.is_quantized(),
        "rosenbrock_backward (QuantizedCPU) expects quantized tensor for x"
    );

    // Dequantize inputs
    at::Tensor x_dequant = x.dequantize();
    at::Tensor a_dequant = a.is_quantized() ? a.dequantize() : a;
    at::Tensor b_dequant = b.is_quantized() ? b.dequantize() : b;

    // grad_output is already a float tensor (from forward pass)
    // Compute gradients in float
    auto [grad_x, grad_a, grad_b] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::rosenbrock_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&
        )>()
        .call(grad_output, x_dequant, a_dequant, b_dequant);

    // Return float gradients (not quantized) for training
    return {grad_x, grad_a, grad_b};
}

/**
 * Double backward pass for quantized CPU Rosenbrock.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> rosenbrock_backward_backward(
    const at::Tensor& grad_grad_x,
    const at::Tensor& grad_grad_a,
    const at::Tensor& grad_grad_b,
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    // Dequantize inputs
    at::Tensor x_dequant = x.is_quantized() ? x.dequantize() : x;
    at::Tensor a_dequant = a.is_quantized() ? a.dequantize() : a;
    at::Tensor b_dequant = b.is_quantized() ? b.dequantize() : b;

    // grad_grad tensors are already float (from backward pass)
    // Compute double backward in float
    auto [gg_output, grad_x, grad_a, grad_b] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::rosenbrock_backward_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&
        )>()
        .call(grad_grad_x, grad_grad_a, grad_grad_b, grad_output, x_dequant, a_dequant, b_dequant);

    // Return float results
    return {gg_output, grad_x, grad_a, grad_b};
}

}  // namespace torchscience::quantized::cpu::optimization::test_functions

TORCH_LIBRARY_IMPL(torchscience, QuantizedCPU, module) {
    module.impl(
        "rosenbrock",
        &torchscience::quantized::cpu::optimization::test_functions::rosenbrock
    );
    module.impl(
        "rosenbrock_backward",
        &torchscience::quantized::cpu::optimization::test_functions::rosenbrock_backward
    );
    module.impl(
        "rosenbrock_backward_backward",
        &torchscience::quantized::cpu::optimization::test_functions::rosenbrock_backward_backward
    );
}
