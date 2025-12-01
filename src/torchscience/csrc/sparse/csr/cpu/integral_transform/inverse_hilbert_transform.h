#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::sparse::csr::cpu::integral_transform {

/**
 * Sparse CSR CPU implementation of inverse Hilbert transform.
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
        input.layout() == at::kSparseCsr,
        "inverse_hilbert_transform (SparseCsrCPU) expects sparse CSR tensor"
    );

    at::Tensor input_dense = input.to_dense();

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::inverse_hilbert_transform", "")
        .typed<at::Tensor(const at::Tensor&, int64_t, int64_t, int64_t, double, const c10::optional<at::Tensor>&)>()
        .call(input_dense, n_param, dim, padding_mode, padding_value, window);
}

/**
 * Backward pass for sparse CSR CPU inverse Hilbert transform.
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
        input.layout() == at::kSparseCsr,
        "inverse_hilbert_transform_backward (SparseCsrCPU) expects sparse CSR tensor for input"
    );

    at::Tensor input_dense = input.to_dense();
    at::Tensor grad_output_dense = (grad_output.layout() == at::kSparseCsr)
        ? grad_output.to_dense() : grad_output;

    at::Tensor grad_input_dense = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::inverse_hilbert_transform_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, double, const c10::optional<at::Tensor>&)>()
        .call(grad_output_dense, input_dense, n_param, dim, padding_mode, padding_value, window);

    return grad_input_dense.to_sparse_csr();
}

/**
 * Double backward pass for sparse CSR CPU inverse Hilbert transform.
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
    at::Tensor input_dense = (input.layout() == at::kSparseCsr)
        ? input.to_dense() : input;
    at::Tensor grad_output_dense = (grad_output.layout() == at::kSparseCsr)
        ? grad_output.to_dense() : grad_output;
    at::Tensor gg_input_dense = (grad_grad_input.layout() == at::kSparseCsr)
        ? grad_grad_input.to_dense() : grad_grad_input;

    auto [gg_output, new_grad_input] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::inverse_hilbert_transform_backward_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, double, const c10::optional<at::Tensor>&
        )>()
        .call(gg_input_dense, grad_output_dense, input_dense, n_param, dim, padding_mode, padding_value, window);

    return {gg_output, new_grad_input};
}

}  // namespace torchscience::sparse::csr::cpu::integral_transform

TORCH_LIBRARY_IMPL(torchscience, SparseCsrCPU, module) {
    module.impl(
        "inverse_hilbert_transform",
        &torchscience::sparse::csr::cpu::integral_transform::inverse_hilbert_transform
    );
    module.impl(
        "inverse_hilbert_transform_backward",
        &torchscience::sparse::csr::cpu::integral_transform::inverse_hilbert_transform_backward
    );
    module.impl(
        "inverse_hilbert_transform_backward_backward",
        &torchscience::sparse::csr::cpu::integral_transform::inverse_hilbert_transform_backward_backward
    );
}
