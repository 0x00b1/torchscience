#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::sparse::csr::cuda::transform {

/**
 * Sparse CSR CUDA implementation of discrete wavelet transform.
 *
 * Baseline implementation: converts to dense, computes, returns dense.
 * Future optimization: exploit sparsity structure.
 */
inline at::Tensor discrete_wavelet_transform(
    const at::Tensor& input,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode
) {
    TORCH_CHECK(
        input.layout() == at::kSparseCsr && input.is_cuda(),
        "discrete_wavelet_transform (SparseCsrCUDA) expects sparse CSR CUDA tensor"
    );

    at::Tensor input_dense = input.to_dense();

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::discrete_wavelet_transform", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t)>()
        .call(input_dense, filter_lo, filter_hi, levels, mode);
}

/**
 * Backward pass for sparse CSR CUDA discrete wavelet transform.
 */
inline at::Tensor discrete_wavelet_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode,
    int64_t input_length
) {
    TORCH_CHECK(
        input.layout() == at::kSparseCsr && input.is_cuda(),
        "discrete_wavelet_transform_backward (SparseCsrCUDA) expects sparse CSR CUDA tensor for input"
    );

    at::Tensor input_dense = input.to_dense();
    at::Tensor grad_output_dense = (grad_output.layout() == at::kSparseCsr)
        ? grad_output.to_dense() : grad_output;

    at::Tensor grad_input_dense = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::discrete_wavelet_transform_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t)>()
        .call(grad_output_dense, input_dense, filter_lo, filter_hi, levels, mode, input_length);

    return grad_input_dense.to_sparse_csr();
}

/**
 * Double backward pass for sparse CSR CUDA discrete wavelet transform.
 */
inline std::tuple<at::Tensor, at::Tensor> discrete_wavelet_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode,
    int64_t input_length
) {
    at::Tensor input_dense = (input.layout() == at::kSparseCsr)
        ? input.to_dense() : input;
    at::Tensor grad_output_dense = (grad_output.layout() == at::kSparseCsr)
        ? grad_output.to_dense() : grad_output;
    at::Tensor gg_input_dense = (grad_grad_input.layout() == at::kSparseCsr)
        ? grad_grad_input.to_dense() : grad_grad_input;

    auto [gg_output, new_grad_input] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::discrete_wavelet_transform_backward_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t
        )>()
        .call(gg_input_dense, grad_output_dense, input_dense, filter_lo, filter_hi, levels, mode, input_length);

    return {gg_output, new_grad_input};
}

}  // namespace torchscience::sparse::csr::cuda::transform

TORCH_LIBRARY_IMPL(torchscience, SparseCsrCUDA, module) {
    module.impl(
        "discrete_wavelet_transform",
        &torchscience::sparse::csr::cuda::transform::discrete_wavelet_transform
    );
    module.impl(
        "discrete_wavelet_transform_backward",
        &torchscience::sparse::csr::cuda::transform::discrete_wavelet_transform_backward
    );
    module.impl(
        "discrete_wavelet_transform_backward_backward",
        &torchscience::sparse::csr::cuda::transform::discrete_wavelet_transform_backward_backward
    );
}
