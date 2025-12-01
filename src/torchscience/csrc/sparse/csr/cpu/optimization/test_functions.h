#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::sparse::csr::cpu::optimization::test_functions {

/**
 * Sparse CSR implementation of the Rosenbrock function.
 *
 * For sparse input, we convert to dense for computation since Rosenbrock
 * requires consecutive elements. The output is always dense (reduction).
 */
inline at::Tensor rosenbrock(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    TORCH_CHECK(
        x.layout() == at::kSparseCsr,
        "rosenbrock (SparseCsrCPU) expects sparse CSR tensor for x"
    );

    // Convert sparse to dense for computation
    at::Tensor x_dense = x.to_dense();

    // Handle a and b - convert if sparse
    at::Tensor a_dense = (a.layout() == at::kSparseCsr) ? a.to_dense() : a;
    at::Tensor b_dense = (b.layout() == at::kSparseCsr) ? b.to_dense() : b;

    // Call the dense implementation
    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::rosenbrock", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(x_dense, a_dense, b_dense);
}

/**
 * Backward pass for sparse CSR Rosenbrock.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> rosenbrock_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    TORCH_CHECK(
        x.layout() == at::kSparseCsr,
        "rosenbrock_backward (SparseCsrCPU) expects sparse CSR tensor for x"
    );

    // Convert to dense for gradient computation
    at::Tensor x_dense = x.to_dense();
    at::Tensor a_dense = (a.layout() == at::kSparseCsr) ? a.to_dense() : a;
    at::Tensor b_dense = (b.layout() == at::kSparseCsr) ? b.to_dense() : b;

    // Compute dense gradients
    auto [grad_x_dense, grad_a, grad_b] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::rosenbrock_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&
        )>()
        .call(grad_output, x_dense, a_dense, b_dense);

    // Convert grad_x back to sparse CSR
    // For CSR, we need to extract values at the same positions as the input
    at::Tensor crow_indices = x.crow_indices();
    at::Tensor col_indices = x.col_indices();

    // Get the number of rows
    int64_t num_rows = crow_indices.size(0) - 1;

    // Extract gradient values at sparse positions
    std::vector<at::Tensor> grad_value_list;
    for (int64_t row = 0; row < num_rows; ++row) {
        int64_t row_start = crow_indices[row].item<int64_t>();
        int64_t row_end = crow_indices[row + 1].item<int64_t>();
        for (int64_t idx = row_start; idx < row_end; ++idx) {
            int64_t col = col_indices[idx].item<int64_t>();
            grad_value_list.push_back(grad_x_dense[row][col].unsqueeze(0));
        }
    }

    at::Tensor grad_values = grad_value_list.empty()
        ? at::empty({0}, grad_x_dense.options())
        : at::cat(grad_value_list, 0);

    at::Tensor grad_x = at::_sparse_csr_tensor_unsafe(
        crow_indices,
        col_indices,
        grad_values,
        x.sizes(),
        x.options().layout(at::kSparseCsr).dtype(grad_values.scalar_type())
    );

    // Handle sparse a and b gradients
    at::Tensor grad_a_out = (a.layout() == at::kSparseCsr)
        ? grad_a.to_sparse_csr()
        : grad_a;
    at::Tensor grad_b_out = (b.layout() == at::kSparseCsr)
        ? grad_b.to_sparse_csr()
        : grad_b;

    return {grad_x, grad_a_out, grad_b_out};
}

/**
 * Double backward pass for sparse CSR Rosenbrock.
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
    // Convert all inputs to dense
    at::Tensor x_dense = (x.layout() == at::kSparseCsr) ? x.to_dense() : x;
    at::Tensor a_dense = (a.layout() == at::kSparseCsr) ? a.to_dense() : a;
    at::Tensor b_dense = (b.layout() == at::kSparseCsr) ? b.to_dense() : b;

    at::Tensor gg_x_dense = grad_grad_x.defined() && (grad_grad_x.layout() == at::kSparseCsr)
        ? grad_grad_x.to_dense() : grad_grad_x;
    at::Tensor gg_a_dense = grad_grad_a.defined() && (grad_grad_a.layout() == at::kSparseCsr)
        ? grad_grad_a.to_dense() : grad_grad_a;
    at::Tensor gg_b_dense = grad_grad_b.defined() && (grad_grad_b.layout() == at::kSparseCsr)
        ? grad_grad_b.to_dense() : grad_grad_b;

    // Compute dense double backward
    auto [gg_output, grad_x_dense, grad_a, grad_b] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::rosenbrock_backward_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&
        )>()
        .call(gg_x_dense, gg_a_dense, gg_b_dense, grad_output, x_dense, a_dense, b_dense);

    // Convert grad_x back to sparse CSR if input was sparse
    at::Tensor grad_x;
    if (x.layout() == at::kSparseCsr && grad_x_dense.defined()) {
        grad_x = grad_x_dense.to_sparse_csr();
    } else {
        grad_x = grad_x_dense;
    }

    return {gg_output, grad_x, grad_a, grad_b};
}

}  // namespace torchscience::sparse::csr::cpu::optimization::test_functions

TORCH_LIBRARY_IMPL(torchscience, SparseCsrCPU, module) {
    module.impl(
        "rosenbrock",
        &torchscience::sparse::csr::cpu::optimization::test_functions::rosenbrock
    );
    module.impl(
        "rosenbrock_backward",
        &torchscience::sparse::csr::cpu::optimization::test_functions::rosenbrock_backward
    );
    module.impl(
        "rosenbrock_backward_backward",
        &torchscience::sparse::csr::cpu::optimization::test_functions::rosenbrock_backward_backward
    );
}
