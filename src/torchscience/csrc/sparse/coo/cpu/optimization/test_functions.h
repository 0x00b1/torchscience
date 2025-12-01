#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::sparse::coo::cpu::optimization::test_functions {

/**
 * Sparse COO implementation of the Rosenbrock function.
 *
 * For sparse input, we convert to dense for computation since Rosenbrock
 * requires consecutive elements. The output is always dense (reduction).
 *
 * Note: Sparse tensors with implicit zeros will be treated as having zeros
 * at those positions, which affects the Rosenbrock computation.
 */
inline at::Tensor rosenbrock(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    TORCH_CHECK(
        x.is_sparse(),
        "rosenbrock (SparseCPU) expects sparse COO tensor for x"
    );

    // Convert sparse to dense for computation
    // Rosenbrock requires consecutive elements, so we need the full tensor
    at::Tensor x_dense = x.to_dense();

    // Handle a and b - convert if sparse
    at::Tensor a_dense = a.is_sparse() ? a.to_dense() : a;
    at::Tensor b_dense = b.is_sparse() ? b.to_dense() : b;

    // Call the dense implementation
    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::rosenbrock", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(x_dense, a_dense, b_dense);
}

/**
 * Backward pass for sparse COO Rosenbrock.
 *
 * Returns gradients in sparse format matching the input sparsity pattern.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> rosenbrock_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    TORCH_CHECK(
        x.is_sparse(),
        "rosenbrock_backward (SparseCPU) expects sparse COO tensor for x"
    );

    // Convert to dense for gradient computation
    at::Tensor x_dense = x.to_dense();
    at::Tensor a_dense = a.is_sparse() ? a.to_dense() : a;
    at::Tensor b_dense = b.is_sparse() ? b.to_dense() : b;

    // Compute dense gradients
    auto [grad_x_dense, grad_a, grad_b] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::rosenbrock_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&
        )>()
        .call(grad_output, x_dense, a_dense, b_dense);

    // Convert grad_x back to sparse with same sparsity pattern as input
    // We use the indices from x and gather the corresponding gradient values
    at::Tensor indices = x._indices();
    at::Tensor grad_values = grad_x_dense.index({indices[0], indices[1]});

    at::Tensor grad_x = at::_sparse_coo_tensor_unsafe(
        indices,
        grad_values,
        x.sizes(),
        x.options().dtype(grad_values.scalar_type())
    )._coalesced_(x.is_coalesced());

    // Handle sparse a and b gradients
    at::Tensor grad_a_out = a.is_sparse()
        ? grad_a.to_sparse()
        : grad_a;
    at::Tensor grad_b_out = b.is_sparse()
        ? grad_b.to_sparse()
        : grad_b;

    return {grad_x, grad_a_out, grad_b_out};
}

/**
 * Double backward pass for sparse COO Rosenbrock.
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
    at::Tensor x_dense = x.is_sparse() ? x.to_dense() : x;
    at::Tensor a_dense = a.is_sparse() ? a.to_dense() : a;
    at::Tensor b_dense = b.is_sparse() ? b.to_dense() : b;

    at::Tensor gg_x_dense = grad_grad_x.defined() && grad_grad_x.is_sparse()
        ? grad_grad_x.to_dense() : grad_grad_x;
    at::Tensor gg_a_dense = grad_grad_a.defined() && grad_grad_a.is_sparse()
        ? grad_grad_a.to_dense() : grad_grad_a;
    at::Tensor gg_b_dense = grad_grad_b.defined() && grad_grad_b.is_sparse()
        ? grad_grad_b.to_dense() : grad_grad_b;

    // Compute dense double backward
    auto [gg_output, grad_x_dense, grad_a, grad_b] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::rosenbrock_backward_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&
        )>()
        .call(gg_x_dense, gg_a_dense, gg_b_dense, grad_output, x_dense, a_dense, b_dense);

    // Convert grad_x back to sparse if input was sparse
    at::Tensor grad_x;
    if (x.is_sparse() && grad_x_dense.defined()) {
        at::Tensor indices = x._indices();
        at::Tensor grad_values = grad_x_dense.index({indices[0], indices[1]});
        grad_x = at::_sparse_coo_tensor_unsafe(
            indices,
            grad_values,
            x.sizes(),
            x.options().dtype(grad_values.scalar_type())
        )._coalesced_(x.is_coalesced());
    } else {
        grad_x = grad_x_dense;
    }

    return {gg_output, grad_x, grad_a, grad_b};
}

}  // namespace torchscience::sparse::coo::cpu::optimization::test_functions

TORCH_LIBRARY_IMPL(torchscience, SparseCPU, module) {
    module.impl(
        "rosenbrock",
        &torchscience::sparse::coo::cpu::optimization::test_functions::rosenbrock
    );
    module.impl(
        "rosenbrock_backward",
        &torchscience::sparse::coo::cpu::optimization::test_functions::rosenbrock_backward
    );
    module.impl(
        "rosenbrock_backward_backward",
        &torchscience::sparse::coo::cpu::optimization::test_functions::rosenbrock_backward_backward
    );
}
