#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::optimization::test_functions {

/**
 * Meta kernel for Rosenbrock function.
 *
 * Computes output shape without performing actual computation.
 * Input shape: (..., n) where n >= 2
 * Output shape: (...)  - reduction along last dimension
 */
inline at::Tensor rosenbrock(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    TORCH_CHECK(
        x.dim() >= 1 && x.size(-1) >= 2,
        "rosenbrock requires at least 2 elements in the last dimension"
    );

    // Output shape is input shape with last dimension removed
    auto output_sizes = x.sizes().vec();
    output_sizes.pop_back();

    // Determine output dtype (promote inputs to common dtype)
    auto ab_dtype = at::result_type(a, b);
    auto output_dtype = at::promote_types(x.scalar_type(), ab_dtype);

    // Create meta tensor with correct shape and dtype
    return at::empty(
        output_sizes,
        x.options().dtype(output_dtype).device(at::kMeta)
    );
}

/**
 * Meta kernel for Rosenbrock backward.
 *
 * grad_output shape: (...)
 * x shape: (..., n)
 * Output: grad_x (..., n), grad_a (broadcast shape), grad_b (broadcast shape)
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> rosenbrock_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    auto ab_dtype = at::result_type(a, b);
    auto output_dtype = at::promote_types(x.scalar_type(), ab_dtype);

    // grad_x has same shape as x
    at::Tensor grad_x = at::empty(
        x.sizes(),
        x.options().dtype(output_dtype).device(at::kMeta)
    );

    // grad_a has same shape as a (or broadcast compatible)
    at::Tensor grad_a = at::empty(
        a.sizes(),
        a.options().dtype(output_dtype).device(at::kMeta)
    );

    // grad_b has same shape as b (or broadcast compatible)
    at::Tensor grad_b = at::empty(
        b.sizes(),
        b.options().dtype(output_dtype).device(at::kMeta)
    );

    return {grad_x, grad_a, grad_b};
}

/**
 * Meta kernel for Rosenbrock double backward.
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
    const bool has_grad_grad_x = grad_grad_x.defined();
    const bool has_grad_grad_a = grad_grad_a.defined();
    const bool has_grad_grad_b = grad_grad_b.defined();

    if (!has_grad_grad_x && !has_grad_grad_a && !has_grad_grad_b) {
        return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }

    auto ab_dtype = at::result_type(a, b);
    auto output_dtype = at::promote_types(x.scalar_type(), ab_dtype);

    // grad_grad_output has same shape as grad_output
    at::Tensor gg_output = at::empty(
        grad_output.sizes(),
        grad_output.options().dtype(output_dtype).device(at::kMeta)
    );

    // grad_x has same shape as x
    at::Tensor grad_x = at::empty(
        x.sizes(),
        x.options().dtype(output_dtype).device(at::kMeta)
    );

    // grad_a has same shape as a
    at::Tensor grad_a = at::empty(
        a.sizes(),
        a.options().dtype(output_dtype).device(at::kMeta)
    );

    // grad_b has same shape as b
    at::Tensor grad_b = at::empty(
        b.sizes(),
        b.options().dtype(output_dtype).device(at::kMeta)
    );

    return {gg_output, grad_x, grad_a, grad_b};
}

}  // namespace torchscience::meta::optimization::test_functions

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "rosenbrock",
        &torchscience::meta::optimization::test_functions::rosenbrock
    );
    module.impl(
        "rosenbrock_backward",
        &torchscience::meta::optimization::test_functions::rosenbrock_backward
    );
    module.impl(
        "rosenbrock_backward_backward",
        &torchscience::meta::optimization::test_functions::rosenbrock_backward_backward
    );
}
