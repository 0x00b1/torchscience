#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::optimization::combinatorial {

inline at::Tensor sinkhorn(
    const at::Tensor& C,
    const at::Tensor& a,
    const at::Tensor& b,
    double epsilon,
    int64_t maxiter,
    double tol
) {
    TORCH_CHECK(C.dim() >= 2, "Cost matrix must have at least 2 dimensions");

    // Output shape is same as C
    return at::empty(C.sizes(), C.options().device(at::kMeta));
}

inline at::Tensor sinkhorn_backward(
    const at::Tensor& grad_output,
    const at::Tensor& P,
    const at::Tensor& C,
    double epsilon
) {
    return at::empty(C.sizes(), C.options().device(at::kMeta));
}

}  // namespace torchscience::meta::optimization::combinatorial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "sinkhorn",
        &torchscience::meta::optimization::combinatorial::sinkhorn
    );
    module.impl(
        "sinkhorn_backward",
        &torchscience::meta::optimization::combinatorial::sinkhorn_backward
    );
}
