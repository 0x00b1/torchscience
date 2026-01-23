// src/torchscience/csrc/autograd/linear_algebra/symmetric_generalized_eigenvalue.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/autograd.h>

namespace torchscience::autograd::linear_algebra {

class SymmetricGeneralizedEigenvalueFunction : public torch::autograd::Function<SymmetricGeneralizedEigenvalueFunction> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& a,
        const at::Tensor& b
    ) {
        at::AutoDispatchBelowAutograd guard;
        auto [eigenvalues, eigenvectors, info] =
            at::_ops::torchscience_symmetric_generalized_eigenvalue::call(a, b);

        ctx->save_for_backward({eigenvalues, eigenvectors, a, b});

        return {eigenvalues, eigenvectors, info};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto eigenvalues = saved[0];
        auto eigenvectors = saved[1];
        auto a = saved[2];
        auto b = saved[3];

        auto grad_eigenvalues = grad_outputs[0];
        auto grad_eigenvectors = grad_outputs[1];
        // grad_info is always None (integer output)

        at::Tensor grad_a;
        at::Tensor grad_b;

        auto V = eigenvectors;
        auto lam = eigenvalues;

        // Eigenvalue gradients: dL/dA = V @ diag(dL/dlam) @ V^T
        if (grad_eigenvalues.defined()) {
            grad_a = at::matmul(V, at::matmul(at::diag_embed(grad_eigenvalues), V.mH()));
            grad_b = -at::matmul(V, at::matmul(at::diag_embed(lam * grad_eigenvalues), V.mH()));
        }

        // Eigenvector gradients via perturbation theory
        if (grad_eigenvectors.defined()) {
            auto BV = at::matmul(b, V);

            // F_ij = 1/(lam_i - lam_j) for i != j, 0 otherwise
            auto lam_diff = lam.unsqueeze(-1) - lam.unsqueeze(-2);
            auto F = at::where(
                at::abs(lam_diff) > 1e-10,
                1.0 / lam_diff,
                at::zeros_like(lam_diff)
            );
            // Zero diagonal
            F = F - at::diag_embed(at::diagonal(F, 0, -2, -1));

            auto proj = at::matmul(BV.mH(), grad_eigenvectors);
            auto contrib = at::matmul(V, at::matmul(F * proj, V.mH()));

            auto a_contrib = at::matmul(at::matmul(b, contrib), b);
            auto b_contrib = -at::matmul(at::matmul(a, contrib), b) - at::matmul(at::matmul(b, contrib), a);

            if (grad_a.defined()) {
                grad_a = grad_a + a_contrib;
            } else {
                grad_a = a_contrib;
            }

            if (grad_b.defined()) {
                grad_b = grad_b + b_contrib;
            } else {
                grad_b = b_contrib;
            }
        }

        // Symmetrize gradients
        if (grad_a.defined()) {
            grad_a = (grad_a + grad_a.mH()) / 2;
        }
        if (grad_b.defined()) {
            grad_b = (grad_b + grad_b.mH()) / 2;
        }

        return {grad_a, grad_b};
    }
};

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> symmetric_generalized_eigenvalue(
    const at::Tensor& a,
    const at::Tensor& b
) {
    auto results = SymmetricGeneralizedEigenvalueFunction::apply(a, b);
    return std::make_tuple(results[0], results[1], results[2]);
}

}  // namespace torchscience::autograd::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("symmetric_generalized_eigenvalue", &torchscience::autograd::linear_algebra::symmetric_generalized_eigenvalue);
}
