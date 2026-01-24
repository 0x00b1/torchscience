// src/torchscience/csrc/autograd/linear_algebra/polar_decomposition.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/autograd.h>

namespace torchscience::autograd::linear_algebra {

class PolarDecompositionFunction : public torch::autograd::Function<PolarDecompositionFunction> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& a,
        c10::string_view side
    ) {
        at::AutoDispatchBelowAutograd guard;

        auto [U, P, info] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polar_decomposition", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&,
                c10::string_view
            )>()
            .call(a, side);

        // Save for backward: we need U, P and the original input
        ctx->save_for_backward({a, U, P});
        ctx->saved_data["side"] = std::string(side);

        return {U, P, info};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto a = saved[0];
        auto U_polar = saved[1];
        auto P = saved[2];
        auto side = ctx->saved_data["side"].toStringRef();

        auto grad_U = grad_outputs[0];
        auto grad_P = grad_outputs[1];
        // grad_info is None (integer output)

        at::Tensor grad_a;

        bool is_right = (side == "right");

        // Use PyTorch's autograd to compute gradients through SVD
        // This ensures numerical correctness by using the same gradient
        // formula that PyTorch uses internally.

        if (grad_U.defined() || grad_P.defined()) {
            // Enable gradient computation for this block
            at::AutoGradMode grad_mode(true);

            // Create a new tensor with gradient tracking
            auto a_grad = a.detach().requires_grad_(true);

            // Compute SVD with gradient tracking
            auto [U_svd, S, Vh] = at::linalg_svd(a_grad, /*full_matrices=*/false);

            // Convert S to the same dtype as input for complex support
            // (SVD returns real singular values even for complex input)
            at::Tensor S_typed = S.to(a_grad.scalar_type());

            // Compute polar factors from SVD
            // U_polar = U_svd @ Vh
            auto U_polar_recomputed = at::matmul(U_svd, Vh);

            at::Tensor P_recomputed;
            if (is_right) {
                // P = V @ diag(S) @ Vh = Vh.mH @ diag(S) @ Vh
                auto V = Vh.mH();
                P_recomputed = at::matmul(V, at::matmul(at::diag_embed(S_typed), Vh));
            } else {
                // P = U_svd @ diag(S) @ U_svd.mH
                P_recomputed = at::matmul(U_svd, at::matmul(at::diag_embed(S_typed), U_svd.mH()));
            }

            // Compute the loss function that would give the correct gradients
            // For complex tensors, the gradient is computed with respect to both
            // real and imaginary parts. We use the Wirtinger derivatives convention:
            // dL/dz = (dL/dx - i*dL/dy) / 2 where z = x + iy
            //
            // The inner product for the loss is: Re(tr(grad.H @ output))
            // which equals sum(real(conj(grad) * output))
            at::Tensor loss;

            if (grad_U.defined() && grad_P.defined()) {
                // Sum contributions from both U and P
                loss = at::sum(grad_U.conj() * U_polar_recomputed) +
                       at::sum(grad_P.conj() * P_recomputed);
            } else if (grad_U.defined()) {
                loss = at::sum(grad_U.conj() * U_polar_recomputed);
            } else {
                loss = at::sum(grad_P.conj() * P_recomputed);
            }

            // For complex tensors, take the real part of the loss
            // This is the standard convention for complex differentiation
            if (at::isComplexType(a.scalar_type())) {
                loss = at::real(loss);
            }

            // Backward pass to get grad_a
            loss.backward();

            grad_a = a_grad.grad();
        }

        return {grad_a, at::Tensor()};  // No gradient for 'side' string
    }
};

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> polar_decomposition(
    const at::Tensor& a,
    c10::string_view side
) {
    auto results = PolarDecompositionFunction::apply(a, side);
    return std::make_tuple(results[0], results[1], results[2]);
}

}  // namespace torchscience::autograd::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("polar_decomposition", &torchscience::autograd::linear_algebra::polar_decomposition);
}
