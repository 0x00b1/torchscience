// src/torchscience/csrc/meta/information_theory/jensen_shannon_divergence.h
#pragma once

#include <string>
#include <tuple>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

namespace torchscience::meta::information_theory {

/**
 * Meta implementation for JS divergence shape inference.
 *
 * @param p First probability distribution
 * @param q Second probability distribution
 * @param dim Dimension along which to compute divergence
 * @param input_type Type of input: "probability", "log_probability", or "logits"
 * @param reduction Reduction to apply: "none", "mean", "batchmean", or "sum"
 * @param base Optional log base (does not affect shape)
 * @param pairwise If true, compute all-pairs divergence matrix
 * @return Empty tensor with correct output shape
 */
inline at::Tensor jensen_shannon_divergence(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    const c10::optional<double>& base,
    bool pairwise
) {
    if (pairwise) {
        // Pairwise mode: output shape is {m, n} for reduction="none", else scalar
        TORCH_CHECK(
            p.dim() == 2 && q.dim() == 2,
            "jensen_shannon_divergence: pairwise mode requires 2D tensors"
        );

        int64_t m = p.size(0);
        int64_t n = q.size(0);

        if (reduction == "none") {
            return at::empty({m, n}, p.options());
        } else {
            return at::empty({}, p.options());
        }
    } else {
        // Standard mode: remove dim from p.sizes()
        int64_t ndim = p.dim();
        int64_t normalized_dim = dim < 0 ? ndim + dim : dim;

        std::vector<int64_t> output_shape;
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != normalized_dim) {
                output_shape.push_back(p.size(i));
            }
        }

        if (reduction == "none") {
            if (output_shape.empty()) {
                return at::empty({}, p.options());
            }
            return at::empty(output_shape, p.options());
        } else {
            return at::empty({}, p.options());
        }
    }
}

/**
 * Meta implementation for JS divergence backward shape inference.
 *
 * @param grad_output Upstream gradient
 * @param p First probability distribution
 * @param q Second probability distribution
 * @param dim Dimension along which divergence was computed
 * @param input_type Type of input
 * @param reduction Reduction that was applied
 * @param base Optional log base (does not affect shape)
 * @param pairwise Whether pairwise mode was used
 * @return Tuple of empty tensors with shapes matching (grad_p, grad_q)
 */
inline std::tuple<at::Tensor, at::Tensor> jensen_shannon_divergence_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    const c10::optional<double>& base,
    bool pairwise
) {
    return std::make_tuple(
        at::empty_like(p),
        at::empty_like(q)
    );
}

/**
 * Meta implementation for JS divergence backward backward shape inference.
 *
 * @param gg_p Upstream gradient w.r.t. grad_p
 * @param gg_q Upstream gradient w.r.t. grad_q
 * @param grad_output Original upstream gradient
 * @param p First probability distribution
 * @param q Second probability distribution
 * @param dim Dimension along which divergence was computed
 * @param input_type Type of input
 * @param reduction Reduction that was applied
 * @param base Optional log base (does not affect shape)
 * @param pairwise Whether pairwise mode was used
 * @return Tuple of empty tensors (grad_grad_output, grad_p, grad_q)
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> jensen_shannon_divergence_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& gg_q,
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    const c10::optional<double>& base,
    bool pairwise
) {
    return std::make_tuple(
        at::empty({}, grad_output.options()),
        at::empty_like(p),
        at::empty_like(q)
    );
}

}  // namespace torchscience::meta::information_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("jensen_shannon_divergence", &torchscience::meta::information_theory::jensen_shannon_divergence);
    m.impl("jensen_shannon_divergence_backward", &torchscience::meta::information_theory::jensen_shannon_divergence_backward);
    m.impl("jensen_shannon_divergence_backward_backward", &torchscience::meta::information_theory::jensen_shannon_divergence_backward_backward);
}
