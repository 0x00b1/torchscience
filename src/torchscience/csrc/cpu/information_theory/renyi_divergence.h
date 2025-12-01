#pragma once

#include <cmath>
#include <string>
#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "../../kernel/information_theory/renyi_divergence.h"
#include "../../kernel/information_theory/renyi_divergence_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

inline at::Tensor renyi_div_preprocess_input(
    const at::Tensor& input,
    int64_t dim,
    const std::string& input_type
) {
    if (input_type == "probability") {
        return input;
    } else if (input_type == "log_probability") {
        return input.exp();
    } else if (input_type == "logits") {
        return at::softmax(input, dim);
    } else {
        TORCH_CHECK(
            false,
            "renyi_divergence: input_type must be 'probability', 'log_probability', or 'logits', got '",
            input_type, "'"
        );
    }
}

inline at::Tensor renyi_div_apply_reduction(
    const at::Tensor& output,
    const std::string& reduction,
    int64_t batch_size
) {
    if (reduction == "none") {
        return output;
    } else if (reduction == "mean") {
        return output.mean();
    } else if (reduction == "batchmean") {
        if (batch_size == 0) {
            return output.sum();
        }
        return output.sum() / static_cast<double>(batch_size);
    } else if (reduction == "sum") {
        return output.sum();
    } else {
        TORCH_CHECK(
            false,
            "renyi_divergence: reduction must be 'none', 'mean', 'batchmean', or 'sum', got '",
            reduction, "'"
        );
    }
}

inline double renyi_div_get_log_base_scale(c10::optional<double> base) {
    if (!base.has_value()) {
        return 1.0;
    }
    double b = base.value();
    TORCH_CHECK(b > 0 && b != 1, "renyi_divergence: base must be positive and not equal to 1");
    return 1.0 / std::log(b);
}

}  // anonymous namespace

inline at::Tensor renyi_divergence(
    const at::Tensor& p,
    const at::Tensor& q,
    double alpha,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base,
    bool pairwise
) {
    TORCH_CHECK(alpha >= 0, "renyi_divergence: alpha must be >= 0, got ", alpha);
    TORCH_CHECK(std::abs(alpha - 1.0) > 1e-6, "renyi_divergence: alpha cannot be 1 (use kullback_leibler_divergence)");

    at::Tensor p_prob = renyi_div_preprocess_input(p, dim, input_type);
    at::Tensor q_prob = renyi_div_preprocess_input(q, dim, input_type);
    double log_base_scale = renyi_div_get_log_base_scale(base);

    if (pairwise) {
        TORCH_CHECK(
            p_prob.dim() == 2 && q_prob.dim() == 2,
            "renyi_divergence: pairwise mode requires 2D tensors"
        );
        TORCH_CHECK(
            p_prob.size(1) == q_prob.size(1),
            "renyi_divergence: feature dimensions must match"
        );

        int64_t m = p_prob.size(0);
        int64_t n = q_prob.size(0);
        int64_t d = p_prob.size(1);

        at::Tensor p_contig = p_prob.contiguous();
        at::Tensor q_contig = q_prob.contiguous();
        at::Tensor output = at::empty({m, n}, p_prob.options());

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            p_prob.scalar_type(),
            "renyi_divergence_pairwise_cpu",
            [&]() {
                const scalar_t* p_ptr = p_contig.data_ptr<scalar_t>();
                const scalar_t* q_ptr = q_contig.data_ptr<scalar_t>();
                scalar_t* out_ptr = output.data_ptr<scalar_t>();
                scalar_t alpha_t = static_cast<scalar_t>(alpha);
                scalar_t scale = static_cast<scalar_t>(log_base_scale);

                at::parallel_for(0, m * n, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        int64_t i = idx / n;
                        int64_t j = idx % n;
                        out_ptr[idx] = torchscience::kernel::information_theory::renyi_divergence_kernel<scalar_t>(
                            p_ptr + i * d,
                            q_ptr + j * d,
                            d,
                            alpha_t,
                            scale
                        );
                    }
                });
            }
        );

        return renyi_div_apply_reduction(output, reduction, m * n);
    }

    // Non-pairwise mode
    int64_t ndim = p_prob.dim();
    if (dim < 0) {
        dim = ndim + dim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "renyi_divergence: dim out of range");

    at::Tensor p_t = p_prob.transpose(dim, -1).contiguous();
    at::Tensor q_t = q_prob.transpose(dim, -1).contiguous();
    int64_t feature_size = p_t.size(-1);
    int64_t batch_size = p_t.numel() / feature_size;

    at::Tensor output = at::empty({batch_size}, p_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p_prob.scalar_type(),
        "renyi_divergence_cpu",
        [&]() {
            const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
            const scalar_t* q_ptr = q_t.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t alpha_t = static_cast<scalar_t>(alpha);
            scalar_t scale = static_cast<scalar_t>(log_base_scale);

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t idx = begin; idx < end; ++idx) {
                    out_ptr[idx] = torchscience::kernel::information_theory::renyi_divergence_kernel<scalar_t>(
                        p_ptr + idx * feature_size,
                        q_ptr + idx * feature_size,
                        feature_size,
                        alpha_t,
                        scale
                    );
                }
            });
        }
    );

    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            output_shape.push_back(p_prob.size(i));
        }
    }
    if (output_shape.empty()) {
        output = output.squeeze();
    } else {
        output = output.view(output_shape);
    }

    return renyi_div_apply_reduction(output, reduction, batch_size);
}

inline std::tuple<at::Tensor, at::Tensor> renyi_divergence_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    double alpha,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base,
    bool pairwise
) {
    at::Tensor p_prob = renyi_div_preprocess_input(p, dim, input_type);
    at::Tensor q_prob = renyi_div_preprocess_input(q, dim, input_type);
    double log_base_scale = renyi_div_get_log_base_scale(base);

    // Non-pairwise mode only for now
    TORCH_CHECK(!pairwise, "renyi_divergence_backward: pairwise mode not yet supported");

    int64_t ndim = p_prob.dim();
    if (dim < 0) {
        dim = ndim + dim;
    }

    at::Tensor p_t = p_prob.transpose(dim, -1).contiguous();
    at::Tensor q_t = q_prob.transpose(dim, -1).contiguous();
    int64_t feature_size = p_t.size(-1);
    int64_t batch_size = p_t.numel() / feature_size;

    at::Tensor grad_p_t = at::zeros_like(p_t);
    at::Tensor grad_q_t = at::zeros_like(q_t);

    double scale = 1.0;
    if (reduction == "mean") {
        scale = 1.0 / static_cast<double>(batch_size);
    } else if (reduction == "batchmean") {
        scale = 1.0 / static_cast<double>(batch_size);
    }

    at::Tensor grad_flat = grad_output.contiguous().view({-1});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p_prob.scalar_type(),
        "renyi_divergence_backward_cpu",
        [&]() {
            const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
            const scalar_t* q_ptr = q_t.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_flat.data_ptr<scalar_t>();
            scalar_t* grad_p_ptr = grad_p_t.data_ptr<scalar_t>();
            scalar_t* grad_q_ptr = grad_q_t.data_ptr<scalar_t>();
            scalar_t alpha_t = static_cast<scalar_t>(alpha);
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);
            scalar_t reduction_scale = static_cast<scalar_t>(scale);

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t idx = begin; idx < end; ++idx) {
                    scalar_t grad_val;
                    if (reduction == "none") {
                        grad_val = grad_ptr[idx];
                    } else {
                        grad_val = grad_ptr[0] * reduction_scale;
                    }

                    torchscience::kernel::information_theory::renyi_divergence_backward_kernel<scalar_t>(
                        grad_val,
                        p_ptr + idx * feature_size,
                        q_ptr + idx * feature_size,
                        feature_size,
                        alpha_t,
                        log_scale,
                        grad_p_ptr + idx * feature_size,
                        grad_q_ptr + idx * feature_size
                    );
                }
            });
        }
    );

    return std::make_tuple(
        grad_p_t.transpose(dim, -1).contiguous(),
        grad_q_t.transpose(dim, -1).contiguous()
    );
}

}  // namespace torchscience::cpu::information_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("renyi_divergence", &torchscience::cpu::information_theory::renyi_divergence);
    m.impl("renyi_divergence_backward", &torchscience::cpu::information_theory::renyi_divergence_backward);
}
