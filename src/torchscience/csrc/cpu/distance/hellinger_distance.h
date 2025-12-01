#pragma once

#include <cmath>
#include <string>
#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "../../kernel/distance/hellinger_distance.h"
#include "../../kernel/distance/hellinger_distance_backward.h"

namespace torchscience::cpu::distance {

namespace {

/**
 * Preprocess input based on input_type.
 *
 * @param input The input tensor
 * @param dim The dimension along which to compute distance
 * @param input_type One of "probability", "log_probability", or "logits"
 * @return Tensor of probabilities
 */
inline at::Tensor preprocess_input(
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
            "hellinger_distance: input_type must be 'probability', 'log_probability', or 'logits', got '",
            input_type, "'"
        );
    }
}

/**
 * Apply reduction to the output tensor.
 *
 * @param output The unreduced output tensor
 * @param reduction One of "none", "mean", or "sum"
 * @return Reduced tensor
 */
inline at::Tensor apply_reduction(
    const at::Tensor& output,
    const std::string& reduction
) {
    if (reduction == "none") {
        return output;
    } else if (reduction == "mean") {
        return output.mean();
    } else if (reduction == "sum") {
        return output.sum();
    } else {
        TORCH_CHECK(
            false,
            "hellinger_distance: reduction must be 'none', 'mean', or 'sum', got '",
            reduction, "'"
        );
    }
}

}  // anonymous namespace

/**
 * Compute Hellinger distance between probability distributions.
 *
 * H(P, Q) = (1/sqrt(2)) * sqrt(sum_i (sqrt(p_i) - sqrt(q_i))^2)
 *
 * @param p First probability distribution
 * @param q Second probability distribution
 * @param dim Dimension along which to compute distance
 * @param input_type Type of input: "probability", "log_probability", or "logits"
 * @param reduction Reduction to apply: "none", "mean", or "sum"
 * @param pairwise If true, compute all-pairs distance matrix
 * @return Hellinger distance
 */
inline at::Tensor hellinger_distance(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise
) {
    // Preprocess inputs to probabilities
    at::Tensor p_prob = preprocess_input(p, dim, input_type);
    at::Tensor q_prob = preprocess_input(q, dim, input_type);

    if (pairwise) {
        // Pairwise mode: compute H(p_i, q_j) for all pairs
        TORCH_CHECK(
            p_prob.dim() == 2 && q_prob.dim() == 2,
            "hellinger_distance: pairwise mode requires 2D tensors, got ",
            p_prob.dim(), "D and ", q_prob.dim(), "D"
        );
        TORCH_CHECK(
            p_prob.size(1) == q_prob.size(1),
            "hellinger_distance: feature dimensions must match, got ",
            p_prob.size(1), " and ", q_prob.size(1)
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
            "hellinger_distance_pairwise_cpu",
            [&]() {
                const scalar_t* p_ptr = p_contig.data_ptr<scalar_t>();
                const scalar_t* q_ptr = q_contig.data_ptr<scalar_t>();
                scalar_t* out_ptr = output.data_ptr<scalar_t>();

                at::parallel_for(0, m * n, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        int64_t i = idx / n;
                        int64_t j = idx % n;
                        out_ptr[idx] = torchscience::kernel::distance::hellinger_distance_kernel<scalar_t>(
                            p_ptr + i * d,
                            q_ptr + j * d,
                            d
                        );
                    }
                });
            }
        );

        return apply_reduction(output, reduction);
    } else {
        // Standard mode: element-wise distance
        TORCH_CHECK(
            p_prob.sizes() == q_prob.sizes(),
            "hellinger_distance: tensor shapes must match, got ",
            p_prob.sizes(), " and ", q_prob.sizes()
        );

        // Normalize dim to positive index
        int64_t ndim = p_prob.dim();
        if (dim < 0) {
            dim = ndim + dim;
        }
        TORCH_CHECK(
            dim >= 0 && dim < ndim,
            "hellinger_distance: dim out of range, got ", dim
        );

        // Move target dimension to last position for contiguous access
        at::Tensor p_t = p_prob.transpose(dim, -1).contiguous();
        at::Tensor q_t = q_prob.transpose(dim, -1).contiguous();

        int64_t feature_size = p_t.size(-1);
        int64_t batch_size = p_t.numel() / feature_size;

        at::Tensor output = at::empty({batch_size}, p_prob.options());

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            p_prob.scalar_type(),
            "hellinger_distance_cpu",
            [&]() {
                const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
                const scalar_t* q_ptr = q_t.data_ptr<scalar_t>();
                scalar_t* out_ptr = output.data_ptr<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        out_ptr[idx] = torchscience::kernel::distance::hellinger_distance_kernel<scalar_t>(
                            p_ptr + idx * feature_size,
                            q_ptr + idx * feature_size,
                            feature_size
                        );
                    }
                });
            }
        );

        // Reshape output to match input shape (without the reduced dimension)
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

        return apply_reduction(output, reduction);
    }
}

/**
 * Compute gradients for Hellinger distance.
 *
 * @param grad_output Upstream gradient
 * @param p First probability distribution
 * @param q Second probability distribution
 * @param dim Dimension along which distance was computed
 * @param input_type Type of input: "probability", "log_probability", or "logits"
 * @param reduction Reduction that was applied
 * @param pairwise Whether pairwise mode was used
 * @return Tuple of gradients (grad_p, grad_q)
 */
inline std::tuple<at::Tensor, at::Tensor> hellinger_distance_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise
) {
    // Preprocess inputs to probabilities
    at::Tensor p_prob = preprocess_input(p, dim, input_type);
    at::Tensor q_prob = preprocess_input(q, dim, input_type);

    at::Tensor grad_p = at::zeros_like(p_prob);
    at::Tensor grad_q = at::zeros_like(q_prob);

    if (pairwise) {
        int64_t m = p_prob.size(0);
        int64_t n = q_prob.size(0);
        int64_t d = p_prob.size(1);

        at::Tensor p_contig = p_prob.contiguous();
        at::Tensor q_contig = q_prob.contiguous();
        at::Tensor grad_contig = grad_output.contiguous();

        // Compute reduction scale factor
        double scale = 1.0;
        if (reduction == "mean") {
            scale = 1.0 / static_cast<double>(m * n);
        } else if (reduction == "sum") {
            scale = 1.0;
        }

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            p_prob.scalar_type(),
            "hellinger_distance_backward_pairwise_cpu",
            [&]() {
                const scalar_t* p_ptr = p_contig.data_ptr<scalar_t>();
                const scalar_t* q_ptr = q_contig.data_ptr<scalar_t>();
                const scalar_t* grad_ptr = grad_contig.data_ptr<scalar_t>();
                scalar_t* grad_p_ptr = grad_p.data_ptr<scalar_t>();
                scalar_t* grad_q_ptr = grad_q.data_ptr<scalar_t>();
                scalar_t scale_t = static_cast<scalar_t>(scale);

                std::vector<scalar_t> temp_grad_p(d);
                std::vector<scalar_t> temp_grad_q(d);

                // Note: Not parallelized due to accumulation into grad_p and grad_q
                for (int64_t i = 0; i < m; ++i) {
                    for (int64_t j = 0; j < n; ++j) {
                        scalar_t grad_val;
                        if (reduction == "none") {
                            grad_val = grad_ptr[i * n + j];
                        } else {
                            grad_val = grad_ptr[0] * scale_t;
                        }

                        // Compute Hellinger distance for this pair
                        scalar_t hellinger_val = torchscience::kernel::distance::hellinger_distance_kernel<scalar_t>(
                            p_ptr + i * d,
                            q_ptr + j * d,
                            d
                        );

                        torchscience::kernel::distance::hellinger_distance_backward_kernel<scalar_t>(
                            grad_val,
                            p_ptr + i * d,
                            q_ptr + j * d,
                            d,
                            hellinger_val,
                            temp_grad_p.data(),
                            temp_grad_q.data()
                        );

                        for (int64_t k = 0; k < d; ++k) {
                            grad_p_ptr[i * d + k] += temp_grad_p[k];
                            grad_q_ptr[j * d + k] += temp_grad_q[k];
                        }
                    }
                }
            }
        );
    } else {
        // Normalize dim
        int64_t ndim = p_prob.dim();
        if (dim < 0) {
            dim = ndim + dim;
        }

        at::Tensor p_t = p_prob.transpose(dim, -1).contiguous();
        at::Tensor q_t = q_prob.transpose(dim, -1).contiguous();

        int64_t feature_size = p_t.size(-1);
        int64_t batch_size = p_t.numel() / feature_size;

        // Create transposed gradient tensors
        at::Tensor grad_p_t = at::zeros_like(p_t);
        at::Tensor grad_q_t = at::zeros_like(q_t);

        // Compute reduction scale factor
        double scale = 1.0;
        if (reduction == "mean") {
            scale = 1.0 / static_cast<double>(batch_size);
        } else if (reduction == "sum") {
            scale = 1.0;
        }

        at::Tensor grad_flat = grad_output.contiguous().view({-1});

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            p_prob.scalar_type(),
            "hellinger_distance_backward_cpu",
            [&]() {
                const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
                const scalar_t* q_ptr = q_t.data_ptr<scalar_t>();
                const scalar_t* grad_ptr = grad_flat.data_ptr<scalar_t>();
                scalar_t* grad_p_ptr = grad_p_t.data_ptr<scalar_t>();
                scalar_t* grad_q_ptr = grad_q_t.data_ptr<scalar_t>();
                scalar_t scale_t = static_cast<scalar_t>(scale);

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        scalar_t grad_val;
                        if (reduction == "none") {
                            grad_val = grad_ptr[idx];
                        } else {
                            grad_val = grad_ptr[0] * scale_t;
                        }

                        // Compute Hellinger distance for this sample
                        scalar_t hellinger_val = torchscience::kernel::distance::hellinger_distance_kernel<scalar_t>(
                            p_ptr + idx * feature_size,
                            q_ptr + idx * feature_size,
                            feature_size
                        );

                        torchscience::kernel::distance::hellinger_distance_backward_kernel<scalar_t>(
                            grad_val,
                            p_ptr + idx * feature_size,
                            q_ptr + idx * feature_size,
                            feature_size,
                            hellinger_val,
                            grad_p_ptr + idx * feature_size,
                            grad_q_ptr + idx * feature_size
                        );
                    }
                });
            }
        );

        // Transpose back to original shape
        grad_p = grad_p_t.transpose(dim, -1).contiguous();
        grad_q = grad_q_t.transpose(dim, -1).contiguous();
    }

    return std::make_tuple(grad_p, grad_q);
}

}  // namespace torchscience::cpu::distance

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("hellinger_distance", &torchscience::cpu::distance::hellinger_distance);
    m.impl("hellinger_distance_backward", &torchscience::cpu::distance::hellinger_distance_backward);
}
