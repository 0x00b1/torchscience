#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "../../kernel/information_theory/coinformation.h"
#include "../../kernel/information_theory/coinformation_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

inline at::Tensor ci_preprocess_input(
    const at::Tensor& input,
    const std::string& input_type
) {
    if (input_type == "probability") {
        return input;
    } else if (input_type == "log_probability") {
        return input.exp();
    } else if (input_type == "logits") {
        // Normalize over all dimensions to get valid joint distribution
        return at::softmax(input.flatten(), 0).view(input.sizes());
    } else {
        TORCH_CHECK(false, "Invalid input_type: ", input_type);
    }
}

inline at::Tensor ci_apply_reduction(
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
        TORCH_CHECK(false, "Invalid reduction: ", reduction);
    }
}

inline double ci_get_log_base_scale(c10::optional<double> base) {
    if (!base.has_value()) {
        return 1.0;
    }
    double b = base.value();
    TORCH_CHECK(b > 0 && b != 1, "base must be positive and not equal to 1");
    return 1.0 / std::log(b);
}

}  // anonymous namespace

inline at::Tensor coinformation(
    const at::Tensor& joint,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    TORCH_CHECK(joint.dim() >= 2, "joint must have at least 2 dimensions");
    TORCH_CHECK(joint.dim() <= 10, "joint must have at most 10 dimensions (2^10 subsets)");

    at::Tensor joint_prob = ci_preprocess_input(joint, input_type);
    double log_base_scale = ci_get_log_base_scale(base);

    int64_t ndims = joint_prob.dim();

    // Get dimension sizes
    std::vector<int64_t> sizes(ndims);
    int64_t total_elements = 1;
    for (int64_t d = 0; d < ndims; ++d) {
        sizes[d] = joint_prob.size(d);
        total_elements *= sizes[d];
    }

    // Number of non-empty subsets = 2^n - 1
    int64_t num_subsets = (1LL << ndims) - 1;

    // Make contiguous
    at::Tensor joint_contig = joint_prob.contiguous();

    // Allocate output (scalar)
    at::Tensor output = at::empty({}, joint_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "coinformation_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_contig.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t scale = static_cast<scalar_t>(log_base_scale);
            scalar_t eps = static_cast<scalar_t>(1e-10);

            // Compute strides for index calculation
            std::vector<int64_t> strides(ndims);
            strides[ndims - 1] = 1;
            for (int64_t d = ndims - 2; d >= 0; --d) {
                strides[d] = strides[d + 1] * sizes[d + 1];
            }

            // Generate all non-empty subsets and compute their marginals and entropies
            std::vector<int64_t> subset_masks(num_subsets);
            std::vector<scalar_t> subset_entropies(num_subsets);

            for (int64_t s = 0; s < num_subsets; ++s) {
                int64_t mask = s + 1;  // Subsets from 1 to 2^n - 1
                subset_masks[s] = mask;

                // Compute size of this marginal
                int64_t marginal_size = 1;
                for (int64_t d = 0; d < ndims; ++d) {
                    if (mask & (1LL << d)) {
                        marginal_size *= sizes[d];
                    }
                }

                // Compute strides for mapping to marginal
                std::vector<int64_t> marginal_strides(ndims, 0);
                int64_t stride = 1;
                for (int64_t d = ndims - 1; d >= 0; --d) {
                    if (mask & (1LL << d)) {
                        marginal_strides[d] = stride;
                        stride *= sizes[d];
                    }
                }

                // Compute marginal by summing over dimensions not in mask
                std::vector<scalar_t> marginal(marginal_size, scalar_t(0));

                for (int64_t i = 0; i < total_elements; ++i) {
                    scalar_t p = joint_ptr[i];

                    // Compute marginal index
                    int64_t remaining = i;
                    int64_t marginal_idx = 0;
                    for (int64_t d = 0; d < ndims; ++d) {
                        int64_t idx_d = remaining / strides[d];
                        remaining = remaining % strides[d];
                        marginal_idx += idx_d * marginal_strides[d];
                    }

                    marginal[marginal_idx] += p;
                }

                // Compute entropy of this marginal
                scalar_t entropy = scalar_t(0);
                for (int64_t i = 0; i < marginal_size; ++i) {
                    scalar_t p = marginal[i];
                    if (p > eps) {
                        entropy -= p * std::log(p);
                    }
                }

                subset_entropies[s] = entropy;
            }

            // Call kernel
            out_ptr[0] = torchscience::kernel::information_theory::coinformation_kernel<scalar_t>(
                joint_ptr,
                sizes.data(),
                ndims,
                num_subsets,
                subset_masks.data(),
                subset_entropies.data(),
                scale
            );
        }
    );

    return ci_apply_reduction(output, reduction);
}

inline at::Tensor coinformation_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    at::Tensor joint_prob = ci_preprocess_input(joint, input_type);
    double log_base_scale = ci_get_log_base_scale(base);

    int64_t ndims = joint_prob.dim();

    // Get dimension sizes
    std::vector<int64_t> sizes(ndims);
    int64_t total_elements = 1;
    for (int64_t d = 0; d < ndims; ++d) {
        sizes[d] = joint_prob.size(d);
        total_elements *= sizes[d];
    }

    // Number of non-empty subsets = 2^n - 1
    int64_t num_subsets = (1LL << ndims) - 1;

    at::Tensor joint_contig = joint_prob.contiguous();
    at::Tensor grad_joint = at::zeros_like(joint_contig);

    // Handle reduction scaling
    double red_scale = 1.0;
    if (reduction == "mean") {
        red_scale = 1.0;  // No batch dimensions for now
    }

    at::Tensor grad_flat = grad_output.contiguous().view({-1});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "coinformation_backward_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_contig.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_flat.data_ptr<scalar_t>();
            scalar_t* grad_joint_ptr = grad_joint.data_ptr<scalar_t>();
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);

            // Compute strides for index calculation
            std::vector<int64_t> strides(ndims);
            strides[ndims - 1] = 1;
            for (int64_t d = ndims - 2; d >= 0; --d) {
                strides[d] = strides[d + 1] * sizes[d + 1];
            }

            // Generate all non-empty subsets and compute their marginals
            std::vector<int64_t> subset_masks(num_subsets);
            std::vector<int64_t> subset_sizes_vec(num_subsets);
            std::vector<int64_t> subset_offsets(num_subsets);

            // Compute total storage needed for all marginals
            int64_t total_marginal_storage = 0;
            for (int64_t s = 0; s < num_subsets; ++s) {
                int64_t mask = s + 1;
                subset_masks[s] = mask;
                subset_offsets[s] = total_marginal_storage;

                int64_t marginal_size = 1;
                for (int64_t d = 0; d < ndims; ++d) {
                    if (mask & (1LL << d)) {
                        marginal_size *= sizes[d];
                    }
                }
                subset_sizes_vec[s] = marginal_size;
                total_marginal_storage += marginal_size;
            }

            // Allocate storage for all marginals
            std::vector<scalar_t> all_marginals(total_marginal_storage, scalar_t(0));

            // Compute all marginals
            for (int64_t s = 0; s < num_subsets; ++s) {
                int64_t mask = subset_masks[s];
                int64_t marginal_size = subset_sizes_vec[s];
                int64_t offset = subset_offsets[s];

                // Compute strides for mapping to marginal
                std::vector<int64_t> marginal_strides(ndims, 0);
                int64_t stride = 1;
                for (int64_t d = ndims - 1; d >= 0; --d) {
                    if (mask & (1LL << d)) {
                        marginal_strides[d] = stride;
                        stride *= sizes[d];
                    }
                }

                // Compute marginal by summing over dimensions not in mask
                for (int64_t i = 0; i < total_elements; ++i) {
                    scalar_t p = joint_ptr[i];

                    // Compute marginal index
                    int64_t remaining = i;
                    int64_t marginal_idx = 0;
                    for (int64_t d = 0; d < ndims; ++d) {
                        int64_t idx_d = remaining / strides[d];
                        remaining = remaining % strides[d];
                        marginal_idx += idx_d * marginal_strides[d];
                    }

                    all_marginals[offset + marginal_idx] += p;
                }
            }

            scalar_t grad_val = grad_ptr[0] * static_cast<scalar_t>(red_scale);

            torchscience::kernel::information_theory::coinformation_backward_kernel<scalar_t>(
                grad_val,
                joint_ptr,
                sizes.data(),
                ndims,
                num_subsets,
                subset_masks.data(),
                all_marginals.data(),
                subset_offsets.data(),
                subset_sizes_vec.data(),
                log_scale,
                grad_joint_ptr
            );
        }
    );

    return grad_joint;
}

}  // namespace torchscience::cpu::information_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("coinformation", &torchscience::cpu::information_theory::coinformation);
    m.impl("coinformation_backward", &torchscience::cpu::information_theory::coinformation_backward);
}
