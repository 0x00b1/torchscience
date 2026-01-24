#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "../../kernel/information_theory/total_correlation.h"
#include "../../kernel/information_theory/total_correlation_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

inline at::Tensor tc_preprocess_input(
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

inline at::Tensor tc_apply_reduction(
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

inline double tc_get_log_base_scale(c10::optional<double> base) {
    if (!base.has_value()) {
        return 1.0;
    }
    double b = base.value();
    TORCH_CHECK(b > 0 && b != 1, "base must be positive and not equal to 1");
    return 1.0 / std::log(b);
}

}  // anonymous namespace

inline at::Tensor total_correlation(
    const at::Tensor& joint,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    TORCH_CHECK(joint.dim() >= 2, "joint must have at least 2 dimensions");

    at::Tensor joint_prob = tc_preprocess_input(joint, input_type);
    double log_base_scale = tc_get_log_base_scale(base);

    int64_t ndims = joint_prob.dim();

    // Get dimension sizes
    std::vector<int64_t> sizes(ndims);
    int64_t total_elements = 1;
    for (int64_t d = 0; d < ndims; ++d) {
        sizes[d] = joint_prob.size(d);
        total_elements *= sizes[d];
    }

    // Make contiguous
    at::Tensor joint_contig = joint_prob.contiguous();

    // Allocate output (scalar for now - no batch support for simplicity)
    at::Tensor output = at::empty({}, joint_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "total_correlation_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_contig.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t scale = static_cast<scalar_t>(log_base_scale);

            // Compute marginals for each dimension
            // Total marginal storage: sum of all dimension sizes
            int64_t marginal_total_size = 0;
            for (int64_t d = 0; d < ndims; ++d) {
                marginal_total_size += sizes[d];
            }

            std::vector<scalar_t> marginals(marginal_total_size, scalar_t(0));

            // Compute strides for index calculation
            std::vector<int64_t> strides(ndims);
            strides[ndims - 1] = 1;
            for (int64_t d = ndims - 2; d >= 0; --d) {
                strides[d] = strides[d + 1] * sizes[d + 1];
            }

            // Compute marginals by summing over other dimensions
            for (int64_t i = 0; i < total_elements; ++i) {
                scalar_t p = joint_ptr[i];

                // Compute multi-index and accumulate to marginals
                int64_t remaining = i;
                int64_t marginal_offset = 0;

                for (int64_t d = 0; d < ndims; ++d) {
                    int64_t idx_d = remaining / strides[d];
                    remaining = remaining % strides[d];
                    marginals[marginal_offset + idx_d] += p;
                    marginal_offset += sizes[d];
                }
            }

            // Call kernel
            out_ptr[0] = torchscience::kernel::information_theory::total_correlation_kernel<scalar_t>(
                joint_ptr,
                sizes.data(),
                ndims,
                marginals.data(),
                scale
            );
        }
    );

    return tc_apply_reduction(output, reduction);
}

inline at::Tensor total_correlation_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    at::Tensor joint_prob = tc_preprocess_input(joint, input_type);
    double log_base_scale = tc_get_log_base_scale(base);

    int64_t ndims = joint_prob.dim();

    // Get dimension sizes
    std::vector<int64_t> sizes(ndims);
    int64_t total_elements = 1;
    for (int64_t d = 0; d < ndims; ++d) {
        sizes[d] = joint_prob.size(d);
        total_elements *= sizes[d];
    }

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
        "total_correlation_backward_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_contig.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_flat.data_ptr<scalar_t>();
            scalar_t* grad_joint_ptr = grad_joint.data_ptr<scalar_t>();
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);

            // Compute marginals for each dimension
            int64_t marginal_total_size = 0;
            for (int64_t d = 0; d < ndims; ++d) {
                marginal_total_size += sizes[d];
            }

            std::vector<scalar_t> marginals(marginal_total_size, scalar_t(0));

            // Compute strides
            std::vector<int64_t> strides(ndims);
            strides[ndims - 1] = 1;
            for (int64_t d = ndims - 2; d >= 0; --d) {
                strides[d] = strides[d + 1] * sizes[d + 1];
            }

            // Compute marginals
            for (int64_t i = 0; i < total_elements; ++i) {
                scalar_t p = joint_ptr[i];
                int64_t remaining = i;
                int64_t marginal_offset = 0;

                for (int64_t d = 0; d < ndims; ++d) {
                    int64_t idx_d = remaining / strides[d];
                    remaining = remaining % strides[d];
                    marginals[marginal_offset + idx_d] += p;
                    marginal_offset += sizes[d];
                }
            }

            scalar_t grad_val = grad_ptr[0] * static_cast<scalar_t>(red_scale);

            torchscience::kernel::information_theory::total_correlation_backward_kernel<scalar_t>(
                grad_val,
                joint_ptr,
                sizes.data(),
                ndims,
                marginals.data(),
                log_scale,
                grad_joint_ptr
            );
        }
    );

    return grad_joint;
}

}  // namespace torchscience::cpu::information_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("total_correlation", &torchscience::cpu::information_theory::total_correlation);
    m.impl("total_correlation_backward", &torchscience::cpu::information_theory::total_correlation_backward);
}
