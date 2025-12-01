#pragma once

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "../../kernel/information_theory/conditional_entropy.h"
#include "../../kernel/information_theory/conditional_entropy_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

inline at::Tensor conditional_preprocess_input(
    const at::Tensor& input,
    const std::string& input_type
) {
    if (input_type == "probability") {
        return input;
    } else if (input_type == "log_probability") {
        return input.exp();
    } else {
        TORCH_CHECK(
            false,
            "conditional_entropy: input_type must be 'probability' or 'log_probability', got '",
            input_type, "'"
        );
    }
}

inline at::Tensor conditional_apply_reduction(
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
            "conditional_entropy: reduction must be 'none', 'mean', or 'sum', got '",
            reduction, "'"
        );
    }
}

inline double conditional_get_log_base_scale(c10::optional<double> base) {
    if (!base.has_value()) {
        return 1.0;
    }
    double b = base.value();
    TORCH_CHECK(b > 0 && b != 1, "conditional_entropy: base must be positive and not equal to 1");
    return 1.0 / std::log(b);
}

}  // anonymous namespace

inline at::Tensor conditional_entropy(
    const at::Tensor& joint,
    int64_t condition_dim,
    int64_t target_dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    at::Tensor joint_prob = conditional_preprocess_input(joint, input_type);
    double log_base_scale = conditional_get_log_base_scale(base);

    int64_t ndim = joint_prob.dim();
    TORCH_CHECK(ndim >= 2, "conditional_entropy: joint must have at least 2 dimensions");

    // Normalize dims
    if (condition_dim < 0) condition_dim = ndim + condition_dim;
    if (target_dim < 0) target_dim = ndim + target_dim;
    TORCH_CHECK(condition_dim >= 0 && condition_dim < ndim, "conditional_entropy: condition_dim out of range");
    TORCH_CHECK(target_dim >= 0 && target_dim < ndim, "conditional_entropy: target_dim out of range");
    TORCH_CHECK(condition_dim != target_dim, "conditional_entropy: condition_dim and target_dim must be different");

    // Determine which is larger for internal representation
    int64_t dim0 = std::min(condition_dim, target_dim);
    int64_t dim1 = std::max(condition_dim, target_dim);
    int64_t internal_condition_dim = (condition_dim < target_dim) ? 0 : 1;

    // Move the two dims to the end
    std::vector<int64_t> perm;
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim0 && i != dim1) {
            perm.push_back(i);
            batch_shape.push_back(joint_prob.size(i));
        }
    }
    perm.push_back(dim0);
    perm.push_back(dim1);

    at::Tensor joint_t = joint_prob.permute(perm).contiguous();
    int64_t size_dim0 = joint_t.size(-2);
    int64_t size_dim1 = joint_t.size(-1);

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }
    joint_t = joint_t.view({batch_size, size_dim0, size_dim1});

    at::Tensor output = at::empty({batch_size}, joint_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "conditional_entropy_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t scale = static_cast<scalar_t>(log_base_scale);

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t idx = begin; idx < end; ++idx) {
                    out_ptr[idx] = torchscience::kernel::information_theory::conditional_entropy_kernel<scalar_t>(
                        joint_ptr + idx * size_dim0 * size_dim1,
                        size_dim0,
                        size_dim1,
                        internal_condition_dim,
                        scale
                    );
                }
            });
        }
    );

    if (!batch_shape.empty()) {
        output = output.view(batch_shape);
    } else {
        output = output.squeeze();
    }

    return conditional_apply_reduction(output, reduction);
}

inline at::Tensor conditional_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    int64_t condition_dim,
    int64_t target_dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    at::Tensor joint_prob = conditional_preprocess_input(joint, input_type);
    double log_base_scale = conditional_get_log_base_scale(base);

    int64_t ndim = joint_prob.dim();

    if (condition_dim < 0) condition_dim = ndim + condition_dim;
    if (target_dim < 0) target_dim = ndim + target_dim;

    int64_t dim0 = std::min(condition_dim, target_dim);
    int64_t dim1 = std::max(condition_dim, target_dim);
    int64_t internal_condition_dim = (condition_dim < target_dim) ? 0 : 1;

    std::vector<int64_t> perm;
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim0 && i != dim1) {
            perm.push_back(i);
            batch_shape.push_back(joint_prob.size(i));
        }
    }
    perm.push_back(dim0);
    perm.push_back(dim1);

    at::Tensor joint_t = joint_prob.permute(perm).contiguous();
    int64_t size_dim0 = joint_t.size(-2);
    int64_t size_dim1 = joint_t.size(-1);

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }
    joint_t = joint_t.view({batch_size, size_dim0, size_dim1});

    at::Tensor grad_joint_t = at::zeros_like(joint_t);

    double scale = 1.0;
    if (reduction == "mean") {
        scale = 1.0 / static_cast<double>(batch_size);
    }

    at::Tensor grad_flat = grad_output.contiguous().view({-1});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "conditional_entropy_backward_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_flat.data_ptr<scalar_t>();
            scalar_t* grad_joint_ptr = grad_joint_t.data_ptr<scalar_t>();
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);
            scalar_t red_scale = static_cast<scalar_t>(scale);

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t idx = begin; idx < end; ++idx) {
                    scalar_t grad_val = (reduction == "none") ? grad_ptr[idx] : grad_ptr[0] * red_scale;

                    torchscience::kernel::information_theory::conditional_entropy_backward_kernel<scalar_t>(
                        grad_val,
                        joint_ptr + idx * size_dim0 * size_dim1,
                        size_dim0,
                        size_dim1,
                        internal_condition_dim,
                        log_scale,
                        grad_joint_ptr + idx * size_dim0 * size_dim1
                    );
                }
            });
        }
    );

    // Reshape back to permuted shape
    std::vector<int64_t> permuted_shape;
    for (auto idx : perm) {
        permuted_shape.push_back(joint_prob.size(idx));
    }
    grad_joint_t = grad_joint_t.view(permuted_shape);

    // Inverse permutation
    std::vector<int64_t> inv_perm(ndim);
    for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
        inv_perm[perm[i]] = i;
    }

    return grad_joint_t.permute(inv_perm).contiguous();
}

inline std::tuple<at::Tensor, at::Tensor> conditional_entropy_backward_backward(
    const at::Tensor& gg_joint,
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    int64_t condition_dim,
    int64_t target_dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    at::Tensor joint_prob = conditional_preprocess_input(joint, input_type);
    double log_base_scale = conditional_get_log_base_scale(base);

    int64_t ndim = joint_prob.dim();

    if (condition_dim < 0) condition_dim = ndim + condition_dim;
    if (target_dim < 0) target_dim = ndim + target_dim;

    int64_t dim0 = std::min(condition_dim, target_dim);
    int64_t dim1 = std::max(condition_dim, target_dim);
    int64_t internal_condition_dim = (condition_dim < target_dim) ? 0 : 1;

    std::vector<int64_t> perm;
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim0 && i != dim1) {
            perm.push_back(i);
            batch_shape.push_back(joint_prob.size(i));
        }
    }
    perm.push_back(dim0);
    perm.push_back(dim1);

    at::Tensor joint_t = joint_prob.permute(perm).contiguous();
    at::Tensor gg_joint_t = gg_joint.defined() ? gg_joint.permute(perm).contiguous() : at::Tensor();

    int64_t size_dim0 = joint_t.size(-2);
    int64_t size_dim1 = joint_t.size(-1);

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }
    joint_t = joint_t.view({batch_size, size_dim0, size_dim1});
    if (gg_joint_t.defined()) {
        gg_joint_t = gg_joint_t.view({batch_size, size_dim0, size_dim1});
    }

    at::Tensor grad_joint_t = at::zeros_like(joint_t);

    double scale = 1.0;
    if (reduction == "mean") {
        scale = 1.0 / static_cast<double>(batch_size);
    }

    at::Tensor grad_out_flat = grad_output.contiguous().view({-1});

    at::Tensor grad_grad_output;
    if (reduction == "none") {
        grad_grad_output = at::zeros({batch_size}, joint_prob.options());
    } else {
        grad_grad_output = at::zeros({1}, joint_prob.options());
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "conditional_entropy_backward_backward_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            const scalar_t* gg_joint_ptr = gg_joint_t.defined() ? gg_joint_t.data_ptr<scalar_t>() : nullptr;
            const scalar_t* grad_out_ptr = grad_out_flat.data_ptr<scalar_t>();
            scalar_t* grad_grad_out_ptr = grad_grad_output.data_ptr<scalar_t>();
            scalar_t* grad_joint_ptr = grad_joint_t.data_ptr<scalar_t>();
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);
            scalar_t reduction_scale = static_cast<scalar_t>(scale);

            int64_t joint_stride = size_dim0 * size_dim1;

            if (reduction == "none") {
                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        scalar_t grad_val = grad_out_ptr[idx];
                        scalar_t grad_grad_out_val = scalar_t(0);

                        torchscience::kernel::information_theory::conditional_entropy_backward_backward_kernel<scalar_t>(
                            gg_joint_ptr ? gg_joint_ptr + idx * joint_stride : nullptr,
                            grad_val,
                            joint_ptr + idx * joint_stride,
                            size_dim0,
                            size_dim1,
                            internal_condition_dim,
                            log_scale,
                            grad_grad_out_val,
                            grad_joint_ptr + idx * joint_stride
                        );

                        grad_grad_out_ptr[idx] = grad_grad_out_val;
                    }
                });
            } else {
                scalar_t total_grad_grad_out = scalar_t(0);
                for (int64_t idx = 0; idx < batch_size; ++idx) {
                    scalar_t grad_val = grad_out_ptr[0] * reduction_scale;
                    scalar_t grad_grad_out_val = scalar_t(0);

                    torchscience::kernel::information_theory::conditional_entropy_backward_backward_kernel<scalar_t>(
                        gg_joint_ptr ? gg_joint_ptr + idx * joint_stride : nullptr,
                        grad_val,
                        joint_ptr + idx * joint_stride,
                        size_dim0,
                        size_dim1,
                        internal_condition_dim,
                        log_scale,
                        grad_grad_out_val,
                        grad_joint_ptr + idx * joint_stride
                    );

                    total_grad_grad_out += grad_grad_out_val * reduction_scale;
                }
                grad_grad_out_ptr[0] = total_grad_grad_out;
            }
        }
    );

    // Reshape back to permuted shape
    std::vector<int64_t> permuted_shape;
    for (auto idx : perm) {
        permuted_shape.push_back(joint_prob.size(idx));
    }
    grad_joint_t = grad_joint_t.view(permuted_shape);

    // Inverse permutation
    std::vector<int64_t> inv_perm(ndim);
    for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
        inv_perm[perm[i]] = i;
    }

    at::Tensor grad_joint = grad_joint_t.permute(inv_perm).contiguous();

    if (reduction == "none" && !batch_shape.empty()) {
        grad_grad_output = grad_grad_output.view(batch_shape);
    }

    return std::make_tuple(grad_grad_output, grad_joint);
}

}  // namespace torchscience::cpu::information_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("conditional_entropy", &torchscience::cpu::information_theory::conditional_entropy);
    m.impl("conditional_entropy_backward", &torchscience::cpu::information_theory::conditional_entropy_backward);
    m.impl("conditional_entropy_backward_backward", &torchscience::cpu::information_theory::conditional_entropy_backward_backward);
}
