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

#include "../../kernel/information_theory/joint_entropy.h"
#include "../../kernel/information_theory/joint_entropy_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

inline at::Tensor joint_preprocess_input(
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
            "joint_entropy: input_type must be 'probability' or 'log_probability', got '",
            input_type, "'"
        );
    }
}

inline at::Tensor joint_apply_reduction(
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
            "joint_entropy: reduction must be 'none', 'mean', or 'sum', got '",
            reduction, "'"
        );
    }
}

inline double joint_get_log_base_scale(c10::optional<double> base) {
    if (!base.has_value()) {
        return 1.0;
    }
    double b = base.value();
    TORCH_CHECK(b > 0 && b != 1, "joint_entropy: base must be positive and not equal to 1");
    return 1.0 / std::log(b);
}

}  // anonymous namespace

inline at::Tensor joint_entropy(
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    TORCH_CHECK(dims.size() >= 1, "joint_entropy: dims must have at least 1 element");

    at::Tensor joint_prob = joint_preprocess_input(joint, input_type);
    double log_base_scale = joint_get_log_base_scale(base);

    int64_t ndim = joint_prob.dim();

    // Normalize dims
    std::vector<int64_t> norm_dims;
    for (auto d : dims) {
        if (d < 0) d = ndim + d;
        TORCH_CHECK(d >= 0 && d < ndim, "joint_entropy: dim out of range");
        norm_dims.push_back(d);
    }

    // Compute total size of dims to reduce
    int64_t reduce_size = 1;
    for (auto d : norm_dims) {
        reduce_size *= joint_prob.size(d);
    }

    // Move all reduce dims to end and flatten them
    std::vector<int64_t> perm;
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        bool is_reduce_dim = false;
        for (auto d : norm_dims) {
            if (i == d) {
                is_reduce_dim = true;
                break;
            }
        }
        if (!is_reduce_dim) {
            perm.push_back(i);
            batch_shape.push_back(joint_prob.size(i));
        }
    }
    for (auto d : norm_dims) {
        perm.push_back(d);
    }

    at::Tensor joint_t = joint_prob.permute(perm).contiguous();
    // Flatten batch dims and reduce dims
    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }
    joint_t = joint_t.view({batch_size, reduce_size});

    at::Tensor output = at::empty({batch_size}, joint_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "joint_entropy_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t scale = static_cast<scalar_t>(log_base_scale);

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t idx = begin; idx < end; ++idx) {
                    out_ptr[idx] = torchscience::kernel::information_theory::joint_entropy_kernel<scalar_t>(
                        joint_ptr + idx * reduce_size,
                        1,  // treat as 1D
                        reduce_size,
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

    return joint_apply_reduction(output, reduction);
}

inline at::Tensor joint_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    at::Tensor joint_prob = joint_preprocess_input(joint, input_type);
    double log_base_scale = joint_get_log_base_scale(base);

    int64_t ndim = joint_prob.dim();

    std::vector<int64_t> norm_dims;
    for (auto d : dims) {
        if (d < 0) d = ndim + d;
        norm_dims.push_back(d);
    }

    int64_t reduce_size = 1;
    for (auto d : norm_dims) {
        reduce_size *= joint_prob.size(d);
    }

    std::vector<int64_t> perm;
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        bool is_reduce_dim = false;
        for (auto d : norm_dims) {
            if (i == d) { is_reduce_dim = true; break; }
        }
        if (!is_reduce_dim) {
            perm.push_back(i);
            batch_shape.push_back(joint_prob.size(i));
        }
    }
    for (auto d : norm_dims) {
        perm.push_back(d);
    }

    at::Tensor joint_t = joint_prob.permute(perm).contiguous();
    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }
    joint_t = joint_t.view({batch_size, reduce_size});

    at::Tensor grad_joint_t = at::zeros_like(joint_t);

    double scale = 1.0;
    if (reduction == "mean") {
        scale = 1.0 / static_cast<double>(batch_size);
    }

    at::Tensor grad_flat = grad_output.contiguous().view({-1});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "joint_entropy_backward_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_flat.data_ptr<scalar_t>();
            scalar_t* grad_joint_ptr = grad_joint_t.data_ptr<scalar_t>();
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);
            scalar_t red_scale = static_cast<scalar_t>(scale);

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t idx = begin; idx < end; ++idx) {
                    scalar_t grad_val = (reduction == "none") ? grad_ptr[idx] : grad_ptr[0] * red_scale;

                    torchscience::kernel::information_theory::joint_entropy_backward_kernel<scalar_t>(
                        grad_val,
                        joint_ptr + idx * reduce_size,
                        1,
                        reduce_size,
                        log_scale,
                        grad_joint_ptr + idx * reduce_size
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

inline std::tuple<at::Tensor, at::Tensor> joint_entropy_backward_backward(
    const at::Tensor& gg_joint,
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    at::Tensor joint_prob = joint_preprocess_input(joint, input_type);
    double log_base_scale = joint_get_log_base_scale(base);

    int64_t ndim = joint_prob.dim();

    std::vector<int64_t> norm_dims;
    for (auto d : dims) {
        if (d < 0) d = ndim + d;
        norm_dims.push_back(d);
    }

    int64_t reduce_size = 1;
    for (auto d : norm_dims) {
        reduce_size *= joint_prob.size(d);
    }

    std::vector<int64_t> perm;
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        bool is_reduce_dim = false;
        for (auto d : norm_dims) {
            if (i == d) { is_reduce_dim = true; break; }
        }
        if (!is_reduce_dim) {
            perm.push_back(i);
            batch_shape.push_back(joint_prob.size(i));
        }
    }
    for (auto d : norm_dims) {
        perm.push_back(d);
    }

    at::Tensor joint_t = joint_prob.permute(perm).contiguous();
    at::Tensor gg_joint_t = gg_joint.defined() ? gg_joint.permute(perm).contiguous() : at::Tensor();

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }
    joint_t = joint_t.view({batch_size, reduce_size});
    if (gg_joint_t.defined()) {
        gg_joint_t = gg_joint_t.view({batch_size, reduce_size});
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
        "joint_entropy_backward_backward_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            const scalar_t* gg_joint_ptr = gg_joint_t.defined() ? gg_joint_t.data_ptr<scalar_t>() : nullptr;
            const scalar_t* grad_out_ptr = grad_out_flat.data_ptr<scalar_t>();
            scalar_t* grad_grad_out_ptr = grad_grad_output.data_ptr<scalar_t>();
            scalar_t* grad_joint_ptr = grad_joint_t.data_ptr<scalar_t>();
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);
            scalar_t reduction_scale = static_cast<scalar_t>(scale);

            if (reduction == "none") {
                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        scalar_t grad_val = grad_out_ptr[idx];
                        scalar_t grad_grad_out_val = scalar_t(0);

                        torchscience::kernel::information_theory::joint_entropy_backward_backward_kernel<scalar_t>(
                            gg_joint_ptr ? gg_joint_ptr + idx * reduce_size : nullptr,
                            grad_val,
                            joint_ptr + idx * reduce_size,
                            1,
                            reduce_size,
                            log_scale,
                            grad_grad_out_val,
                            grad_joint_ptr + idx * reduce_size
                        );

                        grad_grad_out_ptr[idx] = grad_grad_out_val;
                    }
                });
            } else {
                scalar_t total_grad_grad_out = scalar_t(0);
                for (int64_t idx = 0; idx < batch_size; ++idx) {
                    scalar_t grad_val = grad_out_ptr[0] * reduction_scale;
                    scalar_t grad_grad_out_val = scalar_t(0);

                    torchscience::kernel::information_theory::joint_entropy_backward_backward_kernel<scalar_t>(
                        gg_joint_ptr ? gg_joint_ptr + idx * reduce_size : nullptr,
                        grad_val,
                        joint_ptr + idx * reduce_size,
                        1,
                        reduce_size,
                        log_scale,
                        grad_grad_out_val,
                        grad_joint_ptr + idx * reduce_size
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
    m.impl("joint_entropy", &torchscience::cpu::information_theory::joint_entropy);
    m.impl("joint_entropy_backward", &torchscience::cpu::information_theory::joint_entropy_backward);
    m.impl("joint_entropy_backward_backward", &torchscience::cpu::information_theory::joint_entropy_backward_backward);
}
