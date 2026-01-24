#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "../../kernel/information_theory/active_information_storage.h"
#include "../../kernel/information_theory/active_information_storage_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

inline at::Tensor ais_preprocess_input(
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
            "active_information_storage: input_type must be 'probability' or 'log_probability', got '",
            input_type, "'"
        );
    }
}

inline at::Tensor ais_apply_reduction(
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
            "active_information_storage: reduction must be 'none', 'mean', or 'sum', got '",
            reduction, "'"
        );
    }
}

inline double ais_get_log_base_scale(c10::optional<double> base) {
    if (!base.has_value()) {
        return 1.0;
    }
    double b = base.value();
    TORCH_CHECK(b > 0 && b != 1, "active_information_storage: base must be positive and not equal to 1");
    return 1.0 / std::log(b);
}

}  // anonymous namespace

inline at::Tensor active_information_storage(
    const at::Tensor& joint,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    TORCH_CHECK(joint.dim() >= 2, "joint must have at least 2 dimensions");

    int64_t ndim = joint.dim();

    at::Tensor joint_prob = ais_preprocess_input(joint, input_type);
    double log_base_scale = ais_get_log_base_scale(base);

    // The last 2 dimensions are (x_t, x_{t-1})
    // All preceding dimensions are batch dimensions
    int64_t size_curr = joint_prob.size(-2);
    int64_t size_prev = joint_prob.size(-1);

    // Compute batch shape
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < ndim - 2; ++i) {
        batch_shape.push_back(joint_prob.size(i));
    }

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }

    at::Tensor joint_t = joint_prob.contiguous().view({batch_size, size_curr, size_prev});

    at::Tensor output = at::empty({batch_size}, joint_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "active_information_storage_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t scale = static_cast<scalar_t>(log_base_scale);

            int64_t joint_stride = size_curr * size_prev;

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                // Allocate marginals per thread
                std::vector<scalar_t> p_curr(size_curr);
                std::vector<scalar_t> p_prev(size_prev);

                for (int64_t idx = begin; idx < end; ++idx) {
                    const scalar_t* batch_joint = joint_ptr + idx * joint_stride;

                    // Compute marginals
                    std::fill(p_curr.begin(), p_curr.end(), scalar_t(0));
                    std::fill(p_prev.begin(), p_prev.end(), scalar_t(0));

                    for (int64_t i = 0; i < size_curr; ++i) {
                        for (int64_t j = 0; j < size_prev; ++j) {
                            scalar_t p_xy = batch_joint[i * size_prev + j];
                            p_curr[i] += p_xy;
                            p_prev[j] += p_xy;
                        }
                    }

                    out_ptr[idx] = torchscience::kernel::information_theory::active_information_storage_kernel<scalar_t>(
                        batch_joint,
                        p_curr.data(),
                        p_prev.data(),
                        size_curr,
                        size_prev,
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

    return ais_apply_reduction(output, reduction);
}

inline at::Tensor active_information_storage_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    int64_t ndim = joint.dim();

    at::Tensor joint_prob = ais_preprocess_input(joint, input_type);
    double log_base_scale = ais_get_log_base_scale(base);

    int64_t size_curr = joint_prob.size(-2);
    int64_t size_prev = joint_prob.size(-1);

    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < ndim - 2; ++i) {
        batch_shape.push_back(joint_prob.size(i));
    }

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }

    at::Tensor joint_t = joint_prob.contiguous().view({batch_size, size_curr, size_prev});

    at::Tensor grad_joint_t = at::zeros_like(joint_t);

    double scale = 1.0;
    if (reduction == "mean") {
        scale = 1.0 / static_cast<double>(batch_size);
    }

    at::Tensor grad_flat = grad_output.contiguous().view({-1});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "active_information_storage_backward_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_flat.data_ptr<scalar_t>();
            scalar_t* grad_joint_ptr = grad_joint_t.data_ptr<scalar_t>();
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);
            scalar_t red_scale = static_cast<scalar_t>(scale);

            int64_t joint_stride = size_curr * size_prev;

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                std::vector<scalar_t> p_curr(size_curr);
                std::vector<scalar_t> p_prev(size_prev);

                for (int64_t idx = begin; idx < end; ++idx) {
                    const scalar_t* batch_joint = joint_ptr + idx * joint_stride;
                    scalar_t* batch_grad = grad_joint_ptr + idx * joint_stride;

                    // Compute marginals
                    std::fill(p_curr.begin(), p_curr.end(), scalar_t(0));
                    std::fill(p_prev.begin(), p_prev.end(), scalar_t(0));

                    for (int64_t i = 0; i < size_curr; ++i) {
                        for (int64_t j = 0; j < size_prev; ++j) {
                            scalar_t p_xy = batch_joint[i * size_prev + j];
                            p_curr[i] += p_xy;
                            p_prev[j] += p_xy;
                        }
                    }

                    scalar_t grad_val = (reduction == "none") ? grad_ptr[idx] : grad_ptr[0] * red_scale;

                    torchscience::kernel::information_theory::active_information_storage_backward_kernel<scalar_t>(
                        grad_val,
                        batch_joint,
                        p_curr.data(),
                        p_prev.data(),
                        size_curr,
                        size_prev,
                        log_scale,
                        batch_grad
                    );
                }
            });
        }
    );

    // Reshape back to original shape
    std::vector<int64_t> output_shape = batch_shape;
    output_shape.push_back(size_curr);
    output_shape.push_back(size_prev);

    return grad_joint_t.view(output_shape);
}

}  // namespace torchscience::cpu::information_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("active_information_storage", &torchscience::cpu::information_theory::active_information_storage);
    m.impl("active_information_storage_backward", &torchscience::cpu::information_theory::active_information_storage_backward);
}
