#pragma once

#include <cmath>
#include <string>
#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "../../kernel/information_theory/renyi_entropy.h"
#include "../../kernel/information_theory/renyi_entropy_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

inline at::Tensor renyi_preprocess_input(
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
            "renyi_entropy: input_type must be 'probability', 'log_probability', or 'logits', got '",
            input_type, "'"
        );
    }
}

inline at::Tensor renyi_apply_reduction(
    const at::Tensor& output,
    const std::string& reduction,
    int64_t batch_size
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
            "renyi_entropy: reduction must be 'none', 'mean', or 'sum', got '",
            reduction, "'"
        );
    }
}

inline double renyi_get_log_base_scale(c10::optional<double> base) {
    if (!base.has_value()) {
        return 1.0;  // Natural logarithm (nats)
    }
    double b = base.value();
    TORCH_CHECK(b > 0 && b != 1, "renyi_entropy: base must be positive and not equal to 1");
    return 1.0 / std::log(b);
}

}  // anonymous namespace

inline at::Tensor renyi_entropy(
    const at::Tensor& p,
    double alpha,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    TORCH_CHECK(alpha >= 0, "renyi_entropy: alpha must be >= 0, got ", alpha);
    TORCH_CHECK(std::abs(alpha - 1.0) > 1e-6, "renyi_entropy: alpha cannot be 1 (use shannon_entropy)");

    at::Tensor p_prob = renyi_preprocess_input(p, dim, input_type);
    double log_base_scale = renyi_get_log_base_scale(base);

    int64_t ndim = p_prob.dim();
    if (dim < 0) {
        dim = ndim + dim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "renyi_entropy: dim out of range, got ", dim);

    at::Tensor p_t = p_prob.transpose(dim, -1).contiguous();
    int64_t feature_size = p_t.size(-1);
    int64_t batch_size = p_t.numel() / feature_size;

    at::Tensor output = at::empty({batch_size}, p_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p_prob.scalar_type(),
        "renyi_entropy_cpu",
        [&]() {
            const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t scale = static_cast<scalar_t>(log_base_scale);
            scalar_t alpha_t = static_cast<scalar_t>(alpha);

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t idx = begin; idx < end; ++idx) {
                    out_ptr[idx] = torchscience::kernel::information_theory::renyi_entropy_kernel<scalar_t>(
                        p_ptr + idx * feature_size,
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

    return renyi_apply_reduction(output, reduction, batch_size);
}

inline at::Tensor renyi_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    double alpha,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    at::Tensor p_prob = renyi_preprocess_input(p, dim, input_type);
    double log_base_scale = renyi_get_log_base_scale(base);

    int64_t ndim = p_prob.dim();
    if (dim < 0) {
        dim = ndim + dim;
    }

    at::Tensor p_t = p_prob.transpose(dim, -1).contiguous();
    int64_t feature_size = p_t.size(-1);
    int64_t batch_size = p_t.numel() / feature_size;

    at::Tensor grad_p_t = at::zeros_like(p_t);

    double scale = 1.0;
    if (reduction == "mean") {
        scale = 1.0 / static_cast<double>(batch_size);
    }

    at::Tensor grad_flat = grad_output.contiguous().view({-1});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p_prob.scalar_type(),
        "renyi_entropy_backward_cpu",
        [&]() {
            const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_flat.data_ptr<scalar_t>();
            scalar_t* grad_p_ptr = grad_p_t.data_ptr<scalar_t>();
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);
            scalar_t alpha_t = static_cast<scalar_t>(alpha);
            scalar_t reduction_scale = static_cast<scalar_t>(scale);

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t idx = begin; idx < end; ++idx) {
                    scalar_t grad_val;
                    if (reduction == "none") {
                        grad_val = grad_ptr[idx];
                    } else {
                        grad_val = grad_ptr[0] * reduction_scale;
                    }

                    torchscience::kernel::information_theory::renyi_entropy_backward_kernel<scalar_t>(
                        grad_val,
                        p_ptr + idx * feature_size,
                        feature_size,
                        alpha_t,
                        log_scale,
                        grad_p_ptr + idx * feature_size
                    );
                }
            });
        }
    );

    return grad_p_t.transpose(dim, -1).contiguous();
}

inline std::tuple<at::Tensor, at::Tensor> renyi_entropy_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& grad_output,
    const at::Tensor& p,
    double alpha,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    at::Tensor p_prob = renyi_preprocess_input(p, dim, input_type);
    double log_base_scale = renyi_get_log_base_scale(base);

    int64_t ndim = p_prob.dim();
    if (dim < 0) {
        dim = ndim + dim;
    }

    at::Tensor p_t = p_prob.transpose(dim, -1).contiguous();
    int64_t feature_size = p_t.size(-1);
    int64_t batch_size = p_t.numel() / feature_size;

    at::Tensor gg_p_t = gg_p.defined() ? gg_p.transpose(dim, -1).contiguous() : at::Tensor();
    at::Tensor grad_p_t = at::zeros_like(p_t);
    at::Tensor grad_grad_output = at::zeros({1}, p_prob.options());

    double scale = 1.0;
    if (reduction == "mean") {
        scale = 1.0 / static_cast<double>(batch_size);
    }

    at::Tensor grad_flat = grad_output.contiguous().view({-1});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p_prob.scalar_type(),
        "renyi_entropy_backward_backward_cpu",
        [&]() {
            const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
            const scalar_t* gg_p_ptr = gg_p_t.defined() ? gg_p_t.data_ptr<scalar_t>() : nullptr;
            const scalar_t* grad_ptr = grad_flat.data_ptr<scalar_t>();
            scalar_t* grad_p_ptr = grad_p_t.data_ptr<scalar_t>();
            scalar_t* grad_grad_out_ptr = grad_grad_output.data_ptr<scalar_t>();
            scalar_t alpha_t = static_cast<scalar_t>(alpha);
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);
            scalar_t reduction_scale = static_cast<scalar_t>(scale);

            scalar_t total_grad_grad_out = scalar_t(0);

            for (int64_t idx = 0; idx < batch_size; ++idx) {
                scalar_t grad_val;
                if (reduction == "none") {
                    grad_val = grad_ptr[idx];
                } else {
                    grad_val = grad_ptr[0] * reduction_scale;
                }

                scalar_t local_grad_grad_out = scalar_t(0);

                torchscience::kernel::information_theory::renyi_entropy_backward_backward_kernel<scalar_t>(
                    gg_p_ptr ? gg_p_ptr + idx * feature_size : nullptr,
                    grad_val,
                    p_ptr + idx * feature_size,
                    feature_size,
                    alpha_t,
                    log_scale,
                    local_grad_grad_out,
                    grad_p_ptr + idx * feature_size
                );

                total_grad_grad_out += local_grad_grad_out;
            }

            grad_grad_out_ptr[0] = total_grad_grad_out;
        }
    );

    at::Tensor grad_p = grad_p_t.transpose(dim, -1).contiguous();

    return std::make_tuple(grad_grad_output.squeeze(), grad_p);
}

}  // namespace torchscience::cpu::information_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("renyi_entropy", &torchscience::cpu::information_theory::renyi_entropy);
    m.impl("renyi_entropy_backward", &torchscience::cpu::information_theory::renyi_entropy_backward);
    m.impl("renyi_entropy_backward_backward", &torchscience::cpu::information_theory::renyi_entropy_backward_backward);
}
