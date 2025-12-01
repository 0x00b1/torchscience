#pragma once

#include <cmath>
#include <string>
#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../kernel/information_theory/chi_squared_divergence.h"
#include "../../kernel/information_theory/chi_squared_divergence_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

inline at::Tensor chi_apply_reduction(
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
            "chi_squared_divergence: reduction must be 'none', 'mean', or 'sum', got '",
            reduction, "'"
        );
    }
}

}  // anonymous namespace

inline at::Tensor chi_squared_divergence(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& reduction
) {
    TORCH_CHECK(p.sizes() == q.sizes(), "chi_squared_divergence: p and q must have the same shape");

    int64_t ndim = p.dim();
    if (dim < 0) {
        dim = ndim + dim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "chi_squared_divergence: dim out of range, got ", dim);

    at::Tensor p_t = p.transpose(dim, -1).contiguous();
    at::Tensor q_t = q.transpose(dim, -1).contiguous();
    int64_t feature_size = p_t.size(-1);
    int64_t batch_size = p_t.numel() / feature_size;

    at::Tensor output = at::empty({batch_size}, p.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p.scalar_type(),
        "chi_squared_divergence_cpu",
        [&]() {
            const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
            const scalar_t* q_ptr = q_t.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t idx = begin; idx < end; ++idx) {
                    out_ptr[idx] = torchscience::kernel::information_theory::chi_squared_divergence_kernel<scalar_t>(
                        p_ptr + idx * feature_size,
                        q_ptr + idx * feature_size,
                        feature_size
                    );
                }
            });
        }
    );

    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            output_shape.push_back(p.size(i));
        }
    }
    if (output_shape.empty()) {
        output = output.squeeze();
    } else {
        output = output.view(output_shape);
    }

    return chi_apply_reduction(output, reduction);
}

inline std::tuple<at::Tensor, at::Tensor> chi_squared_divergence_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& reduction
) {
    int64_t ndim = p.dim();
    if (dim < 0) {
        dim = ndim + dim;
    }

    at::Tensor p_t = p.transpose(dim, -1).contiguous();
    at::Tensor q_t = q.transpose(dim, -1).contiguous();
    int64_t feature_size = p_t.size(-1);
    int64_t batch_size = p_t.numel() / feature_size;

    at::Tensor grad_p_t = at::zeros_like(p_t);
    at::Tensor grad_q_t = at::zeros_like(q_t);

    double scale = 1.0;
    if (reduction == "mean") {
        scale = 1.0 / static_cast<double>(batch_size);
    }

    at::Tensor grad_flat = grad_output.contiguous().view({-1});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p.scalar_type(),
        "chi_squared_divergence_backward_cpu",
        [&]() {
            const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
            const scalar_t* q_ptr = q_t.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_flat.data_ptr<scalar_t>();
            scalar_t* grad_p_ptr = grad_p_t.data_ptr<scalar_t>();
            scalar_t* grad_q_ptr = grad_q_t.data_ptr<scalar_t>();
            scalar_t reduction_scale = static_cast<scalar_t>(scale);

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t idx = begin; idx < end; ++idx) {
                    scalar_t grad_val;
                    if (reduction == "none") {
                        grad_val = grad_ptr[idx];
                    } else {
                        grad_val = grad_ptr[0] * reduction_scale;
                    }

                    torchscience::kernel::information_theory::chi_squared_divergence_backward_kernel<scalar_t>(
                        grad_val,
                        p_ptr + idx * feature_size,
                        q_ptr + idx * feature_size,
                        feature_size,
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

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> chi_squared_divergence_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& gg_q,
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& reduction
) {
    int64_t ndim = p.dim();
    if (dim < 0) {
        dim = ndim + dim;
    }

    at::Tensor p_t = p.transpose(dim, -1).contiguous();
    at::Tensor q_t = q.transpose(dim, -1).contiguous();
    at::Tensor gg_p_t = gg_p.defined() ? gg_p.transpose(dim, -1).contiguous() : at::Tensor();
    at::Tensor gg_q_t = gg_q.defined() ? gg_q.transpose(dim, -1).contiguous() : at::Tensor();

    int64_t feature_size = p_t.size(-1);
    int64_t batch_size = p_t.numel() / feature_size;

    at::Tensor grad_p_t = at::zeros_like(p_t);
    at::Tensor grad_q_t = at::zeros_like(q_t);

    double scale = 1.0;
    if (reduction == "mean") {
        scale = 1.0 / static_cast<double>(batch_size);
    }

    at::Tensor grad_out_flat = grad_output.contiguous().view({-1});

    at::Tensor grad_grad_output;
    if (reduction == "none") {
        grad_grad_output = at::zeros({batch_size}, p.options());
    } else {
        grad_grad_output = at::zeros({1}, p.options());
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p.scalar_type(),
        "chi_squared_divergence_backward_backward_cpu",
        [&]() {
            const scalar_t* p_ptr = p_t.data_ptr<scalar_t>();
            const scalar_t* q_ptr = q_t.data_ptr<scalar_t>();
            const scalar_t* gg_p_ptr = gg_p_t.defined() ? gg_p_t.data_ptr<scalar_t>() : nullptr;
            const scalar_t* gg_q_ptr = gg_q_t.defined() ? gg_q_t.data_ptr<scalar_t>() : nullptr;
            const scalar_t* grad_out_ptr = grad_out_flat.data_ptr<scalar_t>();
            scalar_t* grad_grad_out_ptr = grad_grad_output.data_ptr<scalar_t>();
            scalar_t* grad_p_ptr = grad_p_t.data_ptr<scalar_t>();
            scalar_t* grad_q_ptr = grad_q_t.data_ptr<scalar_t>();
            scalar_t reduction_scale = static_cast<scalar_t>(scale);

            if (reduction == "none") {
                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        scalar_t grad_val = grad_out_ptr[idx];
                        scalar_t grad_grad_out_val = scalar_t(0);

                        torchscience::kernel::information_theory::chi_squared_divergence_backward_backward_kernel<scalar_t>(
                            gg_p_ptr ? gg_p_ptr + idx * feature_size : nullptr,
                            gg_q_ptr ? gg_q_ptr + idx * feature_size : nullptr,
                            grad_val,
                            p_ptr + idx * feature_size,
                            q_ptr + idx * feature_size,
                            feature_size,
                            grad_grad_out_val,
                            grad_p_ptr + idx * feature_size,
                            grad_q_ptr + idx * feature_size
                        );

                        grad_grad_out_ptr[idx] = grad_grad_out_val;
                    }
                });
            } else {
                scalar_t total_grad_grad_out = scalar_t(0);
                for (int64_t idx = 0; idx < batch_size; ++idx) {
                    scalar_t grad_val = grad_out_ptr[0] * reduction_scale;
                    scalar_t grad_grad_out_val = scalar_t(0);

                    torchscience::kernel::information_theory::chi_squared_divergence_backward_backward_kernel<scalar_t>(
                        gg_p_ptr ? gg_p_ptr + idx * feature_size : nullptr,
                        gg_q_ptr ? gg_q_ptr + idx * feature_size : nullptr,
                        grad_val,
                        p_ptr + idx * feature_size,
                        q_ptr + idx * feature_size,
                        feature_size,
                        grad_grad_out_val,
                        grad_p_ptr + idx * feature_size,
                        grad_q_ptr + idx * feature_size
                    );

                    total_grad_grad_out += grad_grad_out_val * reduction_scale;
                }
                grad_grad_out_ptr[0] = total_grad_grad_out;
            }
        }
    );

    at::Tensor grad_p = grad_p_t.transpose(dim, -1).contiguous();
    at::Tensor grad_q = grad_q_t.transpose(dim, -1).contiguous();

    if (reduction == "none") {
        std::vector<int64_t> output_shape;
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != dim) {
                output_shape.push_back(p.size(i));
            }
        }
        if (!output_shape.empty()) {
            grad_grad_output = grad_grad_output.view(output_shape);
        }
    }

    return std::make_tuple(grad_grad_output, grad_p, grad_q);
}

}  // namespace torchscience::cpu::information_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("chi_squared_divergence", &torchscience::cpu::information_theory::chi_squared_divergence);
    m.impl("chi_squared_divergence_backward", &torchscience::cpu::information_theory::chi_squared_divergence_backward);
    m.impl("chi_squared_divergence_backward_backward", &torchscience::cpu::information_theory::chi_squared_divergence_backward_backward);
}
