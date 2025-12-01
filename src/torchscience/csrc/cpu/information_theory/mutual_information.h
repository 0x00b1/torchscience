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

#include "../../kernel/information_theory/mutual_information.h"
#include "../../kernel/information_theory/mutual_information_backward.h"
#include "../../kernel/information_theory/mutual_information_backward_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

inline at::Tensor mi_preprocess_input(
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
            "mutual_information: input_type must be 'probability' or 'log_probability', got '",
            input_type, "'"
        );
    }
}

inline at::Tensor mi_apply_reduction(
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
            "mutual_information: reduction must be 'none', 'mean', or 'sum', got '",
            reduction, "'"
        );
    }
}

inline double mi_get_log_base_scale(c10::optional<double> base) {
    if (!base.has_value()) {
        return 1.0;
    }
    double b = base.value();
    TORCH_CHECK(b > 0 && b != 1, "mutual_information: base must be positive and not equal to 1");
    return 1.0 / std::log(b);
}

}  // anonymous namespace

inline at::Tensor mutual_information(
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    at::Tensor joint_prob = mi_preprocess_input(joint, input_type);
    double log_base_scale = mi_get_log_base_scale(base);

    int64_t ndim = joint_prob.dim();
    TORCH_CHECK(dims.size() == 2, "mutual_information: dims must have exactly 2 elements");

    int64_t dim0 = dims[0] < 0 ? ndim + dims[0] : dims[0];
    int64_t dim1 = dims[1] < 0 ? ndim + dims[1] : dims[1];
    TORCH_CHECK(dim0 >= 0 && dim0 < ndim, "mutual_information: dims[0] out of range");
    TORCH_CHECK(dim1 >= 0 && dim1 < ndim, "mutual_information: dims[1] out of range");
    TORCH_CHECK(dim0 != dim1, "mutual_information: dims must be different");

    if (dim0 > dim1) std::swap(dim0, dim1);

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
    int64_t size_x = joint_t.size(-2);
    int64_t size_y = joint_t.size(-1);

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }
    joint_t = joint_t.view({batch_size, size_x, size_y});

    at::Tensor output = at::empty({batch_size}, joint_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "mutual_information_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t scale = static_cast<scalar_t>(log_base_scale);

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                // Allocate marginals per thread
                std::vector<scalar_t> p_x(size_x);
                std::vector<scalar_t> p_y(size_y);

                for (int64_t idx = begin; idx < end; ++idx) {
                    const scalar_t* batch_joint = joint_ptr + idx * size_x * size_y;

                    // Compute marginals
                    std::fill(p_x.begin(), p_x.end(), scalar_t(0));
                    std::fill(p_y.begin(), p_y.end(), scalar_t(0));

                    for (int64_t i = 0; i < size_x; ++i) {
                        for (int64_t j = 0; j < size_y; ++j) {
                            scalar_t p_xy = batch_joint[i * size_y + j];
                            p_x[i] += p_xy;
                            p_y[j] += p_xy;
                        }
                    }

                    out_ptr[idx] = torchscience::kernel::information_theory::mutual_information_kernel_dynamic<scalar_t>(
                        batch_joint,
                        p_x.data(),
                        p_y.data(),
                        size_x,
                        size_y,
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

    return mi_apply_reduction(output, reduction);
}

inline at::Tensor mutual_information_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    at::Tensor joint_prob = mi_preprocess_input(joint, input_type);
    double log_base_scale = mi_get_log_base_scale(base);

    int64_t ndim = joint_prob.dim();
    int64_t dim0 = dims[0] < 0 ? ndim + dims[0] : dims[0];
    int64_t dim1 = dims[1] < 0 ? ndim + dims[1] : dims[1];
    if (dim0 > dim1) std::swap(dim0, dim1);

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
    int64_t size_x = joint_t.size(-2);
    int64_t size_y = joint_t.size(-1);

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }
    joint_t = joint_t.view({batch_size, size_x, size_y});

    at::Tensor grad_joint_t = at::zeros_like(joint_t);

    double scale = 1.0;
    if (reduction == "mean") {
        scale = 1.0 / static_cast<double>(batch_size);
    }

    at::Tensor grad_flat = grad_output.contiguous().view({-1});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "mutual_information_backward_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_flat.data_ptr<scalar_t>();
            scalar_t* grad_joint_ptr = grad_joint_t.data_ptr<scalar_t>();
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);
            scalar_t red_scale = static_cast<scalar_t>(scale);

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                std::vector<scalar_t> p_x(size_x);
                std::vector<scalar_t> p_y(size_y);

                for (int64_t idx = begin; idx < end; ++idx) {
                    const scalar_t* batch_joint = joint_ptr + idx * size_x * size_y;
                    scalar_t* batch_grad = grad_joint_ptr + idx * size_x * size_y;

                    // Compute marginals
                    std::fill(p_x.begin(), p_x.end(), scalar_t(0));
                    std::fill(p_y.begin(), p_y.end(), scalar_t(0));

                    for (int64_t i = 0; i < size_x; ++i) {
                        for (int64_t j = 0; j < size_y; ++j) {
                            scalar_t p_xy = batch_joint[i * size_y + j];
                            p_x[i] += p_xy;
                            p_y[j] += p_xy;
                        }
                    }

                    scalar_t grad_val = (reduction == "none") ? grad_ptr[idx] : grad_ptr[0] * red_scale;

                    torchscience::kernel::information_theory::mutual_information_backward_kernel<scalar_t>(
                        grad_val,
                        batch_joint,
                        p_x.data(),
                        p_y.data(),
                        size_x,
                        size_y,
                        log_scale,
                        batch_grad
                    );
                }
            });
        }
    );

    // Reshape back
    std::vector<int64_t> permuted_shape;
    for (auto idx : perm) {
        permuted_shape.push_back(joint_prob.size(idx));
    }
    grad_joint_t = grad_joint_t.view(permuted_shape);

    std::vector<int64_t> inv_perm(ndim);
    for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
        inv_perm[perm[i]] = i;
    }

    return grad_joint_t.permute(inv_perm).contiguous();
}

inline std::tuple<at::Tensor, at::Tensor> mutual_information_backward_backward(
    const at::Tensor& gg_joint,
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    at::Tensor joint_prob = mi_preprocess_input(joint, input_type);
    double log_base_scale = mi_get_log_base_scale(base);

    int64_t ndim = joint_prob.dim();
    int64_t dim0 = dims[0] < 0 ? ndim + dims[0] : dims[0];
    int64_t dim1 = dims[1] < 0 ? ndim + dims[1] : dims[1];
    if (dim0 > dim1) std::swap(dim0, dim1);

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

    int64_t size_x = joint_t.size(-2);
    int64_t size_y = joint_t.size(-1);

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }
    joint_t = joint_t.view({batch_size, size_x, size_y});
    if (gg_joint_t.defined()) {
        gg_joint_t = gg_joint_t.view({batch_size, size_x, size_y});
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
        "mutual_information_backward_backward_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            const scalar_t* gg_joint_ptr = gg_joint_t.defined() ? gg_joint_t.data_ptr<scalar_t>() : nullptr;
            const scalar_t* grad_out_ptr = grad_out_flat.data_ptr<scalar_t>();
            scalar_t* grad_grad_out_ptr = grad_grad_output.data_ptr<scalar_t>();
            scalar_t* grad_joint_ptr = grad_joint_t.data_ptr<scalar_t>();
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);
            scalar_t reduction_scale = static_cast<scalar_t>(scale);

            int64_t joint_stride = size_x * size_y;

            std::vector<scalar_t> p_x(size_x);
            std::vector<scalar_t> p_y(size_y);

            if (reduction == "none") {
                for (int64_t idx = 0; idx < batch_size; ++idx) {
                    const scalar_t* batch_joint = joint_ptr + idx * joint_stride;

                    // Compute marginals
                    std::fill(p_x.begin(), p_x.end(), scalar_t(0));
                    std::fill(p_y.begin(), p_y.end(), scalar_t(0));
                    for (int64_t i = 0; i < size_x; ++i) {
                        for (int64_t j = 0; j < size_y; ++j) {
                            scalar_t p_xy = batch_joint[i * size_y + j];
                            p_x[i] += p_xy;
                            p_y[j] += p_xy;
                        }
                    }

                    scalar_t grad_val = grad_out_ptr[idx];
                    scalar_t grad_grad_out_val = scalar_t(0);

                    torchscience::kernel::information_theory::mutual_information_backward_backward_kernel<scalar_t>(
                        gg_joint_ptr ? gg_joint_ptr + idx * joint_stride : nullptr,
                        grad_val,
                        batch_joint,
                        p_x.data(),
                        p_y.data(),
                        size_x,
                        size_y,
                        log_scale,
                        grad_grad_out_val,
                        grad_joint_ptr + idx * joint_stride
                    );

                    grad_grad_out_ptr[idx] = grad_grad_out_val;
                }
            } else {
                scalar_t total_grad_grad_out = scalar_t(0);
                for (int64_t idx = 0; idx < batch_size; ++idx) {
                    const scalar_t* batch_joint = joint_ptr + idx * joint_stride;

                    // Compute marginals
                    std::fill(p_x.begin(), p_x.end(), scalar_t(0));
                    std::fill(p_y.begin(), p_y.end(), scalar_t(0));
                    for (int64_t i = 0; i < size_x; ++i) {
                        for (int64_t j = 0; j < size_y; ++j) {
                            scalar_t p_xy = batch_joint[i * size_y + j];
                            p_x[i] += p_xy;
                            p_y[j] += p_xy;
                        }
                    }

                    scalar_t grad_val = grad_out_ptr[0] * reduction_scale;
                    scalar_t grad_grad_out_val = scalar_t(0);

                    torchscience::kernel::information_theory::mutual_information_backward_backward_kernel<scalar_t>(
                        gg_joint_ptr ? gg_joint_ptr + idx * joint_stride : nullptr,
                        grad_val,
                        batch_joint,
                        p_x.data(),
                        p_y.data(),
                        size_x,
                        size_y,
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

    // Reshape back
    std::vector<int64_t> permuted_shape;
    for (auto idx : perm) {
        permuted_shape.push_back(joint_prob.size(idx));
    }
    grad_joint_t = grad_joint_t.view(permuted_shape);

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
    m.impl("mutual_information", &torchscience::cpu::information_theory::mutual_information);
    m.impl("mutual_information_backward", &torchscience::cpu::information_theory::mutual_information_backward);
    m.impl("mutual_information_backward_backward", &torchscience::cpu::information_theory::mutual_information_backward_backward);
}
