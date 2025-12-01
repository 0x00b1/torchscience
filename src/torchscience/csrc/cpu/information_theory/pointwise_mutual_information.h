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

#include "../../kernel/information_theory/pointwise_mutual_information.h"
#include "../../kernel/information_theory/pointwise_mutual_information_backward.h"
#include "../../kernel/information_theory/pointwise_mutual_information_backward_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

inline at::Tensor pmi_preprocess_input(
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
            "pointwise_mutual_information: input_type must be 'probability' or 'log_probability', got '",
            input_type, "'"
        );
    }
}

inline double pmi_get_log_base_scale(c10::optional<double> base) {
    if (!base.has_value()) {
        return 1.0;
    }
    double b = base.value();
    TORCH_CHECK(b > 0 && b != 1, "pointwise_mutual_information: base must be positive and not equal to 1");
    return 1.0 / std::log(b);
}

}  // anonymous namespace

inline at::Tensor pointwise_mutual_information(
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    c10::optional<double> base
) {
    at::Tensor joint_prob = pmi_preprocess_input(joint, input_type);
    double log_base_scale = pmi_get_log_base_scale(base);

    int64_t ndim = joint_prob.dim();
    TORCH_CHECK(dims.size() == 2, "pointwise_mutual_information: dims must have exactly 2 elements");

    int64_t dim0 = dims[0] < 0 ? ndim + dims[0] : dims[0];
    int64_t dim1 = dims[1] < 0 ? ndim + dims[1] : dims[1];
    TORCH_CHECK(dim0 >= 0 && dim0 < ndim, "pointwise_mutual_information: dims[0] out of range");
    TORCH_CHECK(dim1 >= 0 && dim1 < ndim, "pointwise_mutual_information: dims[1] out of range");
    TORCH_CHECK(dim0 != dim1, "pointwise_mutual_information: dims must be different");

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

    at::Tensor output = at::empty_like(joint_t);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "pointwise_mutual_information_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t scale = static_cast<scalar_t>(log_base_scale);

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                std::vector<scalar_t> p_x(size_x);
                std::vector<scalar_t> p_y(size_y);

                for (int64_t idx = begin; idx < end; ++idx) {
                    const scalar_t* batch_joint = joint_ptr + idx * size_x * size_y;
                    scalar_t* batch_out = out_ptr + idx * size_x * size_y;

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

                    // Compute PMI for each element
                    for (int64_t i = 0; i < size_x; ++i) {
                        for (int64_t j = 0; j < size_y; ++j) {
                            batch_out[i * size_y + j] =
                                torchscience::kernel::information_theory::pointwise_mutual_information_kernel<scalar_t>(
                                    batch_joint[i * size_y + j],
                                    p_x[i],
                                    p_y[j],
                                    scale
                                );
                        }
                    }
                }
            });
        }
    );

    // Reshape back
    std::vector<int64_t> permuted_shape;
    for (auto idx : perm) {
        permuted_shape.push_back(joint_prob.size(idx));
    }
    output = output.view(permuted_shape);

    std::vector<int64_t> inv_perm(ndim);
    for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
        inv_perm[perm[i]] = i;
    }

    return output.permute(inv_perm).contiguous();
}

inline at::Tensor pointwise_mutual_information_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    c10::optional<double> base
) {
    at::Tensor joint_prob = pmi_preprocess_input(joint, input_type);
    double log_base_scale = pmi_get_log_base_scale(base);

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
    at::Tensor grad_t = grad_output.permute(perm).contiguous();
    int64_t size_x = joint_t.size(-2);
    int64_t size_y = joint_t.size(-1);

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }
    joint_t = joint_t.view({batch_size, size_x, size_y});
    grad_t = grad_t.view({batch_size, size_x, size_y});

    at::Tensor grad_joint_t = at::zeros_like(joint_t);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "pointwise_mutual_information_backward_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_t.data_ptr<scalar_t>();
            scalar_t* grad_joint_ptr = grad_joint_t.data_ptr<scalar_t>();
            scalar_t scale = static_cast<scalar_t>(log_base_scale);

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                std::vector<scalar_t> p_x(size_x);
                std::vector<scalar_t> p_y(size_y);
                std::vector<scalar_t> g_x(size_x);
                std::vector<scalar_t> g_y(size_y);

                for (int64_t idx = begin; idx < end; ++idx) {
                    const scalar_t* batch_joint = joint_ptr + idx * size_x * size_y;
                    const scalar_t* batch_grad = grad_ptr + idx * size_x * size_y;
                    scalar_t* batch_grad_joint = grad_joint_ptr + idx * size_x * size_y;

                    // Compute marginals and gradient sums
                    std::fill(p_x.begin(), p_x.end(), scalar_t(0));
                    std::fill(p_y.begin(), p_y.end(), scalar_t(0));
                    std::fill(g_x.begin(), g_x.end(), scalar_t(0));
                    std::fill(g_y.begin(), g_y.end(), scalar_t(0));

                    for (int64_t i = 0; i < size_x; ++i) {
                        for (int64_t j = 0; j < size_y; ++j) {
                            int64_t ij = i * size_y + j;
                            p_x[i] += batch_joint[ij];
                            p_y[j] += batch_joint[ij];
                            g_x[i] += batch_grad[ij];
                            g_y[j] += batch_grad[ij];
                        }
                    }

                    torchscience::kernel::information_theory::pointwise_mutual_information_backward_kernel<scalar_t>(
                        batch_grad,
                        batch_joint,
                        p_x.data(),
                        p_y.data(),
                        g_x.data(),
                        g_y.data(),
                        size_x,
                        size_y,
                        scale,
                        batch_grad_joint
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

inline std::tuple<at::Tensor, at::Tensor> pointwise_mutual_information_backward_backward(
    const at::Tensor& gg_joint,
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    c10::optional<double> base
) {
    at::Tensor joint_prob = pmi_preprocess_input(joint, input_type);
    double log_base_scale = pmi_get_log_base_scale(base);

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
    at::Tensor grad_t = grad_output.permute(perm).contiguous();
    at::Tensor gg_joint_t = gg_joint.defined() ? gg_joint.permute(perm).contiguous() : at::Tensor();

    int64_t size_x = joint_t.size(-2);
    int64_t size_y = joint_t.size(-1);

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }
    joint_t = joint_t.view({batch_size, size_x, size_y});
    grad_t = grad_t.view({batch_size, size_x, size_y});
    if (gg_joint_t.defined()) {
        gg_joint_t = gg_joint_t.view({batch_size, size_x, size_y});
    }

    at::Tensor grad_grad_output_t = at::zeros_like(joint_t);
    at::Tensor grad_joint_t = at::zeros_like(joint_t);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "pointwise_mutual_information_backward_backward_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_t.data_ptr<scalar_t>();
            const scalar_t* gg_joint_ptr = gg_joint_t.defined() ? gg_joint_t.data_ptr<scalar_t>() : nullptr;
            scalar_t* grad_grad_out_ptr = grad_grad_output_t.data_ptr<scalar_t>();
            scalar_t* grad_joint_ptr = grad_joint_t.data_ptr<scalar_t>();
            scalar_t scale = static_cast<scalar_t>(log_base_scale);

            std::vector<scalar_t> p_x(size_x);
            std::vector<scalar_t> p_y(size_y);

            for (int64_t idx = 0; idx < batch_size; ++idx) {
                const scalar_t* batch_joint = joint_ptr + idx * size_x * size_y;
                const scalar_t* batch_grad = grad_ptr + idx * size_x * size_y;
                const scalar_t* batch_gg = gg_joint_ptr ? gg_joint_ptr + idx * size_x * size_y : nullptr;
                scalar_t* batch_grad_grad_out = grad_grad_out_ptr + idx * size_x * size_y;
                scalar_t* batch_grad_joint = grad_joint_ptr + idx * size_x * size_y;

                // Compute marginals
                std::fill(p_x.begin(), p_x.end(), scalar_t(0));
                std::fill(p_y.begin(), p_y.end(), scalar_t(0));
                for (int64_t i = 0; i < size_x; ++i) {
                    for (int64_t j = 0; j < size_y; ++j) {
                        p_x[i] += batch_joint[i * size_y + j];
                        p_y[j] += batch_joint[i * size_y + j];
                    }
                }

                torchscience::kernel::information_theory::pointwise_mutual_information_backward_backward_kernel<scalar_t>(
                    batch_gg,
                    batch_grad,
                    batch_joint,
                    p_x.data(),
                    p_y.data(),
                    size_x,
                    size_y,
                    scale,
                    batch_grad_grad_out,
                    batch_grad_joint
                );
            }
        }
    );

    // Reshape back
    std::vector<int64_t> permuted_shape;
    for (auto idx : perm) {
        permuted_shape.push_back(joint_prob.size(idx));
    }
    grad_grad_output_t = grad_grad_output_t.view(permuted_shape);
    grad_joint_t = grad_joint_t.view(permuted_shape);

    std::vector<int64_t> inv_perm(ndim);
    for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
        inv_perm[perm[i]] = i;
    }

    return std::make_tuple(
        grad_grad_output_t.permute(inv_perm).contiguous(),
        grad_joint_t.permute(inv_perm).contiguous()
    );
}

}  // namespace torchscience::cpu::information_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("pointwise_mutual_information", &torchscience::cpu::information_theory::pointwise_mutual_information);
    m.impl("pointwise_mutual_information_backward", &torchscience::cpu::information_theory::pointwise_mutual_information_backward);
    m.impl("pointwise_mutual_information_backward_backward", &torchscience::cpu::information_theory::pointwise_mutual_information_backward_backward);
}
