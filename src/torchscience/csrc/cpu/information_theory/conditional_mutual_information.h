#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "../../kernel/information_theory/conditional_mutual_information.h"
#include "../../kernel/information_theory/conditional_mutual_information_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

inline at::Tensor cmi_preprocess_input(
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

inline at::Tensor cmi_apply_reduction(
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

inline double cmi_get_log_base_scale(c10::optional<double> base) {
    if (!base.has_value()) {
        return 1.0;
    }
    double b = base.value();
    TORCH_CHECK(b > 0 && b != 1, "base must be positive and not equal to 1");
    return 1.0 / std::log(b);
}

}  // anonymous namespace

inline at::Tensor conditional_mutual_information(
    const at::Tensor& joint,
    at::IntArrayRef dims_x,
    at::IntArrayRef dims_y,
    at::IntArrayRef dims_z,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    TORCH_CHECK(joint.dim() >= 3, "joint must have at least 3 dimensions");

    // For simplicity, require exactly single-dimension variables
    TORCH_CHECK(
        dims_x.size() == 1 && dims_y.size() == 1 && dims_z.size() == 1,
        "Currently only single-dimension variables are supported"
    );

    int64_t ndim = joint.dim();
    int64_t dim_x = dims_x[0] < 0 ? ndim + dims_x[0] : dims_x[0];
    int64_t dim_y = dims_y[0] < 0 ? ndim + dims_y[0] : dims_y[0];
    int64_t dim_z = dims_z[0] < 0 ? ndim + dims_z[0] : dims_z[0];

    TORCH_CHECK(dim_x >= 0 && dim_x < ndim, "dims_x out of range");
    TORCH_CHECK(dim_y >= 0 && dim_y < ndim, "dims_y out of range");
    TORCH_CHECK(dim_z >= 0 && dim_z < ndim, "dims_z out of range");
    TORCH_CHECK(dim_x != dim_y && dim_x != dim_z && dim_y != dim_z, "dims must be different");

    at::Tensor joint_prob = cmi_preprocess_input(joint, input_type);
    double log_base_scale = cmi_get_log_base_scale(base);

    // Permute to have dims in order (batch..., x, y, z) at the end
    std::vector<int64_t> perm;
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim_x && i != dim_y && i != dim_z) {
            perm.push_back(i);
            batch_shape.push_back(joint_prob.size(i));
        }
    }
    perm.push_back(dim_x);
    perm.push_back(dim_y);
    perm.push_back(dim_z);

    at::Tensor joint_t = joint_prob.permute(perm).contiguous();

    int64_t size_x = joint_t.size(-3);
    int64_t size_y = joint_t.size(-2);
    int64_t size_z = joint_t.size(-1);

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }
    joint_t = joint_t.view({batch_size, size_x, size_y, size_z});

    at::Tensor output = at::empty({batch_size}, joint_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "conditional_mutual_information_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t scale = static_cast<scalar_t>(log_base_scale);

            int64_t joint_stride = size_x * size_y * size_z;

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                // Allocate marginals per thread
                std::vector<scalar_t> p_z(size_z);
                std::vector<scalar_t> p_xz(size_x * size_z);
                std::vector<scalar_t> p_yz(size_y * size_z);

                for (int64_t idx = begin; idx < end; ++idx) {
                    const scalar_t* batch_joint = joint_ptr + idx * joint_stride;

                    // Initialize marginals
                    std::fill(p_z.begin(), p_z.end(), scalar_t(0));
                    std::fill(p_xz.begin(), p_xz.end(), scalar_t(0));
                    std::fill(p_yz.begin(), p_yz.end(), scalar_t(0));

                    // Compute marginals by summing
                    for (int64_t x = 0; x < size_x; ++x) {
                        for (int64_t y = 0; y < size_y; ++y) {
                            for (int64_t z = 0; z < size_z; ++z) {
                                scalar_t p_xyz = batch_joint[(x * size_y + y) * size_z + z];
                                p_z[z] += p_xyz;
                                p_xz[x * size_z + z] += p_xyz;
                                p_yz[y * size_z + z] += p_xyz;
                            }
                        }
                    }

                    out_ptr[idx] = torchscience::kernel::information_theory::conditional_mutual_information_kernel<scalar_t>(
                        batch_joint,
                        p_xz.data(),
                        p_yz.data(),
                        p_z.data(),
                        size_x,
                        size_y,
                        size_z,
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

    return cmi_apply_reduction(output, reduction);
}

inline at::Tensor conditional_mutual_information_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    at::IntArrayRef dims_x,
    at::IntArrayRef dims_y,
    at::IntArrayRef dims_z,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    int64_t ndim = joint.dim();
    int64_t dim_x = dims_x[0] < 0 ? ndim + dims_x[0] : dims_x[0];
    int64_t dim_y = dims_y[0] < 0 ? ndim + dims_y[0] : dims_y[0];
    int64_t dim_z = dims_z[0] < 0 ? ndim + dims_z[0] : dims_z[0];

    at::Tensor joint_prob = cmi_preprocess_input(joint, input_type);
    double log_base_scale = cmi_get_log_base_scale(base);

    // Permute to have dims in order (batch..., x, y, z) at the end
    std::vector<int64_t> perm;
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim_x && i != dim_y && i != dim_z) {
            perm.push_back(i);
            batch_shape.push_back(joint_prob.size(i));
        }
    }
    perm.push_back(dim_x);
    perm.push_back(dim_y);
    perm.push_back(dim_z);

    at::Tensor joint_t = joint_prob.permute(perm).contiguous();

    int64_t size_x = joint_t.size(-3);
    int64_t size_y = joint_t.size(-2);
    int64_t size_z = joint_t.size(-1);

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }
    joint_t = joint_t.view({batch_size, size_x, size_y, size_z});

    at::Tensor grad_joint_t = at::zeros_like(joint_t);

    double scale = 1.0;
    if (reduction == "mean") {
        scale = 1.0 / static_cast<double>(batch_size);
    }

    at::Tensor grad_flat = grad_output.contiguous().view({-1});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "conditional_mutual_information_backward_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_flat.data_ptr<scalar_t>();
            scalar_t* grad_joint_ptr = grad_joint_t.data_ptr<scalar_t>();
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);
            scalar_t red_scale = static_cast<scalar_t>(scale);

            int64_t joint_stride = size_x * size_y * size_z;

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                std::vector<scalar_t> p_z(size_z);
                std::vector<scalar_t> p_xz(size_x * size_z);
                std::vector<scalar_t> p_yz(size_y * size_z);

                for (int64_t idx = begin; idx < end; ++idx) {
                    const scalar_t* batch_joint = joint_ptr + idx * joint_stride;
                    scalar_t* batch_grad = grad_joint_ptr + idx * joint_stride;

                    // Initialize marginals
                    std::fill(p_z.begin(), p_z.end(), scalar_t(0));
                    std::fill(p_xz.begin(), p_xz.end(), scalar_t(0));
                    std::fill(p_yz.begin(), p_yz.end(), scalar_t(0));

                    // Compute marginals
                    for (int64_t x = 0; x < size_x; ++x) {
                        for (int64_t y = 0; y < size_y; ++y) {
                            for (int64_t z = 0; z < size_z; ++z) {
                                scalar_t p_xyz = batch_joint[(x * size_y + y) * size_z + z];
                                p_z[z] += p_xyz;
                                p_xz[x * size_z + z] += p_xyz;
                                p_yz[y * size_z + z] += p_xyz;
                            }
                        }
                    }

                    scalar_t grad_val = (reduction == "none") ? grad_ptr[idx] : grad_ptr[0] * red_scale;

                    torchscience::kernel::information_theory::conditional_mutual_information_backward_kernel<scalar_t>(
                        grad_val,
                        batch_joint,
                        p_xz.data(),
                        p_yz.data(),
                        p_z.data(),
                        size_x,
                        size_y,
                        size_z,
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

    // Compute inverse permutation
    std::vector<int64_t> inv_perm(ndim);
    for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
        inv_perm[perm[i]] = i;
    }

    return grad_joint_t.permute(inv_perm).contiguous();
}

}  // namespace torchscience::cpu::information_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("conditional_mutual_information", &torchscience::cpu::information_theory::conditional_mutual_information);
    m.impl("conditional_mutual_information_backward", &torchscience::cpu::information_theory::conditional_mutual_information_backward);
}
