#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "../../kernel/information_theory/interaction_information.h"
#include "../../kernel/information_theory/interaction_information_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

inline at::Tensor ii_preprocess_input(
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

inline at::Tensor ii_apply_reduction(
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

inline double ii_get_log_base_scale(c10::optional<double> base) {
    if (!base.has_value()) {
        return 1.0;
    }
    double b = base.value();
    TORCH_CHECK(b > 0 && b != 1, "base must be positive and not equal to 1");
    return 1.0 / std::log(b);
}

}  // anonymous namespace

inline at::Tensor interaction_information(
    const at::Tensor& joint,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    TORCH_CHECK(joint.dim() == 3, "joint must have exactly 3 dimensions for interaction information");

    at::Tensor joint_prob = ii_preprocess_input(joint, input_type);
    double log_base_scale = ii_get_log_base_scale(base);

    int64_t size_x = joint_prob.size(0);
    int64_t size_y = joint_prob.size(1);
    int64_t size_z = joint_prob.size(2);

    // Make contiguous
    at::Tensor joint_contig = joint_prob.contiguous();

    // Allocate output (scalar)
    at::Tensor output = at::empty({}, joint_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "interaction_information_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_contig.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t scale = static_cast<scalar_t>(log_base_scale);

            // Allocate marginals
            std::vector<scalar_t> p_x(size_x, scalar_t(0));
            std::vector<scalar_t> p_y(size_y, scalar_t(0));
            std::vector<scalar_t> p_z(size_z, scalar_t(0));
            std::vector<scalar_t> p_xy(size_x * size_y, scalar_t(0));
            std::vector<scalar_t> p_xz(size_x * size_z, scalar_t(0));
            std::vector<scalar_t> p_yz(size_y * size_z, scalar_t(0));

            // Compute all marginals by summing
            for (int64_t x = 0; x < size_x; ++x) {
                for (int64_t y = 0; y < size_y; ++y) {
                    for (int64_t z = 0; z < size_z; ++z) {
                        scalar_t pxyz = joint_ptr[(x * size_y + y) * size_z + z];
                        p_x[x] += pxyz;
                        p_y[y] += pxyz;
                        p_z[z] += pxyz;
                        p_xy[x * size_y + y] += pxyz;
                        p_xz[x * size_z + z] += pxyz;
                        p_yz[y * size_z + z] += pxyz;
                    }
                }
            }

            // Call kernel
            out_ptr[0] = torchscience::kernel::information_theory::interaction_information_kernel<scalar_t>(
                joint_ptr,
                p_x.data(),
                p_y.data(),
                p_z.data(),
                p_xy.data(),
                p_xz.data(),
                p_yz.data(),
                size_x,
                size_y,
                size_z,
                scale
            );
        }
    );

    return ii_apply_reduction(output, reduction);
}

inline at::Tensor interaction_information_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    at::Tensor joint_prob = ii_preprocess_input(joint, input_type);
    double log_base_scale = ii_get_log_base_scale(base);

    int64_t size_x = joint_prob.size(0);
    int64_t size_y = joint_prob.size(1);
    int64_t size_z = joint_prob.size(2);

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
        "interaction_information_backward_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_contig.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_flat.data_ptr<scalar_t>();
            scalar_t* grad_joint_ptr = grad_joint.data_ptr<scalar_t>();
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);

            // Compute marginals
            std::vector<scalar_t> p_x(size_x, scalar_t(0));
            std::vector<scalar_t> p_y(size_y, scalar_t(0));
            std::vector<scalar_t> p_z(size_z, scalar_t(0));
            std::vector<scalar_t> p_xy(size_x * size_y, scalar_t(0));
            std::vector<scalar_t> p_xz(size_x * size_z, scalar_t(0));
            std::vector<scalar_t> p_yz(size_y * size_z, scalar_t(0));

            for (int64_t x = 0; x < size_x; ++x) {
                for (int64_t y = 0; y < size_y; ++y) {
                    for (int64_t z = 0; z < size_z; ++z) {
                        scalar_t pxyz = joint_ptr[(x * size_y + y) * size_z + z];
                        p_x[x] += pxyz;
                        p_y[y] += pxyz;
                        p_z[z] += pxyz;
                        p_xy[x * size_y + y] += pxyz;
                        p_xz[x * size_z + z] += pxyz;
                        p_yz[y * size_z + z] += pxyz;
                    }
                }
            }

            scalar_t grad_val = grad_ptr[0] * static_cast<scalar_t>(red_scale);

            torchscience::kernel::information_theory::interaction_information_backward_kernel<scalar_t>(
                grad_val,
                joint_ptr,
                p_x.data(),
                p_y.data(),
                p_z.data(),
                p_xy.data(),
                p_xz.data(),
                p_yz.data(),
                size_x,
                size_y,
                size_z,
                log_scale,
                grad_joint_ptr
            );
        }
    );

    return grad_joint;
}

}  // namespace torchscience::cpu::information_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("interaction_information", &torchscience::cpu::information_theory::interaction_information);
    m.impl("interaction_information_backward", &torchscience::cpu::information_theory::interaction_information_backward);
}
