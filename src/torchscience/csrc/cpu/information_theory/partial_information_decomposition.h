#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "../../kernel/information_theory/partial_information_decomposition.h"
#include "../../kernel/information_theory/partial_information_decomposition_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

inline at::Tensor pid_preprocess_input(
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

inline double pid_get_log_base_scale(c10::optional<double> base) {
    if (!base.has_value()) {
        return 1.0;
    }
    double b = base.value();
    TORCH_CHECK(b > 0 && b != 1, "base must be positive and not equal to 1");
    return 1.0 / std::log(b);
}

}  // anonymous namespace

inline std::vector<at::Tensor> partial_information_decomposition(
    const at::Tensor& joint,
    const std::string& method,
    const std::string& input_type,
    c10::optional<double> base
) {
    TORCH_CHECK(joint.dim() >= 3, "joint must have at least 3 dimensions");
    TORCH_CHECK(method == "imin", "Only 'imin' method is currently supported");

    int64_t ndim = joint.dim();

    at::Tensor joint_prob = pid_preprocess_input(joint, input_type);
    double log_base_scale = pid_get_log_base_scale(base);

    // The last 3 dimensions are (x, y, z)
    int64_t size_x = joint_prob.size(-3);
    int64_t size_y = joint_prob.size(-2);
    int64_t size_z = joint_prob.size(-1);

    // Compute batch shape
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < ndim - 3; ++i) {
        batch_shape.push_back(joint_prob.size(i));
    }

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }

    at::Tensor joint_t = joint_prob.contiguous().view({batch_size, size_x, size_y, size_z});

    // Output tensors for each component
    at::Tensor redundancy = at::empty({batch_size}, joint_prob.options());
    at::Tensor unique_x = at::empty({batch_size}, joint_prob.options());
    at::Tensor unique_y = at::empty({batch_size}, joint_prob.options());
    at::Tensor synergy = at::empty({batch_size}, joint_prob.options());
    at::Tensor mutual_info = at::empty({batch_size}, joint_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "partial_information_decomposition_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            scalar_t* redundancy_ptr = redundancy.data_ptr<scalar_t>();
            scalar_t* unique_x_ptr = unique_x.data_ptr<scalar_t>();
            scalar_t* unique_y_ptr = unique_y.data_ptr<scalar_t>();
            scalar_t* synergy_ptr = synergy.data_ptr<scalar_t>();
            scalar_t* mutual_info_ptr = mutual_info.data_ptr<scalar_t>();
            scalar_t scale = static_cast<scalar_t>(log_base_scale);

            int64_t joint_stride = size_x * size_y * size_z;

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                // Allocate marginals per thread
                std::vector<scalar_t> p_x(size_x);
                std::vector<scalar_t> p_y(size_y);
                std::vector<scalar_t> p_z(size_z);
                std::vector<scalar_t> p_xy(size_x * size_y);
                std::vector<scalar_t> p_xz(size_x * size_z);
                std::vector<scalar_t> p_yz(size_y * size_z);

                for (int64_t idx = begin; idx < end; ++idx) {
                    const scalar_t* batch_joint = joint_ptr + idx * joint_stride;

                    // Initialize marginals
                    std::fill(p_x.begin(), p_x.end(), scalar_t(0));
                    std::fill(p_y.begin(), p_y.end(), scalar_t(0));
                    std::fill(p_z.begin(), p_z.end(), scalar_t(0));
                    std::fill(p_xy.begin(), p_xy.end(), scalar_t(0));
                    std::fill(p_xz.begin(), p_xz.end(), scalar_t(0));
                    std::fill(p_yz.begin(), p_yz.end(), scalar_t(0));

                    // Compute marginals by summing
                    for (int64_t x = 0; x < size_x; ++x) {
                        for (int64_t y = 0; y < size_y; ++y) {
                            for (int64_t z = 0; z < size_z; ++z) {
                                scalar_t p = batch_joint[(x * size_y + y) * size_z + z];
                                p_x[x] += p;
                                p_y[y] += p;
                                p_z[z] += p;
                                p_xy[x * size_y + y] += p;
                                p_xz[x * size_z + z] += p;
                                p_yz[y * size_z + z] += p;
                            }
                        }
                    }

                    // Compute PID
                    torchscience::kernel::information_theory::partial_information_decomposition_kernel<scalar_t>(
                        batch_joint,
                        p_x.data(),
                        p_y.data(),
                        p_z.data(),
                        p_xy.data(),
                        p_xz.data(),
                        p_yz.data(),
                        size_x,
                        size_y,
                        size_z,
                        scale,
                        &redundancy_ptr[idx],
                        &unique_x_ptr[idx],
                        &unique_y_ptr[idx],
                        &synergy_ptr[idx],
                        &mutual_info_ptr[idx]
                    );
                }
            });
        }
    );

    // Reshape outputs
    if (!batch_shape.empty()) {
        redundancy = redundancy.view(batch_shape);
        unique_x = unique_x.view(batch_shape);
        unique_y = unique_y.view(batch_shape);
        synergy = synergy.view(batch_shape);
        mutual_info = mutual_info.view(batch_shape);
    } else {
        redundancy = redundancy.squeeze();
        unique_x = unique_x.squeeze();
        unique_y = unique_y.squeeze();
        synergy = synergy.squeeze();
        mutual_info = mutual_info.squeeze();
    }

    return {redundancy, unique_x, unique_y, synergy, mutual_info};
}

inline std::vector<at::Tensor> partial_information_decomposition_backward(
    const at::Tensor& grad_redundancy,
    const at::Tensor& grad_unique_x,
    const at::Tensor& grad_unique_y,
    const at::Tensor& grad_synergy,
    const at::Tensor& grad_mutual_info,
    const at::Tensor& joint,
    const std::string& method,
    const std::string& input_type,
    c10::optional<double> base
) {
    int64_t ndim = joint.dim();

    at::Tensor joint_prob = pid_preprocess_input(joint, input_type);
    double log_base_scale = pid_get_log_base_scale(base);

    int64_t size_x = joint_prob.size(-3);
    int64_t size_y = joint_prob.size(-2);
    int64_t size_z = joint_prob.size(-1);

    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < ndim - 3; ++i) {
        batch_shape.push_back(joint_prob.size(i));
    }

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }

    at::Tensor joint_t = joint_prob.contiguous().view({batch_size, size_x, size_y, size_z});

    at::Tensor grad_joint_t = at::zeros_like(joint_t);

    at::Tensor grad_r_flat = grad_redundancy.contiguous().view({-1});
    at::Tensor grad_ux_flat = grad_unique_x.contiguous().view({-1});
    at::Tensor grad_uy_flat = grad_unique_y.contiguous().view({-1});
    at::Tensor grad_s_flat = grad_synergy.contiguous().view({-1});
    at::Tensor grad_mi_flat = grad_mutual_info.contiguous().view({-1});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "partial_information_decomposition_backward_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            const scalar_t* grad_r_ptr = grad_r_flat.data_ptr<scalar_t>();
            const scalar_t* grad_ux_ptr = grad_ux_flat.data_ptr<scalar_t>();
            const scalar_t* grad_uy_ptr = grad_uy_flat.data_ptr<scalar_t>();
            const scalar_t* grad_s_ptr = grad_s_flat.data_ptr<scalar_t>();
            const scalar_t* grad_mi_ptr = grad_mi_flat.data_ptr<scalar_t>();
            scalar_t* grad_joint_ptr = grad_joint_t.data_ptr<scalar_t>();
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);

            int64_t joint_stride = size_x * size_y * size_z;

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                std::vector<scalar_t> p_x(size_x);
                std::vector<scalar_t> p_y(size_y);
                std::vector<scalar_t> p_z(size_z);
                std::vector<scalar_t> p_xy(size_x * size_y);
                std::vector<scalar_t> p_xz(size_x * size_z);
                std::vector<scalar_t> p_yz(size_y * size_z);

                for (int64_t idx = begin; idx < end; ++idx) {
                    const scalar_t* batch_joint = joint_ptr + idx * joint_stride;
                    scalar_t* batch_grad = grad_joint_ptr + idx * joint_stride;

                    // Initialize marginals
                    std::fill(p_x.begin(), p_x.end(), scalar_t(0));
                    std::fill(p_y.begin(), p_y.end(), scalar_t(0));
                    std::fill(p_z.begin(), p_z.end(), scalar_t(0));
                    std::fill(p_xy.begin(), p_xy.end(), scalar_t(0));
                    std::fill(p_xz.begin(), p_xz.end(), scalar_t(0));
                    std::fill(p_yz.begin(), p_yz.end(), scalar_t(0));

                    // Compute marginals
                    for (int64_t x = 0; x < size_x; ++x) {
                        for (int64_t y = 0; y < size_y; ++y) {
                            for (int64_t z = 0; z < size_z; ++z) {
                                scalar_t p = batch_joint[(x * size_y + y) * size_z + z];
                                p_x[x] += p;
                                p_y[y] += p;
                                p_z[z] += p;
                                p_xy[x * size_y + y] += p;
                                p_xz[x * size_z + z] += p;
                                p_yz[y * size_z + z] += p;
                            }
                        }
                    }

                    torchscience::kernel::information_theory::partial_information_decomposition_backward_kernel<scalar_t>(
                        grad_r_ptr[idx],
                        grad_ux_ptr[idx],
                        grad_uy_ptr[idx],
                        grad_s_ptr[idx],
                        grad_mi_ptr[idx],
                        batch_joint,
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
                        batch_grad
                    );
                }
            });
        }
    );

    // Reshape back to original shape
    std::vector<int64_t> output_shape = batch_shape;
    output_shape.push_back(size_x);
    output_shape.push_back(size_y);
    output_shape.push_back(size_z);

    return {grad_joint_t.view(output_shape)};
}

}  // namespace torchscience::cpu::information_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("partial_information_decomposition", &torchscience::cpu::information_theory::partial_information_decomposition);
    m.impl("partial_information_decomposition_backward", &torchscience::cpu::information_theory::partial_information_decomposition_backward);
}
