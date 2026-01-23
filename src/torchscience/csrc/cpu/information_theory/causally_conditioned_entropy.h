#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "../../kernel/information_theory/causally_conditioned_entropy.h"
#include "../../kernel/information_theory/causally_conditioned_entropy_backward.h"

namespace torchscience::cpu::information_theory {

namespace {

inline at::Tensor cce_preprocess_input(
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

inline at::Tensor cce_apply_reduction(
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

inline double cce_get_log_base_scale(c10::optional<double> base) {
    if (!base.has_value()) {
        return 1.0;
    }
    double b = base.value();
    TORCH_CHECK(b > 0 && b != 1, "base must be positive and not equal to 1");
    return 1.0 / std::log(b);
}

}  // anonymous namespace

inline at::Tensor causally_conditioned_entropy(
    const at::Tensor& joint,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    TORCH_CHECK(joint.dim() >= 3, "joint must have at least 3 dimensions");

    int64_t ndim = joint.dim();

    at::Tensor joint_prob = cce_preprocess_input(joint, input_type);
    double log_base_scale = cce_get_log_base_scale(base);

    // The last 3 dimensions are (y_t, y_prev, x_t)
    // All preceding dimensions are batch dimensions
    int64_t size_yt = joint_prob.size(-3);
    int64_t size_yprev = joint_prob.size(-2);
    int64_t size_xt = joint_prob.size(-1);

    // Compute batch shape
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < ndim - 3; ++i) {
        batch_shape.push_back(joint_prob.size(i));
    }

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }

    at::Tensor joint_t = joint_prob.contiguous().view({batch_size, size_yt, size_yprev, size_xt});

    at::Tensor output = at::empty({batch_size}, joint_prob.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "causally_conditioned_entropy_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t scale = static_cast<scalar_t>(log_base_scale);

            int64_t joint_stride = size_yt * size_yprev * size_xt;

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                // Allocate marginals per thread
                std::vector<scalar_t> p_yprev_xt(size_yprev * size_xt);

                for (int64_t idx = begin; idx < end; ++idx) {
                    const scalar_t* batch_joint = joint_ptr + idx * joint_stride;

                    // Initialize marginals
                    std::fill(p_yprev_xt.begin(), p_yprev_xt.end(), scalar_t(0));

                    // Compute marginal p(y_prev, x_t) by summing over y_t
                    // joint has shape [size_yt, size_yprev, size_xt]
                    for (int64_t yt = 0; yt < size_yt; ++yt) {
                        for (int64_t yp = 0; yp < size_yprev; ++yp) {
                            for (int64_t xt = 0; xt < size_xt; ++xt) {
                                scalar_t p = batch_joint[(yt * size_yprev + yp) * size_xt + xt];
                                p_yprev_xt[yp * size_xt + xt] += p;
                            }
                        }
                    }

                    out_ptr[idx] = torchscience::kernel::information_theory::causally_conditioned_entropy_kernel<scalar_t>(
                        batch_joint,
                        p_yprev_xt.data(),
                        size_yt,
                        size_yprev,
                        size_xt,
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

    return cce_apply_reduction(output, reduction);
}

inline at::Tensor causally_conditioned_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    int64_t ndim = joint.dim();

    at::Tensor joint_prob = cce_preprocess_input(joint, input_type);
    double log_base_scale = cce_get_log_base_scale(base);

    int64_t size_yt = joint_prob.size(-3);
    int64_t size_yprev = joint_prob.size(-2);
    int64_t size_xt = joint_prob.size(-1);

    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < ndim - 3; ++i) {
        batch_shape.push_back(joint_prob.size(i));
    }

    int64_t batch_size = 1;
    for (auto s : batch_shape) {
        batch_size *= s;
    }

    at::Tensor joint_t = joint_prob.contiguous().view({batch_size, size_yt, size_yprev, size_xt});

    at::Tensor grad_joint_t = at::zeros_like(joint_t);

    double scale = 1.0;
    if (reduction == "mean") {
        scale = 1.0 / static_cast<double>(batch_size);
    }

    at::Tensor grad_flat = grad_output.contiguous().view({-1});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        joint_prob.scalar_type(),
        "causally_conditioned_entropy_backward_cpu",
        [&]() {
            const scalar_t* joint_ptr = joint_t.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_flat.data_ptr<scalar_t>();
            scalar_t* grad_joint_ptr = grad_joint_t.data_ptr<scalar_t>();
            scalar_t log_scale = static_cast<scalar_t>(log_base_scale);
            scalar_t red_scale = static_cast<scalar_t>(scale);

            int64_t joint_stride = size_yt * size_yprev * size_xt;

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                std::vector<scalar_t> p_yprev_xt(size_yprev * size_xt);

                for (int64_t idx = begin; idx < end; ++idx) {
                    const scalar_t* batch_joint = joint_ptr + idx * joint_stride;
                    scalar_t* batch_grad = grad_joint_ptr + idx * joint_stride;

                    // Initialize marginals
                    std::fill(p_yprev_xt.begin(), p_yprev_xt.end(), scalar_t(0));

                    // Compute marginal p(y_prev, x_t)
                    for (int64_t yt = 0; yt < size_yt; ++yt) {
                        for (int64_t yp = 0; yp < size_yprev; ++yp) {
                            for (int64_t xt = 0; xt < size_xt; ++xt) {
                                scalar_t p = batch_joint[(yt * size_yprev + yp) * size_xt + xt];
                                p_yprev_xt[yp * size_xt + xt] += p;
                            }
                        }
                    }

                    scalar_t grad_val = (reduction == "none") ? grad_ptr[idx] : grad_ptr[0] * red_scale;

                    torchscience::kernel::information_theory::causally_conditioned_entropy_backward_kernel<scalar_t>(
                        grad_val,
                        batch_joint,
                        p_yprev_xt.data(),
                        size_yt,
                        size_yprev,
                        size_xt,
                        log_scale,
                        batch_grad
                    );
                }
            });
        }
    );

    // Reshape back to original shape
    std::vector<int64_t> output_shape = batch_shape;
    output_shape.push_back(size_yt);
    output_shape.push_back(size_yprev);
    output_shape.push_back(size_xt);

    return grad_joint_t.view(output_shape);
}

}  // namespace torchscience::cpu::information_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("causally_conditioned_entropy", &torchscience::cpu::information_theory::causally_conditioned_entropy);
    m.impl("causally_conditioned_entropy_backward", &torchscience::cpu::information_theory::causally_conditioned_entropy_backward);
}
