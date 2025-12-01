#pragma once

#include <cmath>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/macros/Macros.h>
#include <torch/library.h>

namespace torchscience::cpu::graphics::tone_mapping {

namespace {

// Basic reinhard: L_out = L_in / (1 + L_in)
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T reinhard_basic_kernel(T L_in) {
    return L_in / (T(1) + L_in);
}

// Extended reinhard: L_out = L_in * (1 + L_in/L_w^2) / (1 + L_in)
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T reinhard_extended_kernel(T L_in, T L_white_sq) {
    return L_in * (T(1) + L_in / L_white_sq) / (T(1) + L_in);
}

// Backward for basic reinhard
// d/dL_in [L_in / (1 + L_in)] = 1 / (1 + L_in)^2
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T reinhard_basic_backward_kernel(T grad_out, T L_in) {
    T denom = T(1) + L_in;
    return grad_out / (denom * denom);
}

// Backward for extended reinhard
// L_out = L_in * (1 + L_in/L_w^2) / (1 + L_in)
//       = L_in / (1 + L_in) + L_in^2 / (L_w^2 * (1 + L_in))
// d/dL_in = 1/(1+L_in)^2 + (2*L_in*(1+L_in) - L_in^2) / (L_w^2 * (1+L_in)^2)
//         = 1/(1+L_in)^2 * (1 + (2*L_in + L_in^2) / L_w^2)
//         = (1 + (L_in^2 + 2*L_in) / L_w^2) / (1+L_in)^2
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void reinhard_extended_backward_kernel(T grad_out, T L_in, T L_white_sq,
                                        T* grad_L_in, T* grad_L_white_sq) {
    T denom = T(1) + L_in;
    T denom_sq = denom * denom;

    // d/dL_in
    T numer = L_in * L_in + T(2) * L_in;
    *grad_L_in = grad_out * (T(1) + numer / L_white_sq) / denom_sq;

    // d/dL_white_sq = d/d(L_w^2) [L_in^2 / (L_w^2 * (1 + L_in))]
    //              = -L_in^2 / (L_w^4 * (1 + L_in))
    // Note: we return the gradient w.r.t. L_white (not L_white_sq)
    // d/dL_white = d/d(L_white_sq) * d(L_white_sq)/d(L_white)
    //            = (-L_in^2 / (L_w^4 * (1 + L_in))) * 2 * L_white
    //            = -2 * L_in^2 / (L_w^3 * (1 + L_in))
    // But since we pass L_white_sq, we need to compute d/d(L_white_sq) and then
    // the Python API will handle the chain rule for white_point
    // Actually, let's compute the gradient w.r.t. L_white directly
    T L_white = std::sqrt(L_white_sq);
    T L_white_cubed = L_white_sq * L_white;
    *grad_L_white_sq = grad_out * (-T(2) * L_in * L_in / (L_white_cubed * denom));
}

}  // namespace

inline at::Tensor reinhard(
    const at::Tensor& input,
    const std::optional<at::Tensor>& white_point
) {
    auto input_contig = input.contiguous();
    auto output = at::empty_like(input);
    int64_t num_elements = input.numel();

    if (!white_point.has_value()) {
        // Basic reinhard
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            input.scalar_type(), "reinhard_cpu", [&] {
                const scalar_t* input_ptr = input_contig.data_ptr<scalar_t>();
                scalar_t* output_ptr = output.data_ptr<scalar_t>();

                at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        output_ptr[i] = reinhard_basic_kernel(input_ptr[i]);
                    }
                });
            }
        );
    } else {
        // Extended reinhard
        auto wp = white_point.value().contiguous();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            input.scalar_type(), "reinhard_extended_cpu", [&] {
                const scalar_t* input_ptr = input_contig.data_ptr<scalar_t>();
                scalar_t L_white = wp.item<scalar_t>();
                scalar_t L_white_sq = L_white * L_white;
                scalar_t* output_ptr = output.data_ptr<scalar_t>();

                at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        output_ptr[i] = reinhard_extended_kernel(input_ptr[i], L_white_sq);
                    }
                });
            }
        );
    }

    return output;
}

inline std::tuple<at::Tensor, at::Tensor> reinhard_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const std::optional<at::Tensor>& white_point
) {
    auto grad_output_contig = grad_output.contiguous();
    auto input_contig = input.contiguous();
    auto grad_input = at::empty_like(input);
    int64_t num_elements = input.numel();

    at::Tensor grad_white_point;

    if (!white_point.has_value()) {
        // Basic reinhard backward
        grad_white_point = at::tensor(0.0, input.options());

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            input.scalar_type(), "reinhard_backward_cpu", [&] {
                const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
                const scalar_t* input_ptr = input_contig.data_ptr<scalar_t>();
                scalar_t* grad_input_ptr = grad_input.data_ptr<scalar_t>();

                at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        grad_input_ptr[i] = reinhard_basic_backward_kernel(
                            grad_output_ptr[i], input_ptr[i]
                        );
                    }
                });
            }
        );
    } else {
        // Extended reinhard backward
        auto wp = white_point.value().contiguous();
        grad_white_point = at::zeros({}, input.options());

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            input.scalar_type(), "reinhard_extended_backward_cpu", [&] {
                const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
                const scalar_t* input_ptr = input_contig.data_ptr<scalar_t>();
                scalar_t L_white = wp.item<scalar_t>();
                scalar_t L_white_sq = L_white * L_white;
                scalar_t* grad_input_ptr = grad_input.data_ptr<scalar_t>();

                scalar_t sum_grad_white = 0;

                // Serial loop for now since we accumulate grad_white_point
                for (int64_t i = 0; i < num_elements; ++i) {
                    scalar_t grad_L_in, grad_L_white;
                    reinhard_extended_backward_kernel(
                        grad_output_ptr[i], input_ptr[i], L_white_sq,
                        &grad_L_in, &grad_L_white
                    );
                    grad_input_ptr[i] = grad_L_in;
                    sum_grad_white += grad_L_white;
                }

                grad_white_point.fill_(sum_grad_white);
            }
        );
    }

    return std::make_tuple(grad_input, grad_white_point);
}

}  // namespace torchscience::cpu::graphics::tone_mapping

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("reinhard", &torchscience::cpu::graphics::tone_mapping::reinhard);
    m.impl("reinhard_backward", &torchscience::cpu::graphics::tone_mapping::reinhard_backward);
}
