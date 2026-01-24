#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/linspace.h>
#include <torch/library.h>

namespace torchscience::cpu::transform {

/**
 * CPU implementation of Radon transform.
 *
 * Computes line integrals through a 2D image at various angles.
 * R(theta, s) = integral f(x,y) delta(x*cos(theta) + y*sin(theta) - s) dx dy
 *
 * @param input 2D input image tensor [H, W] or batched [..., H, W]
 * @param angles 1D tensor of projection angles in radians
 * @param circle If true, inscribes image in a circle to avoid corner artifacts
 * @return Sinogram of shape [..., num_angles, num_detector_bins]
 */
inline at::Tensor radon_transform(
    const at::Tensor& input,
    const at::Tensor& angles,
    bool circle
) {
    TORCH_CHECK(input.dim() >= 2, "radon_transform: input must be at least 2D");
    TORCH_CHECK(angles.dim() == 1, "radon_transform: angles must be 1D");

    // Get input dimensions
    int64_t ndim = input.dim();
    int64_t H = input.size(-2);
    int64_t W = input.size(-1);
    int64_t num_angles = angles.size(0);

    // Ensure contiguous
    at::Tensor input_c = input.contiguous();
    at::Tensor angles_c = angles.contiguous();

    // Number of detector bins (projection width)
    // For a square image, the maximum projection length is the diagonal
    int64_t num_bins = static_cast<int64_t>(std::ceil(std::sqrt(H * H + W * W)));
    if (num_bins % 2 == 0) {
        num_bins += 1;  // Make odd for symmetry
    }

    // Center of the image
    double center_x = (W - 1.0) / 2.0;
    double center_y = (H - 1.0) / 2.0;

    // Create output tensor
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim - 2; i++) {
        output_shape.push_back(input_c.size(i));
    }
    output_shape.push_back(num_angles);
    output_shape.push_back(num_bins);

    at::Tensor output = at::zeros(output_shape, input_c.options());

    // Flatten batch dimensions
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; i++) {
        batch_size *= input_c.size(i);
    }
    at::Tensor input_flat = input_c.view({batch_size, H, W});
    at::Tensor output_flat = output.view({batch_size, num_angles, num_bins});

    // Compute detector positions (centered at 0)
    double detector_half = (num_bins - 1.0) / 2.0;

    // For circle mode, compute the inscribed circle radius
    double radius = circle ? std::min(center_x, center_y) : std::sqrt(center_x * center_x + center_y * center_y);

    // Process each angle
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input_c.scalar_type(), "radon_transform", [&] {
            auto input_acc = input_flat.accessor<scalar_t, 3>();
            auto output_acc = output_flat.accessor<scalar_t, 3>();
            auto angles_acc = angles_c.accessor<scalar_t, 1>();

            for (int64_t a = 0; a < num_angles; a++) {
                scalar_t theta = angles_acc[a];
                scalar_t cos_theta = std::cos(static_cast<double>(theta));
                scalar_t sin_theta = std::sin(static_cast<double>(theta));

                for (int64_t b = 0; b < batch_size; b++) {
                    // For each detector bin, compute the line integral
                    for (int64_t d = 0; d < num_bins; d++) {
                        // Offset from center of projection
                        double s = (d - detector_half);

                        // Sample along the line x*cos(theta) + y*sin(theta) = s
                        // Parameterize as (x, y) = (s*cos(theta), s*sin(theta)) + t*(-sin(theta), cos(theta))
                        scalar_t sum = 0;
                        int64_t count = 0;

                        // Number of samples along the line
                        int64_t num_samples = static_cast<int64_t>(std::max(H, W) * 2);

                        for (int64_t t_idx = 0; t_idx < num_samples; t_idx++) {
                            double t = (t_idx - num_samples / 2.0);

                            // Point on the line
                            double x = s * cos_theta - t * sin_theta + center_x;
                            double y = s * sin_theta + t * cos_theta + center_y;

                            // Bilinear interpolation
                            if (x >= 0 && x < W - 1 && y >= 0 && y < H - 1) {
                                int64_t x0 = static_cast<int64_t>(x);
                                int64_t y0 = static_cast<int64_t>(y);
                                int64_t x1 = x0 + 1;
                                int64_t y1 = y0 + 1;

                                double fx = x - x0;
                                double fy = y - y0;

                                // Check circle constraint
                                if (!circle || ((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y) <= radius * radius)) {
                                    scalar_t val = static_cast<scalar_t>(
                                        (1 - fx) * (1 - fy) * input_acc[b][y0][x0] +
                                        fx * (1 - fy) * input_acc[b][y0][x1] +
                                        (1 - fx) * fy * input_acc[b][y1][x0] +
                                        fx * fy * input_acc[b][y1][x1]
                                    );
                                    sum += val;
                                    count++;
                                }
                            }
                        }

                        // Normalize by the step size along the line (approximately 1 pixel)
                        if (count > 0) {
                            output_acc[b][a][d] = sum;
                        }
                    }
                }
            }
        }
    );

    return output;
}

/**
 * Backward pass for Radon transform.
 * This is essentially a back-projection operation.
 */
inline at::Tensor radon_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& angles,
    bool circle
) {
    int64_t ndim = input.dim();
    int64_t H = input.size(-2);
    int64_t W = input.size(-1);
    int64_t num_angles = angles.size(0);
    int64_t num_bins = grad_output.size(-1);

    at::Tensor grad_output_c = grad_output.contiguous();
    at::Tensor angles_c = angles.contiguous();

    double center_x = (W - 1.0) / 2.0;
    double center_y = (H - 1.0) / 2.0;
    double detector_half = (num_bins - 1.0) / 2.0;
    double radius = circle ? std::min(center_x, center_y) : std::sqrt(center_x * center_x + center_y * center_y);

    at::Tensor grad_input = at::zeros_like(input);

    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; i++) {
        batch_size *= input.size(i);
    }
    at::Tensor grad_input_flat = grad_input.view({batch_size, H, W});
    at::Tensor grad_output_flat = grad_output_c.view({batch_size, num_angles, num_bins});

    // Back-project: distribute gradients back to the image
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "radon_transform_backward", [&] {
            auto grad_input_acc = grad_input_flat.accessor<scalar_t, 3>();
            auto grad_output_acc = grad_output_flat.accessor<scalar_t, 3>();
            auto angles_acc = angles_c.accessor<scalar_t, 1>();

            for (int64_t a = 0; a < num_angles; a++) {
                scalar_t theta = angles_acc[a];
                scalar_t cos_theta = std::cos(static_cast<double>(theta));
                scalar_t sin_theta = std::sin(static_cast<double>(theta));

                for (int64_t b = 0; b < batch_size; b++) {
                    // Back-project each detector bin
                    for (int64_t d = 0; d < num_bins; d++) {
                        scalar_t grad_val = grad_output_acc[b][a][d];
                        if (grad_val == 0) continue;

                        double s = (d - detector_half);
                        int64_t num_samples = static_cast<int64_t>(std::max(H, W) * 2);

                        for (int64_t t_idx = 0; t_idx < num_samples; t_idx++) {
                            double t = (t_idx - num_samples / 2.0);

                            double x = s * cos_theta - t * sin_theta + center_x;
                            double y = s * sin_theta + t * cos_theta + center_y;

                            if (x >= 0 && x < W - 1 && y >= 0 && y < H - 1) {
                                int64_t x0 = static_cast<int64_t>(x);
                                int64_t y0 = static_cast<int64_t>(y);
                                int64_t x1 = x0 + 1;
                                int64_t y1 = y0 + 1;

                                double fx = x - x0;
                                double fy = y - y0;

                                if (!circle || ((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y) <= radius * radius)) {
                                    // Distribute gradient using bilinear weights
                                    grad_input_acc[b][y0][x0] += static_cast<scalar_t>((1 - fx) * (1 - fy)) * grad_val;
                                    grad_input_acc[b][y0][x1] += static_cast<scalar_t>(fx * (1 - fy)) * grad_val;
                                    grad_input_acc[b][y1][x0] += static_cast<scalar_t>((1 - fx) * fy) * grad_val;
                                    grad_input_acc[b][y1][x1] += static_cast<scalar_t>(fx * fy) * grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    );

    return grad_input;
}

/**
 * Double backward pass for Radon transform.
 */
inline std::tuple<at::Tensor, at::Tensor>
radon_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& angles,
    bool circle
) {
    at::Tensor grad_grad_output = at::Tensor();

    if (grad_grad_input.defined()) {
        grad_grad_output = radon_transform(grad_grad_input, angles, circle);
    }

    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "radon_transform",
        &torchscience::cpu::transform::radon_transform
    );

    module.impl(
        "radon_transform_backward",
        &torchscience::cpu::transform::radon_transform_backward
    );

    module.impl(
        "radon_transform_backward_backward",
        &torchscience::cpu::transform::radon_transform_backward_backward
    );
}
