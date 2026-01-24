#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/fft_ifft.h>
#include <ATen/ops/fft_fftfreq.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/real.h>
#include <ATen/ops/where.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/cos.h>
#include <torch/library.h>

namespace torchscience::cpu::transform {

/**
 * CPU implementation of inverse Radon transform using filtered back-projection.
 *
 * The filtered back-projection algorithm:
 * 1. Apply ramp filter in frequency domain to each projection
 * 2. Back-project the filtered projections onto the image plane
 *
 * @param sinogram Input sinogram of shape [..., num_angles, num_bins]
 * @param angles 1D tensor of projection angles in radians
 * @param circle If true, output is inscribed in a circle
 * @param output_size Target output size (H, W). If -1, computed from sinogram.
 * @param filter_type 0: "ramp", 1: "shepp-logan", 2: "cosine", 3: "hamming", 4: "hann"
 * @return Reconstructed image of shape [..., H, W]
 */
inline at::Tensor inverse_radon_transform(
    const at::Tensor& sinogram,
    const at::Tensor& angles,
    bool circle,
    int64_t output_size,
    int64_t filter_type
) {
    TORCH_CHECK(sinogram.dim() >= 2, "inverse_radon_transform: sinogram must be at least 2D");
    TORCH_CHECK(angles.dim() == 1, "inverse_radon_transform: angles must be 1D");

    int64_t ndim = sinogram.dim();
    int64_t num_angles = sinogram.size(-2);
    int64_t num_bins = sinogram.size(-1);

    TORCH_CHECK(
        angles.size(0) == num_angles,
        "inverse_radon_transform: angles size (", angles.size(0),
        ") must match sinogram angles dimension (", num_angles, ")"
    );

    at::Tensor sinogram_c = sinogram.contiguous();
    at::Tensor angles_c = angles.contiguous();

    // Determine output image size
    // By default, derive from sinogram size
    int64_t img_size = output_size;
    if (img_size <= 0) {
        // num_bins ~ sqrt(2) * img_size for square images
        img_size = static_cast<int64_t>(std::floor(num_bins / std::sqrt(2.0)));
        if (img_size < 1) img_size = 1;
    }
    int64_t H = img_size;
    int64_t W = img_size;

    // Build output shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim - 2; i++) {
        output_shape.push_back(sinogram_c.size(i));
    }
    output_shape.push_back(H);
    output_shape.push_back(W);

    // Flatten batch dimensions
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; i++) {
        batch_size *= sinogram_c.size(i);
    }
    at::Tensor sinogram_flat = sinogram_c.view({batch_size, num_angles, num_bins});

    // Create ramp filter in frequency domain
    // The ramp filter has transfer function |omega|
    int64_t n_fft = num_bins;
    // Pad to next power of 2 for FFT efficiency
    int64_t n_padded = 1;
    while (n_padded < n_fft) {
        n_padded *= 2;
    }
    n_padded *= 2;  // Extra padding to avoid circular convolution artifacts

    // Create frequency array
    at::Tensor freq = at::fft_fftfreq(n_padded, 1.0, sinogram.options());

    // Apply filter window based on filter_type
    // 0: ramp, 1: shepp-logan, 2: cosine, 3: hamming, 4: hann
    at::Tensor filter;
    switch (filter_type) {
        case 1: {
            // Shepp-Logan filter: |omega| * sinc(omega / (2 * omega_max))
            at::Tensor pi_freq = freq * M_PI;
            at::Tensor sinc_arg = pi_freq / 2.0;
            at::Tensor sinc_val = at::where(
                sinc_arg.abs() < 1e-10,
                at::ones_like(sinc_arg),
                at::sin(sinc_arg) / sinc_arg
            );
            filter = 2.0 * freq.abs() * sinc_val;
            break;
        }
        case 2: {
            // Cosine filter
            at::Tensor cos_weight = at::cos(freq * M_PI);
            filter = freq.abs() * cos_weight;
            break;
        }
        case 3: {
            // Hamming filter
            at::Tensor hamming = 0.54 + 0.46 * at::cos(freq * 2.0 * M_PI);
            filter = freq.abs() * hamming;
            break;
        }
        case 4: {
            // Hann filter
            at::Tensor hann = 0.5 * (1.0 + at::cos(freq * 2.0 * M_PI));
            filter = freq.abs() * hann;
            break;
        }
        default:
            // Ramp filter (default)
            filter = freq.abs();
            break;
    }

    // Scale filter appropriately
    filter = filter * (2.0 * M_PI / num_angles);

    // Apply filtering to each projection
    // Pad sinogram for FFT
    at::Tensor sino_padded = at::zeros(
        {batch_size, num_angles, n_padded},
        sinogram.options()
    );
    sino_padded.slice(2, 0, num_bins).copy_(sinogram_flat);

    // FFT along projection dimension
    at::Tensor sino_fft = at::fft_fft(sino_padded, c10::nullopt, -1);

    // Apply filter
    at::Tensor filter_expanded = filter.unsqueeze(0).unsqueeze(0);
    at::Tensor sino_filtered_fft = sino_fft * filter_expanded;

    // Inverse FFT
    at::Tensor sino_filtered = at::real(at::fft_ifft(sino_filtered_fft, c10::nullopt, -1));

    // Extract valid portion
    sino_filtered = sino_filtered.slice(2, 0, num_bins);

    // Create output tensor
    at::Tensor output = at::zeros(output_shape, sinogram.options());
    at::Tensor output_flat = output.view({batch_size, H, W});

    // Center of the output image
    double center_x = (W - 1.0) / 2.0;
    double center_y = (H - 1.0) / 2.0;

    // Center of sinogram bins
    double detector_half = (num_bins - 1.0) / 2.0;

    // For circle mode
    double radius = circle ? std::min(center_x, center_y) : std::sqrt(center_x * center_x + center_y * center_y);

    // Back-projection
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        sinogram.scalar_type(), "inverse_radon_transform", [&] {
            auto output_acc = output_flat.accessor<scalar_t, 3>();
            auto sino_acc = sino_filtered.accessor<scalar_t, 3>();
            auto angles_acc = angles_c.accessor<scalar_t, 1>();

            for (int64_t a = 0; a < num_angles; a++) {
                scalar_t theta = angles_acc[a];
                double cos_theta = std::cos(static_cast<double>(theta));
                double sin_theta = std::sin(static_cast<double>(theta));

                for (int64_t b = 0; b < batch_size; b++) {
                    for (int64_t y = 0; y < H; y++) {
                        for (int64_t x = 0; x < W; x++) {
                            // Check circle constraint
                            double dx = x - center_x;
                            double dy = y - center_y;
                            if (circle && (dx * dx + dy * dy > radius * radius)) {
                                continue;
                            }

                            // Compute projection coordinate
                            // s = x * cos(theta) + y * sin(theta)
                            double s = dx * cos_theta + dy * sin_theta;

                            // Map to bin index
                            double bin_idx = s + detector_half;

                            // Bilinear interpolation from filtered sinogram
                            if (bin_idx >= 0 && bin_idx < num_bins - 1) {
                                int64_t bin0 = static_cast<int64_t>(bin_idx);
                                int64_t bin1 = bin0 + 1;
                                double frac = bin_idx - bin0;

                                scalar_t val = static_cast<scalar_t>(
                                    (1.0 - frac) * sino_acc[b][a][bin0] +
                                    frac * sino_acc[b][a][bin1]
                                );
                                output_acc[b][y][x] += val;
                            }
                        }
                    }
                }
            }
        }
    );

    return output;
}

/**
 * Backward pass for inverse Radon transform.
 * Gradient flows through the back-projection and filtering operations.
 */
inline at::Tensor inverse_radon_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& sinogram,
    const at::Tensor& angles,
    bool circle,
    int64_t output_size,
    int64_t filter_type
) {
    int64_t ndim = sinogram.dim();
    int64_t num_angles = sinogram.size(-2);
    int64_t num_bins = sinogram.size(-1);

    at::Tensor grad_output_c = grad_output.contiguous();
    at::Tensor angles_c = angles.contiguous();

    int64_t H = grad_output.size(-2);
    int64_t W = grad_output.size(-1);

    // Flatten batch dimensions
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; i++) {
        batch_size *= sinogram.size(i);
    }
    at::Tensor grad_output_flat = grad_output_c.view({batch_size, H, W});

    double center_x = (W - 1.0) / 2.0;
    double center_y = (H - 1.0) / 2.0;
    double detector_half = (num_bins - 1.0) / 2.0;
    double radius = circle ? std::min(center_x, center_y) : std::sqrt(center_x * center_x + center_y * center_y);

    // Create gradient for filtered sinogram (pre-filter)
    at::Tensor grad_sino_filtered = at::zeros(
        {batch_size, num_angles, num_bins},
        sinogram.options()
    );

    // Back-project gradients (transpose of back-projection)
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        sinogram.scalar_type(), "inverse_radon_transform_backward", [&] {
            auto grad_output_acc = grad_output_flat.accessor<scalar_t, 3>();
            auto grad_sino_acc = grad_sino_filtered.accessor<scalar_t, 3>();
            auto angles_acc = angles_c.accessor<scalar_t, 1>();

            for (int64_t a = 0; a < num_angles; a++) {
                scalar_t theta = angles_acc[a];
                double cos_theta = std::cos(static_cast<double>(theta));
                double sin_theta = std::sin(static_cast<double>(theta));

                for (int64_t b = 0; b < batch_size; b++) {
                    for (int64_t y = 0; y < H; y++) {
                        for (int64_t x = 0; x < W; x++) {
                            double dx = x - center_x;
                            double dy = y - center_y;
                            if (circle && (dx * dx + dy * dy > radius * radius)) {
                                continue;
                            }

                            double s = dx * cos_theta + dy * sin_theta;
                            double bin_idx = s + detector_half;

                            if (bin_idx >= 0 && bin_idx < num_bins - 1) {
                                int64_t bin0 = static_cast<int64_t>(bin_idx);
                                int64_t bin1 = bin0 + 1;
                                double frac = bin_idx - bin0;

                                scalar_t grad_val = grad_output_acc[b][y][x];
                                grad_sino_acc[b][a][bin0] += static_cast<scalar_t>((1.0 - frac)) * grad_val;
                                grad_sino_acc[b][a][bin1] += static_cast<scalar_t>(frac) * grad_val;
                            }
                        }
                    }
                }
            }
        }
    );

    // Now we need to backprop through the filtering operation
    // The filter is applied in frequency domain: sino_filtered = IFFT(FFT(sino_padded) * filter)
    // Gradient: grad_sino = IFFT(FFT(grad_sino_filtered) * filter)
    // Since the filter is real and symmetric, its gradient is the same

    int64_t n_padded = 1;
    while (n_padded < num_bins) {
        n_padded *= 2;
    }
    n_padded *= 2;

    // Create the same filter
    at::Tensor freq = at::fft_fftfreq(n_padded, 1.0, sinogram.options());
    at::Tensor filter;
    switch (filter_type) {
        case 1: {
            at::Tensor pi_freq = freq * M_PI;
            at::Tensor sinc_arg = pi_freq / 2.0;
            at::Tensor sinc_val = at::where(
                sinc_arg.abs() < 1e-10,
                at::ones_like(sinc_arg),
                at::sin(sinc_arg) / sinc_arg
            );
            filter = 2.0 * freq.abs() * sinc_val;
            break;
        }
        case 2: {
            at::Tensor cos_weight = at::cos(freq * M_PI);
            filter = freq.abs() * cos_weight;
            break;
        }
        case 3: {
            at::Tensor hamming = 0.54 + 0.46 * at::cos(freq * 2.0 * M_PI);
            filter = freq.abs() * hamming;
            break;
        }
        case 4: {
            at::Tensor hann = 0.5 * (1.0 + at::cos(freq * 2.0 * M_PI));
            filter = freq.abs() * hann;
            break;
        }
        default:
            filter = freq.abs();
            break;
    }
    filter = filter * (2.0 * M_PI / num_angles);

    // Pad gradient sinogram
    at::Tensor grad_sino_padded = at::zeros(
        {batch_size, num_angles, n_padded},
        sinogram.options()
    );
    grad_sino_padded.slice(2, 0, num_bins).copy_(grad_sino_filtered);

    // Apply filter in frequency domain (same filter for backward)
    at::Tensor grad_sino_fft = at::fft_fft(grad_sino_padded, c10::nullopt, -1);
    at::Tensor filter_expanded = filter.unsqueeze(0).unsqueeze(0);
    at::Tensor grad_sino_filtered_fft = grad_sino_fft * filter_expanded;
    at::Tensor grad_sinogram_full = at::real(at::fft_ifft(grad_sino_filtered_fft, c10::nullopt, -1));

    // Extract and reshape
    at::Tensor grad_sinogram = grad_sinogram_full.slice(2, 0, num_bins);

    // Reshape to original batch shape
    std::vector<int64_t> grad_shape;
    for (int64_t i = 0; i < ndim - 2; i++) {
        grad_shape.push_back(sinogram.size(i));
    }
    grad_shape.push_back(num_angles);
    grad_shape.push_back(num_bins);

    return grad_sinogram.view(grad_shape);
}

/**
 * Double backward pass for inverse Radon transform.
 */
inline std::tuple<at::Tensor, at::Tensor>
inverse_radon_transform_backward_backward(
    const at::Tensor& grad_grad_sinogram,
    const at::Tensor& grad_output,
    const at::Tensor& sinogram,
    const at::Tensor& angles,
    bool circle,
    int64_t output_size,
    int64_t filter_type
) {
    at::Tensor grad_grad_output = at::Tensor();

    if (grad_grad_sinogram.defined()) {
        // Second derivative is the forward operation again
        grad_grad_output = inverse_radon_transform(
            grad_grad_sinogram, angles, circle, output_size, filter_type
        );
    }

    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "inverse_radon_transform",
        &torchscience::cpu::transform::inverse_radon_transform
    );

    module.impl(
        "inverse_radon_transform_backward",
        &torchscience::cpu::transform::inverse_radon_transform_backward
    );

    module.impl(
        "inverse_radon_transform_backward_backward",
        &torchscience::cpu::transform::inverse_radon_transform_backward_backward
    );
}
