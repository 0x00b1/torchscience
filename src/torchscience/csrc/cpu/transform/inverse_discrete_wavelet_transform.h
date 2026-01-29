#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/conv_transpose1d.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/flip.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/zeros.h>
#include <torch/library.h>

namespace torchscience::cpu::transform {

/**
 * Single-level inverse DWT (synthesis).
 *
 * @param approx Approximation coefficients
 * @param detail Detail coefficients
 * @param filter_lo Reconstruction lowpass filter
 * @param filter_hi Reconstruction highpass filter
 * @param output_length Target output length
 * @return Reconstructed signal
 */
inline at::Tensor idwt_single_level(
    const at::Tensor& approx,
    const at::Tensor& detail,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t output_length
) {
    int64_t filter_len = filter_lo.size(0);

    // Get batch shape
    std::vector<int64_t> original_shape(approx.sizes().begin(), approx.sizes().end());
    int64_t ndim = approx.dim();
    int64_t coeff_len = approx.size(-1);

    at::Tensor approx_conv, detail_conv;
    std::vector<int64_t> batch_shape;

    if (ndim == 1) {
        approx_conv = approx.unsqueeze(0).unsqueeze(0);  // (1, 1, N)
        detail_conv = detail.unsqueeze(0).unsqueeze(0);
    } else {
        int64_t batch_numel = 1;
        for (int64_t i = 0; i < ndim - 1; i++) {
            batch_numel *= original_shape[i];
            batch_shape.push_back(original_shape[i]);
        }
        approx_conv = approx.reshape({batch_numel, 1, coeff_len});
        detail_conv = detail.reshape({batch_numel, 1, coeff_len});
    }

    // Prepare filters for transposed convolution (synthesis)
    // Shape: (in_channels, out_channels/groups, kernel_size)
    at::Tensor lo_kernel = filter_lo.reshape({1, 1, -1});
    at::Tensor hi_kernel = filter_hi.reshape({1, 1, -1});

    // Upsample + filter via transposed convolution with stride 2
    at::Tensor approx_up = at::conv_transpose1d(approx_conv, lo_kernel, {}, 2);
    at::Tensor detail_up = at::conv_transpose1d(detail_conv, hi_kernel, {}, 2);

    // Sum the contributions
    at::Tensor result = approx_up + detail_up;

    // Trim to target output length (remove filter delay)
    int64_t current_len = result.size(-1);
    int64_t trim_total = current_len - output_length;
    int64_t trim_left = (filter_len - 1) / 2;
    int64_t trim_right = trim_total - trim_left;

    if (trim_left > 0 || trim_right > 0) {
        int64_t start = trim_left;
        int64_t length = output_length;
        if (start + length > current_len) {
            length = current_len - start;
        }
        result = result.narrow(-1, start, length);
    }

    // Reshape back
    if (ndim == 1) {
        result = result.squeeze(0).squeeze(0);
    } else {
        std::vector<int64_t> out_shape = batch_shape;
        out_shape.push_back(result.size(-1));
        result = result.squeeze(1).reshape(out_shape);
    }

    return result;
}

// dwt_coeff_len is defined in discrete_wavelet_transform.h which is included first

/**
 * Compute coefficient lengths for each DWT level.
 *
 * The DWT pads the signal, convolves, then downsamples by 2.
 * The output length depends on the padding mode and filter length.
 */
inline std::vector<int64_t> compute_coeff_lengths(
    int64_t input_length,
    int64_t filter_len,
    int64_t levels,
    int64_t mode = 0
) {
    std::vector<int64_t> lengths;
    int64_t current_len = input_length;

    for (int64_t i = 0; i < levels; i++) {
        int64_t coeff_len = dwt_coeff_len(current_len, filter_len, mode);
        lengths.push_back(coeff_len);
        current_len = coeff_len;
    }

    return lengths;
}

/**
 * CPU implementation of inverse discrete wavelet transform.
 *
 * @param coeffs Packed coefficients [cA_n | cD_n | cD_{n-1} | ... | cD_1]
 * @param filter_lo Reconstruction lowpass filter
 * @param filter_hi Reconstruction highpass filter
 * @param levels Number of decomposition levels
 * @param mode Padding mode (unused in reconstruction, kept for API consistency)
 * @param output_length Target output signal length
 * @return Reconstructed signal
 */
inline at::Tensor inverse_discrete_wavelet_transform(
    const at::Tensor& coeffs,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode,
    int64_t output_length
) {
    TORCH_CHECK(coeffs.numel() > 0, "inverse_discrete_wavelet_transform: coeffs must be non-empty");
    TORCH_CHECK(filter_lo.dim() == 1, "inverse_discrete_wavelet_transform: filter_lo must be 1D");
    TORCH_CHECK(filter_hi.dim() == 1, "inverse_discrete_wavelet_transform: filter_hi must be 1D");
    TORCH_CHECK(levels >= 1, "inverse_discrete_wavelet_transform: levels must be >= 1");

    int64_t filter_len = filter_lo.size(0);

    // Compute expected coefficient lengths at each level
    std::vector<int64_t> coeff_lens = compute_coeff_lengths(output_length, filter_len, levels, mode);

    // Unpack coefficients: [cA_n | cD_n | cD_{n-1} | ... | cD_1]
    int64_t total_len = coeffs.size(-1);

    // Final approx length is coeff_lens[levels-1]
    int64_t approx_len = coeff_lens[levels - 1];
    at::Tensor approx = coeffs.narrow(-1, 0, approx_len);

    // Extract detail coefficients (stored in reverse order: cD_n, cD_{n-1}, ..., cD_1)
    std::vector<at::Tensor> details;
    int64_t offset = approx_len;

    for (int64_t i = levels - 1; i >= 0; i--) {
        int64_t detail_len = coeff_lens[i];
        at::Tensor detail = coeffs.narrow(-1, offset, detail_len);
        details.push_back(detail);
        offset += detail_len;
    }

    // Reconstruct level by level (from coarsest to finest)
    at::Tensor reconstructed = approx;

    for (int64_t i = 0; i < levels; i++) {
        // Target length for this level
        int64_t target_len;
        if (i == levels - 1) {
            target_len = output_length;
        } else {
            target_len = coeff_lens[levels - 2 - i];
        }

        reconstructed = idwt_single_level(
            reconstructed, details[i], filter_lo, filter_hi, target_len
        );
    }

    return reconstructed;
}

/**
 * Backward pass for inverse DWT.
 *
 * The backward of inverse DWT is the forward DWT with transposed filters.
 */
inline at::Tensor inverse_discrete_wavelet_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode,
    int64_t output_length
) {
    // Backward of inverse DWT is forward DWT with time-reversed filters
    at::Tensor dec_lo = at::flip(filter_lo, {0});
    at::Tensor dec_hi = at::flip(filter_hi, {0});

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::discrete_wavelet_transform", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t)>()
        .call(grad_output, dec_lo, dec_hi, levels, mode);
}

/**
 * Double backward pass for inverse DWT.
 */
inline std::tuple<at::Tensor, at::Tensor> inverse_discrete_wavelet_transform_backward_backward(
    const at::Tensor& grad_grad_coeffs,
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode,
    int64_t output_length
) {
    at::Tensor grad_grad_output = at::Tensor();

    if (grad_grad_coeffs.defined()) {
        grad_grad_output = inverse_discrete_wavelet_transform(
            grad_grad_coeffs, filter_lo, filter_hi, levels, mode, output_length
        );
    }

    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "inverse_discrete_wavelet_transform",
        &torchscience::cpu::transform::inverse_discrete_wavelet_transform
    );

    module.impl(
        "inverse_discrete_wavelet_transform_backward",
        &torchscience::cpu::transform::inverse_discrete_wavelet_transform_backward
    );

    module.impl(
        "inverse_discrete_wavelet_transform_backward_backward",
        &torchscience::cpu::transform::inverse_discrete_wavelet_transform_backward_backward
    );
}
