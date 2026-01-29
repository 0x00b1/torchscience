#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/conv1d.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/flip.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/zeros.h>
#include <torch/library.h>

namespace torchscience::cpu::transform {

// Padding mode enum: 0=symmetric, 1=reflect, 2=periodic, 3=zero
//
// Symmetric extension (mode 0): reflects including boundary
//   [a, b, c, d] with pad_left=2 -> [b, a, a, b, c, d]
//   This matches PyWavelets' 'symmetric' mode
//
// Reflect extension (mode 1): reflects excluding boundary
//   [a, b, c, d] with pad_left=2 -> [c, b, a, b, c, d]
//   This matches PyTorch's 'reflect' mode

inline at::Tensor symmetric_pad_1d(
    const at::Tensor& input,
    int64_t pad_left,
    int64_t pad_right
) {
    // Symmetric extension includes the boundary point
    // For signal [a, b, c, d]:
    //   Left symmetric pad by 2: [b, a | a, b, c, d]
    //   Right symmetric pad by 2: [a, b, c, d | d, c]

    int64_t n = input.size(-1);
    std::vector<at::Tensor> parts;

    // Left padding: take first pad_left elements and flip
    if (pad_left > 0) {
        // Handle case where pad_left > n by repeating
        int64_t remaining = pad_left;
        std::vector<at::Tensor> left_parts;

        while (remaining > 0) {
            int64_t take = std::min(remaining, n);
            at::Tensor chunk = input.narrow(-1, 0, take);
            chunk = at::flip(chunk, {-1});
            left_parts.insert(left_parts.begin(), chunk);
            remaining -= take;
        }

        for (auto& p : left_parts) {
            parts.push_back(p);
        }
    }

    // Original signal
    parts.push_back(input);

    // Right padding: take last pad_right elements and flip
    if (pad_right > 0) {
        int64_t remaining = pad_right;

        while (remaining > 0) {
            int64_t take = std::min(remaining, n);
            at::Tensor chunk = input.narrow(-1, n - take, take);
            chunk = at::flip(chunk, {-1});
            parts.push_back(chunk);
            remaining -= take;
        }
    }

    return at::cat(parts, -1);
}

inline at::Tensor pad_for_dwt(
    const at::Tensor& input,
    int64_t pad_left,
    int64_t pad_right,
    int64_t mode
) {
    if (pad_left == 0 && pad_right == 0) {
        return input;
    }

    if (mode == 0) {  // symmetric (PyWavelets-compatible)
        return symmetric_pad_1d(input, pad_left, pad_right);
    } else if (mode == 1) {  // reflect (PyTorch-style)
        return at::pad(input, {pad_left, pad_right}, "reflect");
    } else if (mode == 2) {  // periodic
        return at::pad(input, {pad_left, pad_right}, "circular");
    } else {  // zero
        return at::pad(input, {pad_left, pad_right}, "constant");
    }
}

// Compute DWT output length matching PyWavelets behavior
// For symmetric/reflect modes: floor((input_len + filter_len - 1) / 2)
inline int64_t dwt_coeff_len(int64_t input_len, int64_t filter_len, int64_t mode) {
    if (mode == 2) {  // periodic
        return (input_len + 1) / 2;
    } else {
        // symmetric, reflect, zero: include boundary effects
        return (input_len + filter_len - 1) / 2;
    }
}

inline at::Tensor dwt_single_level(
    const at::Tensor& x,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t mode
) {
    int64_t filter_len = filter_lo.size(0);
    int64_t signal_len = x.size(-1);

    // Prepare input for conv1d: need shape (batch, channels, length)
    std::vector<int64_t> original_shape(x.sizes().begin(), x.sizes().end());
    int64_t ndim = x.dim();

    at::Tensor x_conv;
    std::vector<int64_t> batch_shape;

    if (ndim == 1) {
        x_conv = x.unsqueeze(0).unsqueeze(0);  // (1, 1, N)
    } else {
        // Flatten all batch dimensions
        int64_t batch_numel = 1;
        for (int64_t i = 0; i < ndim - 1; i++) {
            batch_numel *= original_shape[i];
            batch_shape.push_back(original_shape[i]);
        }
        x_conv = x.reshape({batch_numel, 1, signal_len});
    }

    // Compute output coefficient length
    int64_t out_len = dwt_coeff_len(signal_len, filter_len, mode);

    // Pad signal for convolution - enough for the desired output length after downsampling
    // After convolution (valid mode) and downsampling, we need 2*out_len samples
    // So padded length must be: 2*out_len + filter_len - 1
    int64_t needed_len = 2 * out_len + filter_len - 1;
    int64_t pad_total = needed_len - signal_len;
    int64_t pad_left = (filter_len - 1) / 2;
    int64_t pad_right = pad_total - pad_left;

    // Ensure non-negative padding
    if (pad_left < 0) pad_left = 0;
    if (pad_right < 0) pad_right = 0;

    at::Tensor x_padded = pad_for_dwt(x_conv, pad_left, pad_right, mode);

    // Prepare filters: shape (out_channels, in_channels/groups, kernel_size)
    // Flip filters for convolution (conv1d does cross-correlation)
    at::Tensor lo_kernel = at::flip(filter_lo, {0}).reshape({1, 1, -1});
    at::Tensor hi_kernel = at::flip(filter_hi, {0}).reshape({1, 1, -1});

    // Convolve (valid mode - no padding in conv1d)
    at::Tensor approx_full = at::conv1d(x_padded, lo_kernel);
    at::Tensor detail_full = at::conv1d(x_padded, hi_kernel);

    // Downsample by 2, taking first out_len coefficients
    at::Tensor approx = approx_full.slice(2, 0, 2 * out_len, 2);
    at::Tensor detail = detail_full.slice(2, 0, 2 * out_len, 2);

    // Reshape back to original batch shape
    if (ndim == 1) {
        approx = approx.squeeze(0).squeeze(0);
        detail = detail.squeeze(0).squeeze(0);
    } else {
        std::vector<int64_t> out_shape = batch_shape;
        out_shape.push_back(out_len);
        approx = approx.squeeze(1).reshape(out_shape);
        detail = detail.squeeze(1).reshape(out_shape);
    }

    // Pack approx and detail into single tensor: [approx | detail]
    return at::cat({approx, detail}, -1);
}

/**
 * CPU implementation of discrete wavelet transform.
 *
 * @param input Input tensor with signal in last dimension
 * @param filter_lo Lowpass decomposition filter
 * @param filter_hi Highpass decomposition filter
 * @param levels Number of decomposition levels
 * @param mode Padding mode (0=symmetric, 1=reflect, 2=periodic, 3=zero)
 * @return Packed coefficients tensor [cA_n | cD_n | cD_{n-1} | ... | cD_1]
 */
inline at::Tensor discrete_wavelet_transform(
    const at::Tensor& input,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode
) {
    TORCH_CHECK(input.numel() > 0, "discrete_wavelet_transform: input must be non-empty");
    TORCH_CHECK(filter_lo.dim() == 1, "discrete_wavelet_transform: filter_lo must be 1D");
    TORCH_CHECK(filter_hi.dim() == 1, "discrete_wavelet_transform: filter_hi must be 1D");
    TORCH_CHECK(filter_lo.size(0) == filter_hi.size(0), "discrete_wavelet_transform: filters must have same length");
    TORCH_CHECK(levels >= 1, "discrete_wavelet_transform: levels must be >= 1");

    at::Tensor approx = input.contiguous();
    std::vector<at::Tensor> details;

    for (int64_t level = 0; level < levels; level++) {
        at::Tensor packed = dwt_single_level(approx, filter_lo, filter_hi, mode);

        // Unpack: first half is approx, second half is detail
        int64_t coeff_len = packed.size(-1) / 2;
        approx = packed.narrow(-1, 0, coeff_len);
        at::Tensor detail = packed.narrow(-1, coeff_len, coeff_len);

        details.push_back(detail);
    }

    // Pack all coefficients: [cA_n | cD_n | cD_{n-1} | ... | cD_1]
    std::vector<at::Tensor> all_coeffs;
    all_coeffs.push_back(approx);
    for (auto it = details.rbegin(); it != details.rend(); ++it) {
        all_coeffs.push_back(*it);
    }

    return at::cat(all_coeffs, -1);
}

/**
 * Backward pass for discrete wavelet transform.
 *
 * The backward of DWT is essentially the inverse DWT with transposed filters.
 */
inline at::Tensor discrete_wavelet_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode,
    int64_t input_length
) {
    // Forward declaration - implemented via inverse DWT
    // The backward of DWT is the inverse DWT with time-reversed filters
    at::Tensor rec_lo = at::flip(filter_lo, {0});
    at::Tensor rec_hi = at::flip(filter_hi, {0});

    // Call inverse DWT dispatcher
    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::inverse_discrete_wavelet_transform", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t)>()
        .call(grad_output, rec_lo, rec_hi, levels, mode, input_length);
}

/**
 * Double backward pass for discrete wavelet transform.
 */
inline std::tuple<at::Tensor, at::Tensor> discrete_wavelet_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode,
    int64_t input_length
) {
    // Second-order gradient: apply DWT to grad_grad_input
    at::Tensor grad_grad_output = at::Tensor();

    if (grad_grad_input.defined()) {
        grad_grad_output = discrete_wavelet_transform(
            grad_grad_input, filter_lo, filter_hi, levels, mode
        );
    }

    // No gradient w.r.t. input from second backward (DWT is linear)
    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "discrete_wavelet_transform",
        &torchscience::cpu::transform::discrete_wavelet_transform
    );

    module.impl(
        "discrete_wavelet_transform_backward",
        &torchscience::cpu::transform::discrete_wavelet_transform_backward
    );

    module.impl(
        "discrete_wavelet_transform_backward_backward",
        &torchscience::cpu::transform::discrete_wavelet_transform_backward_backward
    );
}
