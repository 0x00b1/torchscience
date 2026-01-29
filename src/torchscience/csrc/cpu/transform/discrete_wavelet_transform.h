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
inline at::Tensor pad_for_dwt(
    const at::Tensor& input,
    int64_t pad_left,
    int64_t pad_right,
    int64_t mode
) {
    if (pad_left == 0 && pad_right == 0) {
        return input;
    }

    std::string pad_mode;
    if (mode == 0 || mode == 1) {  // symmetric or reflect
        pad_mode = "reflect";
    } else if (mode == 2) {  // periodic
        pad_mode = "circular";
    } else {  // zero
        pad_mode = "constant";
    }

    // F.pad expects (left, right) for 1D, applied to last dim
    return at::pad(input, {pad_left, pad_right}, pad_mode);
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

    // Pad signal for convolution
    int64_t pad_total = filter_len - 1;
    int64_t pad_left = pad_total / 2;
    int64_t pad_right = pad_total - pad_left;

    at::Tensor x_padded = pad_for_dwt(x_conv, pad_left, pad_right, mode);

    // Prepare filters: shape (out_channels, in_channels/groups, kernel_size)
    // Flip filters for convolution (conv1d does cross-correlation)
    at::Tensor lo_kernel = at::flip(filter_lo, {0}).reshape({1, 1, -1});
    at::Tensor hi_kernel = at::flip(filter_hi, {0}).reshape({1, 1, -1});

    // Convolve
    at::Tensor approx_full = at::conv1d(x_padded, lo_kernel);
    at::Tensor detail_full = at::conv1d(x_padded, hi_kernel);

    // Downsample by 2
    at::Tensor approx = approx_full.slice(2, 0, c10::nullopt, 2);
    at::Tensor detail = detail_full.slice(2, 0, c10::nullopt, 2);

    // Reshape back to original batch shape
    int64_t out_len = approx.size(-1);

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
