#pragma once

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/fft_ifft.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <torch/library.h>

namespace torchscience::cpu::integral_transform {

namespace {

// ============================================================================
// Padding Helpers
// ============================================================================

enum class InversePaddingMode : int64_t {
    Constant = 0,
    Reflect = 1,
    Replicate = 2,
    Circular = 3
};

inline std::string inverse_padding_mode_to_string(int64_t mode) {
    switch (static_cast<InversePaddingMode>(mode)) {
        case InversePaddingMode::Constant: return "constant";
        case InversePaddingMode::Reflect: return "reflect";
        case InversePaddingMode::Replicate: return "replicate";
        case InversePaddingMode::Circular: return "circular";
        default:
            TORCH_CHECK(false, "Invalid padding_mode: ", mode);
    }
}

inline at::Tensor inverse_apply_padding_step(
    const at::Tensor& input,
    int64_t pad_amount,
    int64_t padding_mode,
    double padding_value,
    bool needs_unsqueeze
) {
    at::Tensor input_work = input;

    if (needs_unsqueeze) {
        input_work = input_work.unsqueeze(0);
    }

    std::vector<int64_t> pad_sizes = {0, pad_amount};
    std::string mode_str = inverse_padding_mode_to_string(padding_mode);

    at::Tensor padded;
    if (padding_mode == static_cast<int64_t>(InversePaddingMode::Constant)) {
        padded = at::pad(input_work, pad_sizes, mode_str, padding_value);
    } else {
        padded = at::pad(input_work, pad_sizes, mode_str);
    }

    if (needs_unsqueeze) {
        padded = padded.squeeze(0);
    }

    return padded;
}

inline at::Tensor inverse_apply_padding(
    const at::Tensor& input,
    int64_t target_size,
    int64_t dim,
    int64_t padding_mode,
    double padding_value
) {
    int64_t current_size = input.size(dim);

    if (target_size <= current_size) {
        return input;
    }

    int64_t total_pad = target_size - current_size;
    at::Tensor result = input.movedim(dim, -1);

    bool needs_unsqueeze = (result.dim() == 1) &&
        (padding_mode != static_cast<int64_t>(InversePaddingMode::Constant));

    if (padding_mode == static_cast<int64_t>(InversePaddingMode::Constant)) {
        result = inverse_apply_padding_step(result, total_pad, padding_mode, padding_value, needs_unsqueeze);
    } else {
        int64_t remaining_pad = total_pad;

        while (remaining_pad > 0) {
            int64_t current_dim_size = result.size(-1);
            int64_t max_pad = current_dim_size - 1;
            TORCH_CHECK(max_pad > 0,
                "Cannot use reflect/replicate/circular padding with dimension size 1.");

            int64_t pad_this_step = std::min(remaining_pad, max_pad);
            result = inverse_apply_padding_step(result, pad_this_step, padding_mode, padding_value, needs_unsqueeze);
            remaining_pad -= pad_this_step;
        }
    }

    return result.movedim(-1, dim);
}

// ============================================================================
// Window Application
// ============================================================================

inline at::Tensor inverse_apply_window(
    const at::Tensor& input,
    const at::Tensor& window,
    int64_t dim
) {
    TORCH_CHECK(window.dim() == 1, "window must be 1-D");
    TORCH_CHECK(window.device() == input.device(), "window must be on same device as input");
    TORCH_CHECK(window.size(0) == input.size(dim), "window size must match input size along dim");

    std::vector<int64_t> window_shape(input.dim(), 1);
    window_shape[dim] = window.size(0);

    at::Tensor window_reshaped = window.view(window_shape);

    return input * window_reshaped;
}

// ============================================================================
// Inverse Hilbert Frequency Response
// ============================================================================

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> inverse_hilbert_frequency_response(int64_t k, int64_t n) {
    if (k == 0) {
        return c10::complex<T>(T(0), T(0));
    }

    if (n % 2 == 0 && k == n / 2) {
        return c10::complex<T>(T(0), T(0));
    }

    if (k < (n + 1) / 2) {
        return c10::complex<T>(T(0), T(1));
    }

    return c10::complex<T>(T(0), T(-1));
}

// ============================================================================
// Backward Padding Helpers
// ============================================================================

inline at::Tensor inverse_reflect_padding_backward_step(
    const at::Tensor& grad_output,
    int64_t input_size,
    int64_t pad_amount
) {
    at::Tensor grad_input = grad_output.narrow(-1, 0, input_size).clone();

    for (int64_t i = 0; i < pad_amount; ++i) {
        int64_t src_pos = input_size + i;
        int64_t dst_pos = input_size - 2 - i;

        while (dst_pos < 0) {
            dst_pos = -dst_pos;
        }
        while (dst_pos >= input_size) {
            dst_pos = 2 * input_size - 2 - dst_pos;
            if (dst_pos < 0) dst_pos = -dst_pos;
        }

        if (dst_pos >= 0 && dst_pos < input_size) {
            grad_input.select(-1, dst_pos).add_(grad_output.select(-1, src_pos));
        }
    }

    return grad_input;
}

inline at::Tensor inverse_replicate_padding_backward_step(
    const at::Tensor& grad_output,
    int64_t input_size,
    int64_t pad_amount
) {
    at::Tensor grad_input = grad_output.narrow(-1, 0, input_size).clone();

    for (int64_t i = 0; i < pad_amount; ++i) {
        grad_input.select(-1, input_size - 1).add_(
            grad_output.select(-1, input_size + i)
        );
    }

    return grad_input;
}

inline at::Tensor inverse_circular_padding_backward_step(
    const at::Tensor& grad_output,
    int64_t input_size,
    int64_t pad_amount
) {
    at::Tensor grad_input = grad_output.narrow(-1, 0, input_size).clone();

    for (int64_t i = 0; i < pad_amount; ++i) {
        int64_t dst_pos = i % input_size;
        grad_input.select(-1, dst_pos).add_(
            grad_output.select(-1, input_size + i)
        );
    }

    return grad_input;
}

inline at::Tensor inverse_adjust_backward_gradient_size(
    const at::Tensor& grad,
    int64_t input_size,
    int64_t n,
    int64_t dim,
    int64_t padding_mode = 0
) {
    if (n > input_size) {
        at::Tensor grad_moved = grad.movedim(dim, -1).contiguous();

        if (padding_mode == static_cast<int64_t>(InversePaddingMode::Constant)) {
            at::Tensor result = grad_moved.narrow(-1, 0, input_size).contiguous();
            return result.movedim(-1, dim);
        }

        std::vector<int64_t> sizes;
        int64_t current = input_size;
        while (current < n) {
            sizes.push_back(current);
            int64_t max_pad = current - 1;
            current = std::min(current + max_pad, n);
        }

        at::Tensor grad_work = grad_moved;

        for (auto it = sizes.rbegin(); it != sizes.rend(); ++it) {
            int64_t target_size = *it;
            int64_t current_size = grad_work.size(-1);
            int64_t pad_amount = current_size - target_size;

            if (pad_amount > 0) {
                if (padding_mode == static_cast<int64_t>(InversePaddingMode::Reflect)) {
                    grad_work = inverse_reflect_padding_backward_step(grad_work, target_size, pad_amount);
                } else if (padding_mode == static_cast<int64_t>(InversePaddingMode::Replicate)) {
                    grad_work = inverse_replicate_padding_backward_step(grad_work, target_size, pad_amount);
                } else if (padding_mode == static_cast<int64_t>(InversePaddingMode::Circular)) {
                    grad_work = inverse_circular_padding_backward_step(grad_work, target_size, pad_amount);
                }
            }
        }

        return grad_work.movedim(-1, dim);
    } else if (n < input_size) {
        std::vector<int64_t> pad_shape(grad.sizes().begin(), grad.sizes().end());
        pad_shape[dim] = input_size;
        at::Tensor padded = at::zeros(pad_shape, grad.options());
        padded.narrow(dim, 0, n).copy_(grad);
        return padded;
    }
    return grad.contiguous();
}

}  // anonymous namespace

/**
 * CPU implementation of inverse Hilbert transform with padding and windowing.
 */
inline at::Tensor inverse_hilbert_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    TORCH_CHECK(input.numel() > 0, "inverse_hilbert_transform: input tensor must be non-empty");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim,
        "inverse_hilbert_transform: dim out of range (got ", dim, " for tensor with ", ndim, " dimensions)");

    TORCH_CHECK(input.size(dim) > 0, "inverse_hilbert_transform: transform dimension must have positive size");

    // Determine FFT length
    int64_t input_size = input.size(dim);
    int64_t n = (n_param > 0) ? n_param : input_size;
    TORCH_CHECK(n > 0, "inverse_hilbert_transform: n must be positive");

    // Ensure contiguous for efficient operations
    at::Tensor processed = input.contiguous();

    // Apply padding if needed
    if (n > input_size) {
        processed = inverse_apply_padding(
            processed, n, dim, padding_mode, padding_value
        );
    } else if (n < input_size) {
        // Truncation
        processed = processed.narrow(dim, 0, n);
    }

    // Apply window if provided
    if (window.has_value()) {
        processed = inverse_apply_window(
            processed, window.value(), dim
        );
    }

    // Compute FFT along specified dimension
    at::Tensor spectrum = at::fft_fft(processed, c10::nullopt, dim);

    // Create frequency response tensor
    std::vector<int64_t> response_shape(ndim, 1);
    response_shape[dim] = n;

    at::Tensor response = at::zeros(response_shape, spectrum.options());

    // Fill in the response values
    AT_DISPATCH_COMPLEX_TYPES(
        spectrum.scalar_type(),
        "inverse_hilbert_transform_cpu_response",
        [&]() {
            using real_t = typename c10::scalar_value_type<scalar_t>::type;

            auto response_flat = response.view({n});
            auto* response_data = response_flat.data_ptr<scalar_t>();

            for (int64_t k = 0; k < n; ++k) {
                auto h = inverse_hilbert_frequency_response<real_t>(k, n);
                response_data[k] = scalar_t(h.real(), h.imag());
            }
        }
    );

    // Apply frequency response
    at::Tensor modified_spectrum = spectrum * response;

    // Compute inverse FFT
    at::Tensor result = at::fft_ifft(modified_spectrum, c10::nullopt, dim);

    // If input was real, return real part
    if (!input.is_complex()) {
        result = at::real(result).contiguous();
    }

    return result;
}

/**
 * Backward pass for inverse Hilbert transform on CPU.
 */
inline at::Tensor inverse_hilbert_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    // Normalize dimension
    int64_t ndim = input.dim();
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;

    int64_t input_size = input.size(norm_dim);
    int64_t n = (n_param > 0) ? n_param : input_size;

    // Apply -H^{-1} with size n to grad_output (since (H^{-1})^T = H = -H^{-1})
    at::Tensor grad = -inverse_hilbert_transform(grad_output, n_param, dim, 0, 0.0, c10::nullopt);

    // If window was applied in forward, multiply gradient by window BEFORE size adjustment
    // (chain rule: d/dx[w*x] = w * d/dx[x], where w operates at padded size)
    if (window.has_value()) {
        grad = inverse_apply_window(grad, window.value(), norm_dim);
    }

    // Adjust size to match input shape (with proper gradient accumulation for padding)
    grad = inverse_adjust_backward_gradient_size(
        grad, input_size, n, norm_dim, padding_mode
    );

    return grad;
}

/**
 * Double backward pass for inverse Hilbert transform on CPU.
 */
inline std::tuple<at::Tensor, at::Tensor> inverse_hilbert_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    // grad_grad_output = H^{-1}[grad_grad_input]
    at::Tensor grad_grad_output = inverse_hilbert_transform(
        grad_grad_input, n_param, dim, padding_mode, padding_value, window
    );

    // No second-order term (H^{-1} is linear)
    at::Tensor new_grad_input = at::zeros_like(input);

    return std::make_tuple(grad_grad_output, new_grad_input);
}

}  // namespace torchscience::cpu::integral_transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "inverse_hilbert_transform",
        &torchscience::cpu::integral_transform::inverse_hilbert_transform
    );

    module.impl(
        "inverse_hilbert_transform_backward",
        &torchscience::cpu::integral_transform::inverse_hilbert_transform_backward
    );

    module.impl(
        "inverse_hilbert_transform_backward_backward",
        &torchscience::cpu::integral_transform::inverse_hilbert_transform_backward_backward
    );
}
