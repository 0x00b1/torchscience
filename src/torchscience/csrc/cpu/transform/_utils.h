#pragma once

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/zeros.h>
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

namespace torchscience::cpu::transform {

// ============================================================================
// Padding Helpers
// ============================================================================

enum class PaddingMode : int64_t {
    Constant = 0,
    Reflect = 1,
    Replicate = 2,
    Circular = 3
};

inline std::string padding_mode_to_string(int64_t mode) {
    switch (static_cast<PaddingMode>(mode)) {
        case PaddingMode::Constant: return "constant";
        case PaddingMode::Reflect: return "reflect";
        case PaddingMode::Replicate: return "replicate";
        case PaddingMode::Circular: return "circular";
        default:
            TORCH_CHECK(false, "Invalid padding_mode: ", mode);
    }
}

inline at::Tensor apply_padding_step(
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
    std::string mode_str = padding_mode_to_string(padding_mode);

    at::Tensor padded;
    if (padding_mode == static_cast<int64_t>(PaddingMode::Constant)) {
        padded = at::pad(input_work, pad_sizes, mode_str, padding_value);
    } else {
        padded = at::pad(input_work, pad_sizes, mode_str);
    }

    if (needs_unsqueeze) {
        padded = padded.squeeze(0);
    }

    return padded;
}

inline at::Tensor apply_padding(
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
        (padding_mode != static_cast<int64_t>(PaddingMode::Constant));

    if (padding_mode == static_cast<int64_t>(PaddingMode::Constant)) {
        result = apply_padding_step(result, total_pad, padding_mode, padding_value, needs_unsqueeze);
    } else {
        int64_t remaining_pad = total_pad;

        while (remaining_pad > 0) {
            int64_t current_dim_size = result.size(-1);
            int64_t max_pad = current_dim_size - 1;
            TORCH_CHECK(max_pad > 0,
                "Cannot use reflect/replicate/circular padding with dimension size 1.");

            int64_t pad_this_step = std::min(remaining_pad, max_pad);
            result = apply_padding_step(result, pad_this_step, padding_mode, padding_value, needs_unsqueeze);
            remaining_pad -= pad_this_step;
        }
    }

    return result.movedim(-1, dim);
}

// ============================================================================
// Window Application
// ============================================================================

inline at::Tensor apply_window(
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
// Backward Padding Helpers
// ============================================================================

inline at::Tensor reflect_padding_backward_step(
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

inline at::Tensor replicate_padding_backward_step(
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

inline at::Tensor circular_padding_backward_step(
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

inline at::Tensor adjust_backward_gradient_size(
    const at::Tensor& grad,
    int64_t input_size,
    int64_t n,
    int64_t dim,
    int64_t padding_mode = 0
) {
    if (n > input_size) {
        at::Tensor grad_moved = grad.movedim(dim, -1).contiguous();

        if (padding_mode == static_cast<int64_t>(PaddingMode::Constant)) {
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
                if (padding_mode == static_cast<int64_t>(PaddingMode::Reflect)) {
                    grad_work = reflect_padding_backward_step(grad_work, target_size, pad_amount);
                } else if (padding_mode == static_cast<int64_t>(PaddingMode::Replicate)) {
                    grad_work = replicate_padding_backward_step(grad_work, target_size, pad_amount);
                } else if (padding_mode == static_cast<int64_t>(PaddingMode::Circular)) {
                    grad_work = circular_padding_backward_step(grad_work, target_size, pad_amount);
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

}  // namespace torchscience::cpu::transform
