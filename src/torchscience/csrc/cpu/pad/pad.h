#pragma once

#include <cmath>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "../../kernel/pad/pad.h"
#include "../../kernel/pad/extrapolate.h"
#include "../../kernel/pad/normalize.h"

namespace torchscience::cpu::pad {

namespace {

using namespace torchscience::kernel::pad;

template <typename scalar_t>
void pad_1d_kernel(
    scalar_t* output_data,
    const scalar_t* input_data,
    int64_t in_size,
    int64_t out_size,
    int64_t pad_before,
    PaddingMode mode,
    scalar_t value,
    int order
) {
    bool use_extrapolation = is_extrapolation_mode(mode);

    for (int64_t i = 0; i < out_size; ++i) {
        if (use_extrapolation) {
            if (i >= pad_before && i < pad_before + in_size) {
                output_data[i] = input_data[i - pad_before];
            } else {
                scalar_t distance;
                bool left_side;
                if (i < pad_before) {
                    distance = static_cast<scalar_t>(pad_before - i);
                    left_side = true;
                } else {
                    distance = static_cast<scalar_t>(i - (pad_before + in_size) + 1);
                    left_side = false;
                }

                auto weights = get_extrapolation_weights(distance, mode, order);

                scalar_t result = scalar_t(0);
                for (int j = 0; j < weights.num_points; ++j) {
                    int64_t src_idx;
                    if (left_side) {
                        src_idx = j;
                    } else {
                        src_idx = in_size - 1 - j;
                    }
                    if (src_idx >= 0 && src_idx < in_size) {
                        result += weights.weights[j] * input_data[src_idx];
                    }
                }
                output_data[i] = result;
            }
        } else {
            auto [src_idx, mult] = map_index(i, in_size, pad_before, mode);
            if (src_idx >= 0) {
                output_data[i] = static_cast<scalar_t>(mult) * input_data[src_idx];
            } else {
                output_data[i] = value;
            }
        }
    }
}

}  // anonymous namespace

inline at::Tensor pad(
    const at::Tensor& input,
    const std::vector<int64_t>& padding,
    const std::string& mode_str,
    double value,
    const c10::optional<std::vector<int64_t>>& dim,
    int64_t order,
    const c10::optional<at::Tensor>& out
) {
    TORCH_CHECK(input.numel() > 0, "pad: input tensor must be non-empty");
    TORCH_CHECK(order >= 1 && order <= MAX_EXTRAPOLATION_ORDER,
        "pad: order must be between 1 and ", MAX_EXTRAPOLATION_ORDER, ", got ", order);

    PaddingMode mode = parse_mode(mode_str);
    int64_t ndim = input.dim();

    auto [dims, amounts] = normalize_padding(padding, dim, ndim);

    if (dims.empty()) {
        if (out.has_value()) {
            out.value().copy_(input);
            return out.value();
        }
        return input.clone();
    }

    std::vector<int64_t> output_shape = compute_output_shape(
        input.sizes().vec(), dims, amounts
    );

    at::Tensor result;
    if (out.has_value()) {
        result = out.value();
        TORCH_CHECK(result.sizes().vec() == output_shape,
            "pad: out tensor has wrong shape");
    } else {
        result = at::empty(output_shape, input.options());
    }

    at::Tensor current = input.contiguous();

    for (size_t d = 0; d < dims.size(); ++d) {
        int64_t dim_idx = dims[d];
        int64_t pad_before = amounts[d * 2];
        int64_t pad_after = amounts[d * 2 + 1];

        if (pad_before == 0 && pad_after == 0) {
            continue;
        }

        int64_t in_size = current.size(dim_idx);
        int64_t out_size = in_size + pad_before + pad_after;

        std::vector<int64_t> step_shape = current.sizes().vec();
        step_shape[dim_idx] = out_size;
        at::Tensor step_output = at::empty(step_shape, current.options());

        at::Tensor current_t = current.movedim(dim_idx, -1).contiguous();
        at::Tensor output_t = step_output.movedim(dim_idx, -1);

        int64_t batch_size = current_t.numel() / in_size;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16,
            current.scalar_type(),
            "pad_cpu",
            [&]() {
                scalar_t fill_value = static_cast<scalar_t>(value);
                const scalar_t* in_ptr = current_t.data_ptr<scalar_t>();
                scalar_t* out_ptr = output_t.data_ptr<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
                    for (int64_t b = start; b < end; ++b) {
                        pad_1d_kernel(
                            out_ptr + b * out_size,
                            in_ptr + b * in_size,
                            in_size,
                            out_size,
                            pad_before,
                            mode,
                            fill_value,
                            static_cast<int>(order)
                        );
                    }
                });
            }
        );

        current = output_t.movedim(-1, dim_idx).contiguous();
    }

    if (out.has_value()) {
        out.value().copy_(current);
        return out.value();
    }

    return current;
}

inline at::Tensor pad_backward(
    const at::Tensor& grad_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& padding,
    const std::string& mode_str,
    const c10::optional<std::vector<int64_t>>& dim,
    int64_t order
) {
    PaddingMode mode = parse_mode(mode_str);
    int64_t ndim = static_cast<int64_t>(input_shape.size());

    auto [dims, amounts] = normalize_padding(padding, dim, ndim);

    at::Tensor grad_input = at::zeros(input_shape, grad_output.options());

    if (dims.empty()) {
        grad_input.copy_(grad_output);
        return grad_input;
    }

    at::Tensor current_grad = grad_output.contiguous();

    for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
        int64_t dim_idx = dims[d];
        int64_t pad_before = amounts[d * 2];
        int64_t pad_after = amounts[d * 2 + 1];

        if (pad_before == 0 && pad_after == 0) {
            continue;
        }

        std::vector<int64_t> target_shape = current_grad.sizes().vec();
        target_shape[dim_idx] -= pad_before + pad_after;
        int64_t in_size = target_shape[dim_idx];
        int64_t out_size = current_grad.size(dim_idx);

        at::Tensor step_grad = at::zeros(target_shape, current_grad.options());

        at::Tensor grad_t = current_grad.movedim(dim_idx, -1).contiguous();
        at::Tensor step_t = step_grad.movedim(dim_idx, -1);

        int64_t batch_size = grad_t.numel() / out_size;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16,
            current_grad.scalar_type(),
            "pad_backward_cpu",
            [&]() {
                const scalar_t* grad_ptr = grad_t.data_ptr<scalar_t>();
                scalar_t* out_ptr = step_t.data_ptr<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
                    for (int64_t b = start; b < end; ++b) {
                        const scalar_t* g_row = grad_ptr + b * out_size;
                        scalar_t* o_row = out_ptr + b * in_size;

                        for (int64_t i = 0; i < in_size; ++i) {
                            o_row[i] = g_row[pad_before + i];
                        }

                        if (mode == PaddingMode::Constant) {
                            // Constant: gradients in padding region don't propagate
                        } else if (mode == PaddingMode::Replicate) {
                            for (int64_t i = 0; i < pad_before; ++i) {
                                o_row[0] += g_row[i];
                            }
                            for (int64_t i = 0; i < pad_after; ++i) {
                                o_row[in_size - 1] += g_row[pad_before + in_size + i];
                            }
                        } else if (mode == PaddingMode::Reflect || mode == PaddingMode::ReflectOdd) {
                            bool odd = (mode == PaddingMode::ReflectOdd);
                            for (int64_t i = 0; i < pad_before; ++i) {
                                int64_t rel_idx = -(pad_before - i);
                                int64_t src_idx = reflect_index(rel_idx, in_size);
                                scalar_t mult = odd ? static_cast<scalar_t>(reflect_sign(rel_idx, in_size)) : scalar_t(1);
                                o_row[src_idx] += mult * g_row[i];
                            }
                            for (int64_t i = 0; i < pad_after; ++i) {
                                int64_t rel_idx = in_size + i;
                                int64_t src_idx = reflect_index(rel_idx, in_size);
                                scalar_t mult = odd ? static_cast<scalar_t>(reflect_sign(rel_idx, in_size)) : scalar_t(1);
                                o_row[src_idx] += mult * g_row[pad_before + in_size + i];
                            }
                        } else if (mode == PaddingMode::Circular) {
                            for (int64_t i = 0; i < pad_before; ++i) {
                                int64_t rel_idx = -(pad_before - i);
                                int64_t src_idx = ((rel_idx % in_size) + in_size) % in_size;
                                o_row[src_idx] += g_row[i];
                            }
                            for (int64_t i = 0; i < pad_after; ++i) {
                                int64_t rel_idx = in_size + i;
                                int64_t src_idx = rel_idx % in_size;
                                o_row[src_idx] += g_row[pad_before + in_size + i];
                            }
                        } else if (is_extrapolation_mode(mode)) {
                            for (int64_t i = 0; i < pad_before; ++i) {
                                scalar_t distance = static_cast<scalar_t>(pad_before - i);
                                auto weights = get_extrapolation_weights(distance, mode, static_cast<int>(order));
                                for (int j = 0; j < weights.num_points; ++j) {
                                    if (j < in_size) {
                                        o_row[j] += weights.weights[j] * g_row[i];
                                    }
                                }
                            }
                            for (int64_t i = 0; i < pad_after; ++i) {
                                scalar_t distance = static_cast<scalar_t>(i + 1);
                                auto weights = get_extrapolation_weights(distance, mode, static_cast<int>(order));
                                for (int j = 0; j < weights.num_points; ++j) {
                                    int64_t src_idx = in_size - 1 - j;
                                    if (src_idx >= 0) {
                                        o_row[src_idx] += weights.weights[j] * g_row[pad_before + in_size + i];
                                    }
                                }
                            }
                        }
                    }
                });
            }
        );

        current_grad = step_t.movedim(-1, dim_idx).contiguous();
    }

    grad_input.copy_(current_grad);
    return grad_input;
}

inline at::Tensor pad_backward_backward(
    const at::Tensor& grad_grad_input,
    const std::vector<int64_t>& padding,
    const std::string& mode_str,
    const c10::optional<std::vector<int64_t>>& dim,
    int64_t order
) {
    return pad(grad_grad_input, padding, mode_str, 0.0, dim, order, c10::nullopt);
}

}  // namespace torchscience::cpu::pad

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("pad", &torchscience::cpu::pad::pad);
    module.impl("pad_backward", &torchscience::cpu::pad::pad_backward);
    module.impl("pad_backward_backward", &torchscience::cpu::pad::pad_backward_backward);
}
