#pragma once

#include <algorithm>
#include <limits>
#include <vector>

namespace torchscience::kernel::morphology {

/**
 * Compute backward pass for erosion.
 *
 * Gradient flows through the argmin indicator:
 *   grad_input[argmin_position] += grad_output[output_position]
 *
 * This function computes the contribution for a single output position.
 *
 * @param grad_output_val Gradient value at this output position
 * @param input_ptr Input tensor data (needed to find argmin)
 * @param se_ptr Structuring element data (nullptr for flat SE)
 * @param se_mask_ptr Binary mask for SE support
 * @param output_idx Linear index into output tensor
 * @param grad_input_ptr Gradient accumulator for input (atomic add)
 * @param input_shape Shape of input tensor
 * @param se_shape Shape of structuring element
 * @param origin Origin offset for SE
 * @param ndim Number of spatial dimensions
 * @param input_strides Strides for input tensor
 * @param se_strides Strides for structuring element
 * @param padding_mode Padding mode
 * @param padding_value Value for zero padding
 */
template <typename T>
void erosion_backward_scalar(
    T grad_output_val,
    const T* input_ptr,
    const T* se_ptr,
    const bool* se_mask_ptr,
    int64_t output_idx,
    T* grad_input_ptr,
    const int64_t* input_shape,
    const int64_t* se_shape,
    const int64_t* origin,
    int64_t ndim,
    const int64_t* input_strides,
    const int64_t* se_strides,
    int64_t padding_mode,
    T padding_value
) {
    if (grad_output_val == T(0)) {
        return;
    }

    // Convert output linear index to coordinates
    std::vector<int64_t> out_coords(ndim);
    int64_t remaining = output_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        out_coords[d] = remaining % input_shape[d];
        remaining /= input_shape[d];
    }

    // Compute origin (default to center)
    std::vector<int64_t> se_origin(ndim);
    for (int64_t d = 0; d < ndim; ++d) {
        se_origin[d] = origin ? origin[d] : se_shape[d] / 2;
    }

    T min_val = std::numeric_limits<T>::infinity();
    int64_t argmin_offset = -1;

    // Iterate over all positions in the structuring element to find argmin
    std::vector<int64_t> se_coords(ndim, 0);
    int64_t total_se_elements = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        total_se_elements *= se_shape[d];
    }

    for (int64_t se_idx = 0; se_idx < total_se_elements; ++se_idx) {
        // Convert se_idx to coordinates
        int64_t temp = se_idx;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            se_coords[d] = temp % se_shape[d];
            temp /= se_shape[d];
        }

        // Check if this SE position is in support
        if (se_mask_ptr) {
            int64_t mask_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                mask_offset += se_coords[d] * se_strides[d];
            }
            if (!se_mask_ptr[mask_offset]) {
                continue;
            }
        }

        // Compute input coordinates
        std::vector<int64_t> in_coords(ndim);
        bool valid = true;
        bool from_padding = false;

        for (int64_t d = 0; d < ndim; ++d) {
            int64_t coord = out_coords[d] + se_coords[d] - se_origin[d];

            if (coord < 0 || coord >= input_shape[d]) {
                switch (padding_mode) {
                    case 0:  // zeros
                        valid = false;
                        from_padding = true;
                        break;
                    case 1:  // reflect
                        if (coord < 0) {
                            coord = -coord - 1;
                            if (coord >= input_shape[d]) coord = input_shape[d] - 1;
                        } else {
                            coord = 2 * input_shape[d] - coord - 1;
                            if (coord < 0) coord = 0;
                        }
                        break;
                    case 2:  // replicate
                        coord = std::max(int64_t(0), std::min(coord, input_shape[d] - 1));
                        break;
                    case 3:  // circular
                        coord = ((coord % input_shape[d]) + input_shape[d]) % input_shape[d];
                        break;
                }
            }
            in_coords[d] = coord;
        }

        T val;
        int64_t current_offset = -1;

        if (!valid) {
            val = padding_value;
        } else {
            // Compute input offset
            current_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                current_offset += in_coords[d] * input_strides[d];
            }
            val = input_ptr[current_offset];

            // Subtract SE weight for non-flat erosion
            if (se_ptr) {
                int64_t se_offset = 0;
                for (int64_t d = 0; d < ndim; ++d) {
                    se_offset += se_coords[d] * se_strides[d];
                }
                val -= se_ptr[se_offset];
            }
        }

        if (val < min_val) {
            min_val = val;
            argmin_offset = from_padding ? -1 : current_offset;
        }
    }

    // Accumulate gradient at argmin position
    if (argmin_offset >= 0) {
        grad_input_ptr[argmin_offset] += grad_output_val;
    }
}

}  // namespace torchscience::kernel::morphology
