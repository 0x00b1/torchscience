#pragma once

#include <algorithm>
#include <limits>
#include <vector>

namespace torchscience::kernel::morphology {

/**
 * Compute N-dimensional dilation for a single output position.
 *
 * Dilation is defined as:
 *   flat:     dilation(f, B)(x) = max{f(x - b) : b in B}
 *   non-flat: dilation(f, g)(x) = max{f(x - b) + g(b) : b in support(g)}
 *
 * Note: For dilation, we use (x - b) instead of (x + b) as in erosion.
 * This makes dilation the adjoint of erosion when using reflected SE.
 *
 * @param input_ptr Pointer to input tensor data
 * @param se_ptr Pointer to structuring element data (nullptr for flat SE)
 * @param se_mask_ptr Pointer to binary mask indicating SE support
 * @param output_idx Linear index into output tensor
 * @param input_shape Shape of input tensor (spatial dims only)
 * @param se_shape Shape of structuring element (spatial dims only)
 * @param origin Origin offset for SE (nullptr = center)
 * @param ndim Number of spatial dimensions
 * @param input_strides Strides for input tensor
 * @param se_strides Strides for structuring element
 * @param padding_mode 0=zeros, 1=reflect, 2=replicate, 3=circular
 * @param padding_value Value for zero padding (typically -inf for dilation)
 * @return Dilated value at the output position
 */
template <typename T>
T dilation_scalar(
    const T* input_ptr,
    const T* se_ptr,
    const bool* se_mask_ptr,
    int64_t output_idx,
    const int64_t* input_shape,
    const int64_t* se_shape,
    const int64_t* origin,
    int64_t ndim,
    const int64_t* input_strides,
    const int64_t* se_strides,
    int64_t padding_mode,
    T padding_value
) {
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

    T max_val = -std::numeric_limits<T>::infinity();

    // Iterate over all positions in the structuring element
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

        // Compute input coordinates: x - b + origin (reflected)
        // This is equivalent to x - (b - origin)
        std::vector<int64_t> in_coords(ndim);
        bool valid = true;

        for (int64_t d = 0; d < ndim; ++d) {
            int64_t coord = out_coords[d] - se_coords[d] + se_origin[d];

            if (coord < 0 || coord >= input_shape[d]) {
                switch (padding_mode) {
                    case 0:  // zeros (use padding_value)
                        valid = false;
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
        if (!valid) {
            val = padding_value;
        } else {
            // Compute input offset
            int64_t in_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                in_offset += in_coords[d] * input_strides[d];
            }
            val = input_ptr[in_offset];

            // Add SE weight for non-flat (grayscale) dilation
            if (se_ptr) {
                int64_t se_offset = 0;
                for (int64_t d = 0; d < ndim; ++d) {
                    se_offset += se_coords[d] * se_strides[d];
                }
                val += se_ptr[se_offset];
            }
        }

        max_val = std::max(max_val, val);
    }

    return max_val;
}

/**
 * Compute dilation and track the argmax position for backward pass.
 *
 * @param input_ptr Pointer to input tensor data
 * @param se_ptr Pointer to structuring element data (nullptr for flat SE)
 * @param se_mask_ptr Pointer to binary mask indicating SE support
 * @param output_idx Linear index into output tensor
 * @param input_shape Shape of input tensor (spatial dims only)
 * @param se_shape Shape of structuring element (spatial dims only)
 * @param origin Origin offset for SE
 * @param ndim Number of spatial dimensions
 * @param input_strides Strides for input tensor
 * @param se_strides Strides for structuring element
 * @param padding_mode Padding mode
 * @param padding_value Value for zero padding
 * @param argmax_input_idx Output: linear index of argmax in input tensor (-1 if padded)
 * @return Dilated value
 */
template <typename T>
T dilation_scalar_with_argmax(
    const T* input_ptr,
    const T* se_ptr,
    const bool* se_mask_ptr,
    int64_t output_idx,
    const int64_t* input_shape,
    const int64_t* se_shape,
    const int64_t* origin,
    int64_t ndim,
    const int64_t* input_strides,
    const int64_t* se_strides,
    int64_t padding_mode,
    T padding_value,
    int64_t& argmax_input_idx
) {
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

    T max_val = -std::numeric_limits<T>::infinity();
    argmax_input_idx = -1;

    // Iterate over all positions in the structuring element
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

        // Compute input coordinates: x - b + origin
        std::vector<int64_t> in_coords(ndim);
        bool valid = true;
        bool from_padding = false;

        for (int64_t d = 0; d < ndim; ++d) {
            int64_t coord = out_coords[d] - se_coords[d] + se_origin[d];

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
        int64_t current_in_idx = -1;

        if (!valid) {
            val = padding_value;
        } else {
            // Compute input offset
            int64_t in_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                in_offset += in_coords[d] * input_strides[d];
            }
            val = input_ptr[in_offset];

            // Compute linear index for argmax tracking
            current_in_idx = 0;
            int64_t stride = 1;
            for (int64_t d = ndim - 1; d >= 0; --d) {
                current_in_idx += in_coords[d] * stride;
                stride *= input_shape[d];
            }

            // Add SE weight for non-flat dilation
            if (se_ptr) {
                int64_t se_offset = 0;
                for (int64_t d = 0; d < ndim; ++d) {
                    se_offset += se_coords[d] * se_strides[d];
                }
                val += se_ptr[se_offset];
            }
        }

        if (val > max_val) {
            max_val = val;
            argmax_input_idx = from_padding ? -1 : current_in_idx;
        }
    }

    return max_val;
}

}  // namespace torchscience::kernel::morphology
