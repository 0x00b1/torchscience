#pragma once

#include <algorithm>
#include <limits>
#include <vector>

namespace torchscience::kernel::morphology {

/**
 * Compute N-dimensional erosion for a single output position.
 *
 * Erosion is defined as:
 *   flat:     erosion(f, B)(x) = min{f(x + b) : b in B}
 *   non-flat: erosion(f, g)(x) = min{f(x + b) - g(b) : b in support(g)}
 *
 * @param input_ptr Pointer to input tensor data
 * @param se_ptr Pointer to structuring element data (nullptr for flat SE)
 * @param se_mask_ptr Pointer to binary mask indicating SE support (nullptr = all true)
 * @param output_idx Linear index into output tensor
 * @param input_shape Shape of input tensor (spatial dims only)
 * @param se_shape Shape of structuring element (spatial dims only)
 * @param origin Origin offset for SE (nullptr = center)
 * @param ndim Number of spatial dimensions
 * @param input_strides Strides for input tensor
 * @param se_strides Strides for structuring element
 * @param padding_mode 0=zeros, 1=reflect, 2=replicate, 3=circular
 * @param padding_value Value for zero padding (typically +inf for erosion)
 * @return Eroded value at the output position
 */
template <typename T>
T erosion_scalar(
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

    T min_val = std::numeric_limits<T>::infinity();

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

        // Check if this SE position is in support (if mask provided)
        if (se_mask_ptr) {
            int64_t mask_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                mask_offset += se_coords[d] * se_strides[d];
            }
            if (!se_mask_ptr[mask_offset]) {
                continue;
            }
        }

        // Compute input coordinates: x + b - origin
        std::vector<int64_t> in_coords(ndim);
        bool valid = true;

        for (int64_t d = 0; d < ndim; ++d) {
            int64_t coord = out_coords[d] + se_coords[d] - se_origin[d];

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

            // Subtract SE weight for non-flat (grayscale) erosion
            if (se_ptr) {
                int64_t se_offset = 0;
                for (int64_t d = 0; d < ndim; ++d) {
                    se_offset += se_coords[d] * se_strides[d];
                }
                val -= se_ptr[se_offset];
            }
        }

        min_val = std::min(min_val, val);
    }

    return min_val;
}

/**
 * Compute erosion and track the argmin position for backward pass.
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
 * @param argmin_input_idx Output: linear index of argmin in input tensor (-1 if padded)
 * @return Eroded value
 */
template <typename T>
T erosion_scalar_with_argmin(
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
    int64_t& argmin_input_idx
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

    T min_val = std::numeric_limits<T>::infinity();
    argmin_input_idx = -1;

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

        // Compute input coordinates: x + b - origin
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

            // Compute linear index for argmin tracking
            current_in_idx = 0;
            int64_t stride = 1;
            for (int64_t d = ndim - 1; d >= 0; --d) {
                current_in_idx += in_coords[d] * stride;
                stride *= input_shape[d];
            }

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
            argmin_input_idx = from_padding ? -1 : current_in_idx;
        }
    }

    return min_val;
}

}  // namespace torchscience::kernel::morphology
