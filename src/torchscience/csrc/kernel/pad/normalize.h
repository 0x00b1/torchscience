#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <c10/util/Optional.h>

namespace torchscience::kernel::pad {

// Normalize padding to canonical form: [(before_0, after_0), (before_1, after_1), ...]
// Returns pair of (dimensions to pad, padding amounts as flat vector [b0,a0,b1,a1,...])
inline std::pair<std::vector<int64_t>, std::vector<int64_t>> normalize_padding(
    const std::vector<int64_t>& padding,
    const c10::optional<std::vector<int64_t>>& dim,
    int64_t ndim
) {
    std::vector<int64_t> dims;
    std::vector<int64_t> amounts;

    if (padding.empty()) {
        return {dims, amounts};
    }

    // Determine which dimensions to pad
    if (dim.has_value()) {
        dims = dim.value();
        // Normalize negative dimensions
        for (auto& d : dims) {
            if (d < 0) {
                d += ndim;
            }
            TORCH_CHECK(d >= 0 && d < ndim,
                "Dimension out of range (expected to be in range of [", -ndim, ", ", ndim - 1,
                "], but got ", d, ")");
        }
    } else {
        // Infer dimensions from padding length
        // PyTorch-style: padding specifies pairs for trailing dimensions
        int64_t num_dims = static_cast<int64_t>(padding.size()) / 2;
        if (padding.size() % 2 != 0) {
            // Single value or odd length - assume symmetric padding
            num_dims = static_cast<int64_t>(padding.size());
        }
        for (int64_t i = 0; i < num_dims && i < ndim; ++i) {
            dims.push_back(ndim - 1 - i);  // Trailing dimensions
        }
        std::reverse(dims.begin(), dims.end());
    }

    // Parse padding amounts based on format
    if (padding.size() == 1) {
        // Single int: same padding on all sides
        for (size_t i = 0; i < dims.size(); ++i) {
            amounts.push_back(padding[0]);  // before
            amounts.push_back(padding[0]);  // after
        }
    } else if (padding.size() == 2) {
        // (before, after): same for all dims
        for (size_t i = 0; i < dims.size(); ++i) {
            amounts.push_back(padding[0]);  // before
            amounts.push_back(padding[1]);  // after
        }
    } else if (padding.size() == dims.size() * 2) {
        // PyTorch-style: (left_n, right_n, left_n-1, right_n-1, ...)
        // Convert to our format: (before_0, after_0, before_1, after_1, ...)
        for (size_t i = 0; i < dims.size(); ++i) {
            size_t rev_i = dims.size() - 1 - i;
            amounts.push_back(padding[rev_i * 2]);      // before
            amounts.push_back(padding[rev_i * 2 + 1]);  // after
        }
    } else {
        TORCH_CHECK(false,
            "Invalid padding specification. Expected 1, 2, or ", dims.size() * 2,
            " values, got ", padding.size());
    }

    return {dims, amounts};
}

// Compute output shape after padding
inline std::vector<int64_t> compute_output_shape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& dims,
    const std::vector<int64_t>& amounts
) {
    std::vector<int64_t> output_shape = input_shape;
    for (size_t i = 0; i < dims.size(); ++i) {
        int64_t before = amounts[i * 2];
        int64_t after = amounts[i * 2 + 1];
        output_shape[dims[i]] += before + after;
    }
    return output_shape;
}

}  // namespace torchscience::kernel::pad
