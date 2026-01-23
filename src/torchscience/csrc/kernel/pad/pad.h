#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include <c10/macros/Macros.h>

namespace torchscience::kernel::pad {

constexpr int64_t MAX_EXTRAPOLATION_ORDER = 8;

enum class PaddingMode : int64_t {
    Constant = 0,
    Replicate = 1,
    Reflect = 2,
    ReflectOdd = 3,
    Circular = 4,
    Linear = 5,
    Polynomial = 6,
    Spline = 7,
    Smooth = 8,
};

inline PaddingMode parse_mode(const std::string& mode) {
    if (mode == "constant") return PaddingMode::Constant;
    if (mode == "replicate") return PaddingMode::Replicate;
    if (mode == "reflect") return PaddingMode::Reflect;
    if (mode == "reflect_odd") return PaddingMode::ReflectOdd;
    if (mode == "circular") return PaddingMode::Circular;
    if (mode == "linear") return PaddingMode::Linear;
    if (mode == "polynomial") return PaddingMode::Polynomial;
    if (mode == "spline") return PaddingMode::Spline;
    if (mode == "smooth") return PaddingMode::Smooth;
    TORCH_CHECK(false, "Unknown padding mode: ", mode,
        ". Supported modes: constant, replicate, reflect, reflect_odd, "
        "circular, linear, polynomial, spline, smooth");
}

inline bool is_extrapolation_mode(PaddingMode mode) {
    return mode == PaddingMode::Linear ||
           mode == PaddingMode::Polynomial ||
           mode == PaddingMode::Spline ||
           mode == PaddingMode::Smooth;
}

// Edge-inclusive reflection index computation
// For reflect mode: values at boundaries are included in reflection
template <typename IndexType>
C10_HOST_DEVICE C10_ALWAYS_INLINE
IndexType reflect_index(IndexType idx, IndexType size) {
    // Handle index outside [0, size)
    // Reflection pattern: 0,1,2,...,size-1,size-2,...,1,0,1,2,...
    if (idx < 0) {
        idx = -idx;
    }
    if (size == 1) {
        return 0;
    }
    // Period is 2*(size-1)
    IndexType period = 2 * (size - 1);
    idx = idx % period;
    if (idx >= size) {
        idx = period - idx;
    }
    return idx;
}

// Compute reflection sign for antisymmetric (odd) reflection
template <typename IndexType>
C10_HOST_DEVICE C10_ALWAYS_INLINE
int reflect_sign(IndexType idx, IndexType size) {
    if (size == 1) {
        return 1;
    }
    if (idx < 0) {
        idx = -idx;
    }
    IndexType period = 2 * (size - 1);
    IndexType cycle = idx / period;
    IndexType pos = idx % period;
    // Sign flips each time we cross a boundary
    int sign = (cycle % 2 == 0) ? 1 : -1;
    if (pos >= size) {
        sign = -sign;
    }
    return sign;
}

// Core index mapping for simple modes (not extrapolation)
// Returns (source_index, multiplier) where:
// - source_index is the input index to read from (-1 means use constant value)
// - multiplier is 1 for normal, -1 for antisymmetric reflection, 0 for constant
template <typename IndexType>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::pair<IndexType, int> map_index(
    IndexType out_idx,
    IndexType in_size,
    IndexType pad_before,
    PaddingMode mode
) {
    // Check if in valid region
    if (out_idx >= pad_before && out_idx < pad_before + in_size) {
        return {out_idx - pad_before, 1};
    }

    // Map index relative to input bounds
    IndexType rel_idx = out_idx - pad_before;

    switch (mode) {
        case PaddingMode::Constant:
            return {-1, 0};

        case PaddingMode::Replicate:
            if (rel_idx < 0) {
                return {0, 1};
            } else {
                return {in_size - 1, 1};
            }

        case PaddingMode::Reflect: {
            IndexType mapped = reflect_index(rel_idx, in_size);
            return {mapped, 1};
        }

        case PaddingMode::ReflectOdd: {
            IndexType mapped = reflect_index(rel_idx, in_size);
            int sign = reflect_sign(rel_idx, in_size);
            return {mapped, sign};
        }

        case PaddingMode::Circular: {
            IndexType mapped = ((rel_idx % in_size) + in_size) % in_size;
            return {mapped, 1};
        }

        default:
            // Extrapolation modes handled separately
            return {-1, 0};
    }
}

}  // namespace torchscience::kernel::pad
