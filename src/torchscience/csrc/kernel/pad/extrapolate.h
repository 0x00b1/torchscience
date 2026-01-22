#pragma once

#include <cmath>
#include <cstdint>

#include <c10/macros/Macros.h>

#include "pad.h"

namespace torchscience::kernel::pad {

template <typename T>
struct ExtrapolationWeights {
    int num_points;
    T weights[MAX_EXTRAPOLATION_ORDER + 1];
    int offset;  // Starting offset from edge (0 = edge value)
};

// Linear extrapolation: f(x) = f(0) + x * (f(0) - f(-1))
// At distance d from edge: output = (1+d)*f(0) - d*f(1)
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
ExtrapolationWeights<T> linear_weights(T distance_from_edge) {
    ExtrapolationWeights<T> result;
    result.num_points = 2;
    result.offset = 0;
    result.weights[0] = T(1) + distance_from_edge;
    result.weights[1] = -distance_from_edge;
    for (int i = 2; i <= MAX_EXTRAPOLATION_ORDER; ++i) {
        result.weights[i] = T(0);
    }
    return result;
}

// Polynomial extrapolation using Lagrange interpolation
// Uses order+1 points from the edge to fit polynomial of degree order
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
ExtrapolationWeights<T> polynomial_weights(T distance_from_edge, int order) {
    ExtrapolationWeights<T> result;
    result.num_points = order + 1;
    result.offset = 0;

    // Initialize all weights to zero
    for (int i = 0; i <= MAX_EXTRAPOLATION_ORDER; ++i) {
        result.weights[i] = T(0);
    }

    // Lagrange basis polynomials evaluated at -distance_from_edge
    // We're extrapolating to position -distance (before the edge at 0)
    // Using points at positions 0, 1, 2, ..., order
    T x = -distance_from_edge;  // The extrapolation point

    for (int i = 0; i <= order; ++i) {
        T basis = T(1);
        for (int j = 0; j <= order; ++j) {
            if (i != j) {
                basis *= (x - T(j)) / (T(i) - T(j));
            }
        }
        result.weights[i] = basis;
    }

    return result;
}

// Smooth extrapolation: C1 continuous (matches value and derivative)
// Same as linear for basic implementation
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
ExtrapolationWeights<T> smooth_weights(T distance_from_edge) {
    return linear_weights(distance_from_edge);
}

// Spline extrapolation: cubic spline continuation
// For simplicity, uses cubic polynomial (order=3)
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
ExtrapolationWeights<T> spline_weights(T distance_from_edge) {
    return polynomial_weights(distance_from_edge, 3);
}

// Get extrapolation weights based on mode
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
ExtrapolationWeights<T> get_extrapolation_weights(
    T distance_from_edge,
    PaddingMode mode,
    int order
) {
    switch (mode) {
        case PaddingMode::Linear:
            return linear_weights(distance_from_edge);
        case PaddingMode::Polynomial:
            return polynomial_weights(distance_from_edge, order);
        case PaddingMode::Spline:
            return spline_weights(distance_from_edge);
        case PaddingMode::Smooth:
            return smooth_weights(distance_from_edge);
        default:
            // Should not reach here
            return linear_weights(distance_from_edge);
    }
}

}  // namespace torchscience::kernel::pad
