// src/torchscience/csrc/kernel/statistics/descriptive/kurtosis.h
#pragma once

#include <cmath>
#include <limits>

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

namespace torchscience::kernel::statistics::descriptive {

// ============================================================================
// Forward: Compute kurtosis for a contiguous array
// ============================================================================

/**
 * Compute kurtosis of a 1D array using two-pass algorithm.
 *
 * @param data Input array of n elements
 * @param n Number of elements
 * @param fisher If true, compute excess kurtosis (subtract 3)
 * @param bias If true, return biased estimate; if false, apply correction
 * @return Kurtosis value (or NaN if n <= 3 for unbiased, or n < 2)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T kurtosis(
    const T* data,
    int64_t n,
    bool fisher,
    bool bias
) {
    if (n < 2) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (!bias && n <= 3) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    T mean = sum / T(n);

    T m2 = T(0);
    T m4 = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T d = data[i] - mean;
        T d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= T(n);
    m4 /= T(n);

    if (m2 == T(0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T m2_sq = m2 * m2;
    T g2 = m4 / m2_sq;
    if (fisher) {
        g2 -= T(3);
    }

    if (!bias) {
        T n_f = T(n);
        T correction = ((n_f - T(1)) / ((n_f - T(2)) * (n_f - T(3)))) *
                      ((n_f + T(1)) * g2 + T(6));
        return correction;
    }

    return g2;
}

/**
 * Compute kurtosis of complex magnitudes.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T kurtosis_complex(
    const c10::complex<T>* data,
    int64_t n,
    bool fisher,
    bool bias
) {
    if (n < 2) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (!bias && n <= 3) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += std::abs(data[i]);
    }
    T mean = sum / T(n);

    T m2 = T(0);
    T m4 = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T mag = std::abs(data[i]);
        T d = mag - mean;
        T d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= T(n);
    m4 /= T(n);

    if (m2 == T(0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T m2_sq = m2 * m2;
    T g2 = m4 / m2_sq;
    if (fisher) {
        g2 -= T(3);
    }

    if (!bias) {
        T n_f = T(n);
        T correction = ((n_f - T(1)) / ((n_f - T(2)) * (n_f - T(3)))) *
                      ((n_f + T(1)) * g2 + T(6));
        return correction;
    }

    return g2;
}

// ============================================================================
// Backward: Compute gradient of kurtosis w.r.t. input
// ============================================================================

/**
 * Compute gradient of kurtosis w.r.t. input array.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void kurtosis_backward(
    T grad_output,
    const T* data,
    int64_t n,
    bool fisher,
    bool bias,
    T* grad_input
) {
    if (n < 2 || (!bias && n <= 3)) {
        for (int64_t i = 0; i < n; ++i) {
            grad_input[i] = T(0);
        }
        return;
    }

    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    T mean = sum / T(n);

    T m2 = T(0);
    T m3 = T(0);
    T m4 = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T d = data[i] - mean;
        T d2 = d * d;
        T d3 = d2 * d;
        m2 += d2;
        m3 += d3;
        m4 += d2 * d2;
    }
    m2 /= T(n);
    m3 /= T(n);
    m4 /= T(n);

    if (m2 == T(0)) {
        for (int64_t i = 0; i < n; ++i) {
            grad_input[i] = T(0);
        }
        return;
    }

    T m2_sq = m2 * m2;
    T g2 = m4 / m2_sq;
    T coeff = T(4) / (T(n) * m2_sq);

    if (bias) {
        for (int64_t i = 0; i < n; ++i) {
            T d = data[i] - mean;
            T d3 = d * d * d;
            grad_input[i] = grad_output * coeff * (d3 - m3 - g2 * m2 * d);
        }
    } else {
        T n_f = T(n);
        T dG2_dg2 = ((n_f - T(1)) * (n_f + T(1))) / ((n_f - T(2)) * (n_f - T(3)));

        for (int64_t i = 0; i < n; ++i) {
            T d = data[i] - mean;
            T d3 = d * d * d;
            T dg2_dxi = coeff * (d3 - m3 - g2 * m2 * d);
            grad_input[i] = grad_output * dG2_dg2 * dg2_dxi;
        }
    }
}

/**
 * Compute gradient for complex input (magnitudes).
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void kurtosis_backward_complex(
    T grad_output,
    const c10::complex<T>* data,
    int64_t n,
    bool fisher,
    bool bias,
    c10::complex<T>* grad_input
) {
    if (n < 2 || (!bias && n <= 3)) {
        for (int64_t i = 0; i < n; ++i) {
            grad_input[i] = c10::complex<T>(T(0), T(0));
        }
        return;
    }

    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += std::abs(data[i]);
    }
    T mean = sum / T(n);

    T m2 = T(0);
    T m3 = T(0);
    T m4 = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T mag = std::abs(data[i]);
        T d = mag - mean;
        T d2 = d * d;
        T d3 = d2 * d;
        m2 += d2;
        m3 += d3;
        m4 += d2 * d2;
    }
    m2 /= T(n);
    m3 /= T(n);
    m4 /= T(n);

    if (m2 == T(0)) {
        for (int64_t i = 0; i < n; ++i) {
            grad_input[i] = c10::complex<T>(T(0), T(0));
        }
        return;
    }

    T m2_sq = m2 * m2;
    T g2 = m4 / m2_sq;
    T g2_excess = g2 - T(3);

    T k_for_grad = fisher ? g2_excess : g2;
    T coeff = T(4) / (T(n) * m2_sq);

    T dG2_dg2 = T(1);
    if (!bias) {
        T n_f = T(n);
        dG2_dg2 = ((n_f - T(1)) * (n_f + T(1))) / ((n_f - T(2)) * (n_f - T(3)));
    }

    for (int64_t i = 0; i < n; ++i) {
        T mag = std::abs(data[i]);
        T d = mag - mean;
        T d3 = d * d * d;
        T dk_dmag = coeff * (d3 - m3 - T(2) * k_for_grad * m2 * d);
        if (!bias) {
            dk_dmag *= dG2_dg2;
        }

        if (mag > T(0)) {
            T grad_mag = grad_output * dk_dmag;
            grad_input[i] = c10::complex<T>(
                grad_mag * data[i].real() / mag,
                grad_mag * data[i].imag() / mag
            );
        } else {
            grad_input[i] = c10::complex<T>(T(0), T(0));
        }
    }
}

// ============================================================================
// Double Backward: Second-order derivatives
// ============================================================================

/**
 * Compute second-order derivatives for kurtosis.
 *
 * Uses O(n) algorithm by precomputing weighted sums instead of O(n²) nested loops.
 *
 * The key insight is that the Hessian terms can be decomposed into:
 *   Σᵢ gg_i * ∂²k/∂xᵢ∂xⱼ = f(j, precomputed_sums)
 *
 * where precomputed_sums = {Σgg, Σgg·d, Σgg·d², Σgg·d³} are computed once.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void kurtosis_backward_backward(
    const T* grad_grad_input,
    T grad_output,
    const T* data,
    int64_t n,
    bool fisher,
    bool bias,
    T& grad_grad_output,
    T* new_grad_input
) {
    grad_grad_output = T(0);
    for (int64_t i = 0; i < n; ++i) {
        new_grad_input[i] = T(0);
    }

    if (n < 2 || (!bias && n <= 3)) {
        return;
    }

    // Compute mean
    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    T mean = sum / T(n);

    // Compute moments
    T m2 = T(0), m3 = T(0), m4 = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T d = data[i] - mean;
        T d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= T(n);
    m3 /= T(n);
    m4 /= T(n);

    if (m2 == T(0)) {
        return;
    }

    T m2_sq = m2 * m2;
    T g2 = m4 / m2_sq;  // Pearson kurtosis
    T n_f = T(n);
    T coeff = T(4) / (n_f * m2_sq);

    T dG2_dg2 = T(1);
    if (!bias) {
        dG2_dg2 = ((n_f - T(1)) * (n_f + T(1))) / ((n_f - T(2)) * (n_f - T(3)));
    }

    // Precompute weighted sums - O(n)
    T sum_gg = T(0);
    T sum_gg_d = T(0);
    T sum_gg_d2 = T(0);
    T sum_gg_d3 = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T d = data[i] - mean;
        T d2 = d * d;
        T d3 = d2 * d;
        T gg = grad_grad_input[i];
        sum_gg += gg;
        sum_gg_d += gg * d;
        sum_gg_d2 += gg * d2;
        sum_gg_d3 += gg * d3;
    }

    // grad_grad_output = Σᵢ gg_i * dk/dxᵢ
    // where dk/dxᵢ = coeff * (dᵢ³ - m₃ - g₂·m₂·dᵢ)
    T g2_m2 = g2 * m2;
    T sum_f = sum_gg_d3 - m3 * sum_gg - g2_m2 * sum_gg_d;
    grad_grad_output = coeff * sum_f;
    if (!bias) {
        grad_grad_output *= dG2_dg2;
    }

    // Precompute constants for Hessian
    T inv_n = T(1) / n_f;

    // new_grad_input[j] = grad_output * Σᵢ(gg_i * ∂²k/∂xᵢ∂xⱼ)
    //
    // The Hessian decomposes into 4 terms (see derivation below).
    // Each term becomes O(1) per j using precomputed sums.
    //
    // Term 1: ∂(dᵢ³)/∂xⱼ = 3dᵢ²(δᵢⱼ - 1/n)
    //   → 3·gg_j·dⱼ² - (3/n)·sum_gg_d2
    //
    // Term 2: -∂m₃/∂xⱼ = -(3/n)(dⱼ² - m₂)
    //   → -(3/n)(dⱼ² - m₂)·sum_gg
    //
    // Term 3: -∂(g₂·m₂·dᵢ)/∂xⱼ
    //   → -dg₂/dxⱼ·m₂·sum_gg_d - g₂·dm₂/dxⱼ·sum_gg_d
    //      - g₂·m₂·gg_j + (g₂·m₂/n)·sum_gg
    //
    // Term 4: (∂coeff/∂xⱼ)·Σᵢ gg_i·fᵢ = dcoeff/dxⱼ · sum_f

    for (int64_t j = 0; j < n; ++j) {
        T dj = data[j] - mean;
        T dj2 = dj * dj;
        T dj3 = dj2 * dj;
        T gg_j = grad_grad_input[j];

        // Term 1: ∂(dᵢ³)/∂xⱼ contribution
        T term1 = T(3) * gg_j * dj2 - T(3) * inv_n * sum_gg_d2;

        // Term 2: -∂m₃/∂xⱼ contribution
        T dm3_dxj = T(3) * inv_n * (dj2 - m2);
        T term2 = -dm3_dxj * sum_gg;

        // Term 3: -∂(g₂·m₂·dᵢ)/∂xⱼ contribution
        T dg2_dxj = coeff * (dj3 - m3 - g2_m2 * dj);
        T dm2_dxj = T(2) * inv_n * dj;
        T term3 = -dg2_dxj * m2 * sum_gg_d
                  - g2 * dm2_dxj * sum_gg_d
                  - g2_m2 * gg_j
                  + g2_m2 * inv_n * sum_gg;

        // Term 4: (∂coeff/∂xⱼ) · sum_f
        T dcoeff_dxj = -T(2) * coeff / m2 * dm2_dxj;
        T term4 = dcoeff_dxj * sum_f;

        T hessian_contrib = coeff * (term1 + term2 + term3) + term4;
        if (!bias) {
            hessian_contrib *= dG2_dg2;
        }

        new_grad_input[j] = grad_output * hessian_contrib;
    }
}

}  // namespace torchscience::kernel::statistics::descriptive
