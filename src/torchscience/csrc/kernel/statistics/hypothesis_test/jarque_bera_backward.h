#pragma once

#include <cmath>
#include <limits>

#include <c10/macros/Macros.h>

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Backward pass for Jarque-Bera test.
 *
 * Computes gradients with respect to input data.
 * The JB statistic is:
 *   JB = (n/6) * (S^2 + (K-3)^2/4)
 *
 * where S = m3/(m2^1.5), K = m4/m2^2, and m_k are central moments.
 *
 * @param grad_statistic Gradient with respect to JB statistic
 * @param data Input data
 * @param grad_input Output gradient (must be pre-allocated)
 * @param n Number of samples
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void jarque_bera_backward(
    T grad_statistic,
    const T* data,
    T* grad_input,
    int64_t n
) {
    if (n < 3) {
        for (int64_t i = 0; i < n; ++i) {
            grad_input[i] = T(0);
        }
        return;
    }

    // Forward pass: compute mean and moments
    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    T mean = sum / T(n);

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

    if (m2 <= T(0)) {
        for (int64_t i = 0; i < n; ++i) {
            grad_input[i] = T(0);
        }
        return;
    }

    // Compute skewness and excess kurtosis
    T m2_sqrt = std::sqrt(m2);
    T m2_32 = m2 * m2_sqrt;  // m2^1.5
    T m2_sq = m2 * m2;       // m2^2

    T S = m3 / m2_32;        // skewness
    T K = m4 / m2_sq;        // kurtosis
    T E = K - T(3);          // excess kurtosis

    // JB = (n/6) * (S^2 + E^2/4)
    // dJB/dS = (n/6) * 2S = (n/3) * S
    // dJB/dE = (n/6) * (E/2) = (n/12) * E
    T dJB_dS = (T(n) / T(3)) * S;
    T dJB_dE = (T(n) / T(12)) * E;

    // S = m3 / m2^1.5
    // dS/dm3 = 1/m2^1.5
    // dS/dm2 = -1.5 * m3 / m2^2.5
    T dS_dm3 = T(1) / m2_32;
    T dS_dm2 = T(-1.5) * m3 / (m2_sq * m2_sqrt);

    // K = m4 / m2^2
    // dK/dm4 = 1/m2^2
    // dK/dm2 = -2 * m4 / m2^3
    T dK_dm4 = T(1) / m2_sq;
    T dK_dm2 = T(-2) * m4 / (m2_sq * m2);

    // dE/dm2 = dK/dm2, dE/dm4 = dK/dm4

    // Chain rule
    T dJB_dm2 = dJB_dS * dS_dm2 + dJB_dE * dK_dm2;
    T dJB_dm3 = dJB_dS * dS_dm3;
    T dJB_dm4 = dJB_dE * dK_dm4;

    // Compute gradients with respect to each data point
    // d = x_i - mean
    // dm2/dx_i = (2d/n) * (1 - 1/n) + (other terms summing to d-dependent contribution)
    // Using: dm_k/dx_i = (k * d^{k-1} * (1 - 1/n) - k * m_k) / n (approximately)
    // More precisely:
    // m_k = (1/n) * sum(d_j^k)
    // dm_k/dx_i = (1/n) * k * d_i^{k-1} * dd_i/dx_i + (1/n) * sum_{j!=i} k * d_j^{k-1} * dd_j/dx_i
    // where dd_i/dx_i = 1 - 1/n, dd_j/dx_i = -1/n (for j != i)
    //
    // Simplifying:
    // dm_k/dx_i = (k/n) * [ d_i^{k-1} * (1 - 1/n) - (1/n) * sum_{j!=i} d_j^{k-1} ]
    //           = (k/n) * [ d_i^{k-1} - (1/n) * sum_j d_j^{k-1} ]
    //           = (k/n) * [ d_i^{k-1} - (n * m_{k-1})/n ]  (for k even when m_{k-1} = m_{k-1})
    //
    // For simplicity, use the chain rule directly:
    // d_i = x_i - mean
    // dd_i/dx_i = 1 - 1/n (from d_mean/dx_i = 1/n)
    // dd_j/dx_i = -1/n for j != i
    //
    // m_k = (1/n) sum_j d_j^k
    // dm_k/dx_j = (1/n) * k * d_j^{k-1} * dd_j/dx_i
    // dm_k/dx_i = (1/n) * [ k * d_i^{k-1} * (1-1/n) + sum_{j!=i} k * d_j^{k-1} * (-1/n) ]
    //           = (k/n) * [ d_i^{k-1} - (1/n) * sum_j d_j^{k-1} ]

    // For numerical stability, we need m1 = mean of d^1 = 0
    // m_1 = 0 by definition (since d = x - mean)

    // Let S_k = (1/n) * sum_j d_j^k (i.e., S_k = m_k for k >= 1, with S_0 = 1)
    // We need S_1 = (1/n) * sum d = 0
    // S_3 = m3 (already computed, reuse)

    // Compute sum of d^k for k = 1, 3
    T S1 = T(0), S3 = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T d = data[i] - mean;
        S1 += d;
        S3 += d * d * d;
    }
    S1 /= T(n);  // Should be ~0
    S3 /= T(n);  // = m3

    for (int64_t i = 0; i < n; ++i) {
        T d = data[i] - mean;
        T d2 = d * d;
        T d3 = d2 * d;

        // dm2/dx_i = (2/n) * (d - S1) = (2/n) * d (since S1 ≈ 0)
        T dm2_dxi = (T(2) / T(n)) * d;

        // dm3/dx_i = (3/n) * (d^2 - S2) where S2 = m2
        T dm3_dxi = (T(3) / T(n)) * (d2 - m2);

        // dm4/dx_i = (4/n) * (d^3 - S3) where S3 = m3
        T dm4_dxi = (T(4) / T(n)) * (d3 - m3);

        grad_input[i] = grad_statistic * (
            dJB_dm2 * dm2_dxi +
            dJB_dm3 * dm3_dxi +
            dJB_dm4 * dm4_dxi
        );
    }
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
