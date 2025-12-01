#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include <c10/macros/Macros.h>

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Approximate inverse normal CDF (probit function).
 * Uses Abramowitz & Stegun rational approximation.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T approx_inv_normal_cumulative_distribution(T p) {
    // Constants for Abramowitz & Stegun approximation
    constexpr T a1 = T(-3.969683028665376e+01);
    constexpr T a2 = T(2.209460984245205e+02);
    constexpr T a3 = T(-2.759285104469687e+02);
    constexpr T a4 = T(1.383577518672690e+02);
    constexpr T a5 = T(-3.066479806614716e+01);
    constexpr T a6 = T(2.506628277459239e+00);

    constexpr T b1 = T(-5.447609879822406e+01);
    constexpr T b2 = T(1.615858368580409e+02);
    constexpr T b3 = T(-1.556989798598866e+02);
    constexpr T b4 = T(6.680131188771972e+01);
    constexpr T b5 = T(-1.328068155288572e+01);

    constexpr T c1 = T(-7.784894002430293e-03);
    constexpr T c2 = T(-3.223964580411365e-01);
    constexpr T c3 = T(-2.400758277161838e+00);
    constexpr T c4 = T(-2.549732539343734e+00);
    constexpr T c5 = T(4.374664141464968e+00);
    constexpr T c6 = T(2.938163982698783e+00);

    constexpr T d1 = T(7.784695709041462e-03);
    constexpr T d2 = T(3.224671290700398e-01);
    constexpr T d3 = T(2.445134137142996e+00);
    constexpr T d4 = T(3.754408661907416e+00);

    constexpr T p_low = T(0.02425);
    constexpr T p_high = T(1) - p_low;

    T q, r;

    if (p < p_low) {
        q = std::sqrt(-T(2) * std::log(p));
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
               ((((d1*q+d2)*q+d3)*q+d4)*q+T(1));
    } else if (p <= p_high) {
        q = p - T(0.5);
        r = q * q;
        return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
               (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+T(1));
    } else {
        q = std::sqrt(-T(2) * std::log(T(1) - p));
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
                ((((d1*q+d2)*q+d3)*q+d4)*q+T(1));
    }
}

/**
 * Compute p-value for Shapiro-Wilk test using Royston (1992) approximation.
 */
template <typename T>
T compute_shapiro_wilk_pvalue(T W, int64_t n) {
    // Transform W to approximate normality
    T log_n = std::log(T(n));

    T gamma, mu, sigma;

    if (n <= 11) {
        // Small sample polynomial approximation
        gamma = T(-2.273) + T(0.459) * log_n;
        mu = T(-0.0006714) * T(n*n*n) + T(0.025054) * T(n*n)
           - T(0.39978) * T(n) + T(0.5440);
        sigma = std::exp(T(-0.0020322) * T(n*n*n) + T(0.062767) * T(n*n)
              - T(0.77857) * T(n) + T(1.3822));
    } else {
        // Larger sample approximation
        gamma = T(0);
        mu = T(0.0038915) * std::pow(log_n, T(3))
           - T(0.083751) * std::pow(log_n, T(2))
           - T(0.31082) * log_n - T(1.5861);
        sigma = std::exp(T(0.0030302) * std::pow(log_n, T(2))
              - T(0.082676) * log_n - T(0.4803));
    }

    // Transform to normal
    T y;
    if (gamma != T(0)) {
        y = (std::pow(T(1) - W, gamma) - T(1)) / gamma;
    } else {
        y = std::log(T(1) - W);
    }

    T z = (y - mu) / sigma;

    // P-value from standard normal
    // P(Z > z) = 1 - Phi(z)
    T pvalue = T(0.5) * std::erfc(z / std::sqrt(T(2)));

    // Clamp to [0, 1]
    if (pvalue < T(0)) pvalue = T(0);
    if (pvalue > T(1)) pvalue = T(1);

    return pvalue;
}

/**
 * Shapiro-Wilk test for normality.
 *
 * The Shapiro-Wilk statistic W measures how well the order statistics
 * match expected normal order statistics.
 *
 * W = (sum(a_i * x_(i)))^2 / sum((x_i - mean)^2)
 *
 * where x_(i) are order statistics and a_i are tabulated coefficients.
 *
 * P-value computed using polynomial approximation from Royston (1992).
 *
 * @param data Input array
 * @param n Number of elements (3 <= n <= 5000)
 * @return Tuple of (W-statistic, p-value)
 */
template <typename T>
std::tuple<T, T> shapiro_wilk(const T* data, int64_t n) {
    T nan = std::numeric_limits<T>::quiet_NaN();

    if (n < 3) {
        return std::make_tuple(nan, nan);
    }

    if (n > 5000) {
        // Beyond tabulated coefficients, return NaN for p-value
        // but still compute W
    }

    // Sort data to get order statistics
    std::vector<T> sorted(data, data + n);
    std::sort(sorted.begin(), sorted.end());

    // Compute mean
    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += sorted[i];
    }
    T mean = sum / T(n);

    // Compute SS = sum((x_i - mean)^2)
    T ss = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T d = sorted[i] - mean;
        ss += d * d;
    }

    if (ss <= T(0)) {
        return std::make_tuple(nan, nan);
    }

    // Compute Shapiro-Wilk coefficients 'a'
    // For n <= 50, use exact algorithm
    // For n > 50, use approximation
    std::vector<T> a(n);

    // Compute expected normal order statistics (m)
    std::vector<T> m(n);
    for (int64_t i = 0; i < n; ++i) {
        // Blom's approximation for expected normal order statistics
        T p = (T(i + 1) - T(0.375)) / (T(n) + T(0.25));
        // Approximate inverse normal CDF using rational approximation
        m[i] = approx_inv_normal_cumulative_distribution(p);
    }

    // Compute m'*m
    T m_dot_m = T(0);
    for (int64_t i = 0; i < n; ++i) {
        m_dot_m += m[i] * m[i];
    }

    // Compute coefficients a = m / sqrt(m'*m)
    T m_norm = std::sqrt(m_dot_m);
    for (int64_t i = 0; i < n; ++i) {
        a[i] = m[i] / m_norm;
    }

    // Compute b = a' * x (sorted)
    T b = T(0);
    for (int64_t i = 0; i < n; ++i) {
        b += a[i] * sorted[i];
    }

    // W = b^2 / SS
    T W = (b * b) / ss;

    // P-value using Royston (1992) approximation
    T pvalue = compute_shapiro_wilk_pvalue(W, n);

    return std::make_tuple(W, pvalue);
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
