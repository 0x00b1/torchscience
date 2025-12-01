#pragma once

#include <cmath>
#include <limits>
#include <tuple>

#include <c10/macros/Macros.h>

#include "../../probability/chi2_survival.h"

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Jarque-Bera test for normality.
 *
 * Tests whether sample data has skewness and kurtosis matching a normal distribution.
 *
 * JB = (n/6) * (S^2 + (K - 3)^2 / 4)
 *
 * where S is skewness and K is kurtosis.
 *
 * @param data Pointer to contiguous sample data
 * @param n Number of samples
 * @return Tuple of (JB statistic, p-value)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<T, T> jarque_bera(const T* data, int64_t n) {
    using torchscience::kernel::probability::chi2_survival;

    T nan = std::numeric_limits<T>::quiet_NaN();

    // Need at least 3 samples
    if (n < 3) {
        return std::make_tuple(nan, nan);
    }

    // Compute mean
    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    T mean = sum / T(n);

    // Compute central moments
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

    // Check for zero variance
    if (m2 <= T(0)) {
        return std::make_tuple(nan, nan);
    }

    // Skewness and kurtosis (Fisher's definition)
    T std_dev = std::sqrt(m2);
    T skewness = m3 / (std_dev * std_dev * std_dev);
    T kurtosis = m4 / (m2 * m2);
    T excess_kurtosis = kurtosis - T(3);

    // JB = (n/6) * (S^2 + (K-3)^2/4)
    T jb = (T(n) / T(6)) * (skewness * skewness + excess_kurtosis * excess_kurtosis / T(4));

    // P-value from chi-square survival function with df=2
    T pvalue = chi2_survival(jb, T(2));

    return std::make_tuple(jb, pvalue);
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
