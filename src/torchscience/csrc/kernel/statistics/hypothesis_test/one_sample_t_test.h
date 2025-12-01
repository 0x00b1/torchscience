// src/torchscience/csrc/kernel/statistics/hypothesis_test/one_sample_t_test.h
#pragma once

#include <cmath>
#include <limits>
#include <tuple>

#include <c10/macros/Macros.h>

#include "t_test_common.h"

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * One-sample t-test.
 *
 * Tests if the mean of a sample differs from a population mean (popmean).
 *
 * Formula:
 *   t = (sample_mean - popmean) / (sample_std / sqrt(n))
 *   df = n - 1
 *
 * @param data Input array of n elements
 * @param n Number of elements
 * @param popmean Population mean to test against (null hypothesis)
 * @param alternative Alternative hypothesis: "two-sided", "less", or "greater"
 * @return Tuple of (t_statistic, p_value, degrees_of_freedom)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<T, T, T> one_sample_t_test(
    const T* data,
    int64_t n,
    T popmean,
    Alternative alternative
) {
    T nan = std::numeric_limits<T>::quiet_NaN();

    // Need at least 2 observations for variance calculation
    if (n < 2) {
        return std::make_tuple(nan, nan, nan);
    }

    // Compute sample mean
    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    T mean = sum / T(n);

    // Compute sample variance (unbiased estimator with n-1 denominator)
    T sum_sq = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T d = data[i] - mean;
        sum_sq += d * d;
    }
    T variance = sum_sq / T(n - 1);

    // Check for zero variance
    if (variance <= T(0)) {
        return std::make_tuple(nan, nan, nan);
    }

    T std_dev = std::sqrt(variance);
    T sqrt_n = std::sqrt(T(n));

    // Compute t-statistic
    T t_stat = (mean - popmean) / (std_dev / sqrt_n);

    // Degrees of freedom
    T df = T(n - 1);

    // Compute p-value
    T pvalue = t_pvalue(t_stat, df, alternative);

    return std::make_tuple(t_stat, pvalue, df);
}

/**
 * Overload that takes pre-computed sample statistics.
 *
 * @param mean Sample mean
 * @param std_dev Sample standard deviation (with n-1 denominator)
 * @param n Number of elements
 * @param popmean Population mean to test against
 * @param alternative Alternative hypothesis
 * @return Tuple of (t_statistic, p_value, degrees_of_freedom)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<T, T, T> one_sample_t_test_from_stats(
    T mean,
    T std_dev,
    int64_t n,
    T popmean,
    Alternative alternative
) {
    T nan = std::numeric_limits<T>::quiet_NaN();

    if (n < 2) {
        return std::make_tuple(nan, nan, nan);
    }

    if (std_dev <= T(0)) {
        return std::make_tuple(nan, nan, nan);
    }

    T sqrt_n = std::sqrt(T(n));
    T t_stat = (mean - popmean) / (std_dev / sqrt_n);
    T df = T(n - 1);
    T pvalue = t_pvalue(t_stat, df, alternative);

    return std::make_tuple(t_stat, pvalue, df);
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
