// src/torchscience/csrc/kernel/statistics/hypothesis_test/paired_t_test.h
#pragma once

#include <cmath>
#include <limits>
#include <tuple>

#include <c10/macros/Macros.h>

#include "t_test_common.h"
#include "one_sample_t_test.h"

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Paired t-test.
 *
 * Tests if the mean difference between paired observations is zero.
 * This is equivalent to a one-sample t-test on the differences with popmean=0.
 *
 * Formula:
 *   d_i = x1_i - x2_i
 *   t = mean(d) / (std(d) / sqrt(n))
 *   df = n - 1
 *
 * @param data1 First sample array
 * @param data2 Second sample array (must have same length as data1)
 * @param n Number of paired observations
 * @param alternative Alternative hypothesis
 * @return Tuple of (t_statistic, p_value, degrees_of_freedom)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<T, T, T> paired_t_test(
    const T* data1,
    const T* data2,
    int64_t n,
    Alternative alternative
) {
    T nan = std::numeric_limits<T>::quiet_NaN();

    // Need at least 2 observations
    if (n < 2) {
        return std::make_tuple(nan, nan, nan);
    }

    // Compute differences and their mean
    T sum_diff = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum_diff += data1[i] - data2[i];
    }
    T mean_diff = sum_diff / T(n);

    // Compute variance of differences
    T sum_sq = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T diff = data1[i] - data2[i];
        T d = diff - mean_diff;
        sum_sq += d * d;
    }
    T variance = sum_sq / T(n - 1);

    // Check for zero variance
    if (variance <= T(0)) {
        return std::make_tuple(nan, nan, nan);
    }

    T std_dev = std::sqrt(variance);
    T sqrt_n = std::sqrt(T(n));

    // Compute t-statistic (testing against popmean = 0)
    T t_stat = mean_diff / (std_dev / sqrt_n);

    // Degrees of freedom
    T df = T(n - 1);

    // Compute p-value
    T pvalue = t_pvalue(t_stat, df, alternative);

    return std::make_tuple(t_stat, pvalue, df);
}

/**
 * Paired t-test using pre-computed difference array.
 *
 * Equivalent to one_sample_t_test with popmean=0.
 *
 * @param differences Array of paired differences (d_i = x1_i - x2_i)
 * @param n Number of differences
 * @param alternative Alternative hypothesis
 * @return Tuple of (t_statistic, p_value, degrees_of_freedom)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<T, T, T> paired_t_test_from_differences(
    const T* differences,
    int64_t n,
    Alternative alternative
) {
    return one_sample_t_test(differences, n, T(0), alternative);
}

/**
 * Overload that takes pre-computed statistics of the differences.
 *
 * @param mean_diff Mean of differences
 * @param std_diff Standard deviation of differences (with n-1 denominator)
 * @param n Number of paired observations
 * @param alternative Alternative hypothesis
 * @return Tuple of (t_statistic, p_value, degrees_of_freedom)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<T, T, T> paired_t_test_from_stats(
    T mean_diff,
    T std_diff,
    int64_t n,
    Alternative alternative
) {
    return one_sample_t_test_from_stats(mean_diff, std_diff, n, T(0), alternative);
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
