// src/torchscience/csrc/kernel/statistics/hypothesis_test/two_sample_t_test.h
#pragma once

#include <cmath>
#include <limits>
#include <tuple>

#include <c10/macros/Macros.h>

#include "t_test_common.h"

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Two-sample t-test.
 *
 * Tests if the means of two samples differ.
 *
 * Student's t-test (equal_var=true):
 *   Pooled variance: s_p^2 = ((n1-1)*s1^2 + (n2-1)*s2^2) / (n1 + n2 - 2)
 *   t = (mean1 - mean2) / (s_p * sqrt(1/n1 + 1/n2))
 *   df = n1 + n2 - 2
 *
 * Welch's t-test (equal_var=false):
 *   t = (mean1 - mean2) / sqrt(s1^2/n1 + s2^2/n2)
 *   Welch-Satterthwaite df:
 *     df = (s1^2/n1 + s2^2/n2)^2 / ((s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1))
 *
 * @param data1 First sample array
 * @param n1 Number of elements in first sample
 * @param data2 Second sample array
 * @param n2 Number of elements in second sample
 * @param equal_var If true, assume equal variances (Student's t-test)
 * @param alternative Alternative hypothesis
 * @return Tuple of (t_statistic, p_value, degrees_of_freedom)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<T, T, T> two_sample_t_test(
    const T* data1,
    int64_t n1,
    const T* data2,
    int64_t n2,
    bool equal_var,
    Alternative alternative
) {
    T nan = std::numeric_limits<T>::quiet_NaN();

    // Need at least 2 observations in each group
    if (n1 < 2 || n2 < 2) {
        return std::make_tuple(nan, nan, nan);
    }

    // Compute sample mean and variance for first sample
    T sum1 = T(0);
    for (int64_t i = 0; i < n1; ++i) {
        sum1 += data1[i];
    }
    T mean1 = sum1 / T(n1);

    T sum_sq1 = T(0);
    for (int64_t i = 0; i < n1; ++i) {
        T d = data1[i] - mean1;
        sum_sq1 += d * d;
    }
    T var1 = sum_sq1 / T(n1 - 1);

    // Compute sample mean and variance for second sample
    T sum2 = T(0);
    for (int64_t i = 0; i < n2; ++i) {
        sum2 += data2[i];
    }
    T mean2 = sum2 / T(n2);

    T sum_sq2 = T(0);
    for (int64_t i = 0; i < n2; ++i) {
        T d = data2[i] - mean2;
        sum_sq2 += d * d;
    }
    T var2 = sum_sq2 / T(n2 - 1);

    T t_stat, df;

    if (equal_var) {
        // Student's t-test with pooled variance
        T pooled_var = (T(n1 - 1) * var1 + T(n2 - 1) * var2) / T(n1 + n2 - 2);

        if (pooled_var <= T(0)) {
            return std::make_tuple(nan, nan, nan);
        }

        T se = std::sqrt(pooled_var * (T(1) / T(n1) + T(1) / T(n2)));
        t_stat = (mean1 - mean2) / se;
        df = T(n1 + n2 - 2);
    } else {
        // Welch's t-test
        T v1_n1 = var1 / T(n1);
        T v2_n2 = var2 / T(n2);
        T vn_sum = v1_n1 + v2_n2;

        if (vn_sum <= T(0)) {
            return std::make_tuple(nan, nan, nan);
        }

        T se = std::sqrt(vn_sum);
        t_stat = (mean1 - mean2) / se;

        // Welch-Satterthwaite degrees of freedom
        T numerator = vn_sum * vn_sum;
        T denom1 = (v1_n1 * v1_n1) / T(n1 - 1);
        T denom2 = (v2_n2 * v2_n2) / T(n2 - 1);
        df = numerator / (denom1 + denom2);
    }

    T pvalue = t_pvalue(t_stat, df, alternative);

    return std::make_tuple(t_stat, pvalue, df);
}

/**
 * Overload that takes pre-computed sample statistics.
 *
 * @param mean1 Mean of first sample
 * @param var1 Variance of first sample (with n1-1 denominator)
 * @param n1 Number of elements in first sample
 * @param mean2 Mean of second sample
 * @param var2 Variance of second sample (with n2-1 denominator)
 * @param n2 Number of elements in second sample
 * @param equal_var If true, assume equal variances
 * @param alternative Alternative hypothesis
 * @return Tuple of (t_statistic, p_value, degrees_of_freedom)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<T, T, T> two_sample_t_test_from_stats(
    T mean1,
    T var1,
    int64_t n1,
    T mean2,
    T var2,
    int64_t n2,
    bool equal_var,
    Alternative alternative
) {
    T nan = std::numeric_limits<T>::quiet_NaN();

    if (n1 < 2 || n2 < 2) {
        return std::make_tuple(nan, nan, nan);
    }

    T t_stat, df;

    if (equal_var) {
        T pooled_var = (T(n1 - 1) * var1 + T(n2 - 1) * var2) / T(n1 + n2 - 2);

        if (pooled_var <= T(0)) {
            return std::make_tuple(nan, nan, nan);
        }

        T se = std::sqrt(pooled_var * (T(1) / T(n1) + T(1) / T(n2)));
        t_stat = (mean1 - mean2) / se;
        df = T(n1 + n2 - 2);
    } else {
        T v1_n1 = var1 / T(n1);
        T v2_n2 = var2 / T(n2);
        T vn_sum = v1_n1 + v2_n2;

        if (vn_sum <= T(0)) {
            return std::make_tuple(nan, nan, nan);
        }

        T se = std::sqrt(vn_sum);
        t_stat = (mean1 - mean2) / se;

        T numerator = vn_sum * vn_sum;
        T denom1 = (v1_n1 * v1_n1) / T(n1 - 1);
        T denom2 = (v2_n2 * v2_n2) / T(n2 - 1);
        df = numerator / (denom1 + denom2);
    }

    T pvalue = t_pvalue(t_stat, df, alternative);

    return std::make_tuple(t_stat, pvalue, df);
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
