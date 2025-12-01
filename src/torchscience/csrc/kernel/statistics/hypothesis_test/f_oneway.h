#pragma once

#include <cmath>
#include <limits>
#include <tuple>

#include <c10/macros/Macros.h>

#include "../../probability/f_survival.h"

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * One-way ANOVA F-test.
 *
 * Tests whether multiple groups have equal means.
 *
 * Formula:
 *   F = (between-group variance) / (within-group variance)
 *     = (SS_between / df_between) / (SS_within / df_within)
 *
 * where:
 *   SS_between = sum(n_i * (mean_i - grand_mean)^2)
 *   SS_within = sum((x_ij - mean_i)^2)
 *   df_between = k - 1  (k = number of groups)
 *   df_within = N - k   (N = total samples)
 *
 * @param data Pointer to data array (groups concatenated)
 * @param group_sizes Array of group sizes
 * @param k Number of groups
 * @return Tuple of (F-statistic, p-value)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<T, T> f_oneway(
    const T* data,
    const int64_t* group_sizes,
    int64_t k
) {
    using torchscience::kernel::probability::f_survival;

    T nan = std::numeric_limits<T>::quiet_NaN();

    if (k < 2) {
        return std::make_tuple(nan, nan);
    }

    // Calculate total N and grand mean
    int64_t N = 0;
    T grand_sum = T(0);

    int64_t offset = 0;
    for (int64_t g = 0; g < k; ++g) {
        int64_t n_g = group_sizes[g];
        N += n_g;
        for (int64_t i = 0; i < n_g; ++i) {
            grand_sum += data[offset + i];
        }
        offset += n_g;
    }

    if (N <= k) {
        return std::make_tuple(nan, nan);
    }

    T grand_mean = grand_sum / T(N);

    // Calculate group means and SS_between, SS_within
    T ss_between = T(0);
    T ss_within = T(0);

    offset = 0;
    for (int64_t g = 0; g < k; ++g) {
        int64_t n_g = group_sizes[g];

        // Group mean
        T group_sum = T(0);
        for (int64_t i = 0; i < n_g; ++i) {
            group_sum += data[offset + i];
        }
        T group_mean = group_sum / T(n_g);

        // SS_between contribution
        T diff = group_mean - grand_mean;
        ss_between += T(n_g) * diff * diff;

        // SS_within contribution
        for (int64_t i = 0; i < n_g; ++i) {
            T d = data[offset + i] - group_mean;
            ss_within += d * d;
        }

        offset += n_g;
    }

    // Degrees of freedom
    T df_between = T(k - 1);
    T df_within = T(N - k);

    if (df_within <= T(0) || ss_within <= T(0)) {
        return std::make_tuple(nan, nan);
    }

    // Mean squares
    T ms_between = ss_between / df_between;
    T ms_within = ss_within / df_within;

    // F-statistic
    T F = ms_between / ms_within;

    // P-value from F-distribution survival function
    T pvalue = f_survival(F, df_between, df_within);

    return std::make_tuple(F, pvalue);
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
