#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include <c10/macros/Macros.h>

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Standard normal CDF using error function.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T normal_cumulative_distribution(T x) {
    return T(0.5) * (T(1) + std::erf(x / std::sqrt(T(2))));
}

/**
 * Anderson-Darling test for normality.
 *
 * A^2 = -n - (1/n) * sum_{i=1}^{n} (2i-1) * [ln(F(Y_i)) + ln(1-F(Y_{n+1-i}))]
 *
 * where Y_i are standardized order statistics and F is the standard normal CDF.
 *
 * @param data Input array
 * @param n Number of elements
 * @return Tuple of (A^2 statistic, array of 5 critical values)
 *
 * Critical values are for significance levels: 15%, 10%, 5%, 2.5%, 1%
 */
template <typename T>
std::tuple<T, std::array<T, 5>> anderson_darling(const T* data, int64_t n) {
    T nan = std::numeric_limits<T>::quiet_NaN();

    // Critical values for normal distribution at 15%, 10%, 5%, 2.5%, 1%
    std::array<T, 5> critical_values = {
        T(0.576),  // 15%
        T(0.656),  // 10%
        T(0.787),  // 5%
        T(0.918),  // 2.5%
        T(1.092)   // 1%
    };

    if (n < 8) {
        std::array<T, 5> nan_cv;
        for (int i = 0; i < 5; ++i) {
            nan_cv[i] = nan;
        }
        return std::make_tuple(nan, nan_cv);
    }

    // Compute mean and std
    T sum = T(0);
    for (int64_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    T mean = sum / T(n);

    T ss = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T d = data[i] - mean;
        ss += d * d;
    }
    T std_dev = std::sqrt(ss / T(n - 1));

    if (std_dev <= T(0)) {
        std::array<T, 5> nan_cv;
        for (int i = 0; i < 5; ++i) {
            nan_cv[i] = nan;
        }
        return std::make_tuple(nan, nan_cv);
    }

    // Standardize and sort
    std::vector<T> y(n);
    for (int64_t i = 0; i < n; ++i) {
        y[i] = (data[i] - mean) / std_dev;
    }
    std::sort(y.begin(), y.end());

    // Compute A^2
    T A2 = T(0);
    for (int64_t i = 0; i < n; ++i) {
        T F_yi = normal_cumulative_distribution(y[i]);
        T F_yn_minus_i = normal_cumulative_distribution(y[n - 1 - i]);

        // Clamp to avoid log(0)
        F_yi = std::max(F_yi, T(1e-15));
        F_yn_minus_i = std::max(F_yn_minus_i, T(1e-15));
        F_yi = std::min(F_yi, T(1) - T(1e-15));
        F_yn_minus_i = std::min(F_yn_minus_i, T(1) - T(1e-15));

        T coef = T(2 * (i + 1) - 1);
        A2 += coef * (std::log(F_yi) + std::log(T(1) - F_yn_minus_i));
    }

    A2 = -T(n) - A2 / T(n);

    // Apply sample size correction (Stephens 1974)
    A2 = A2 * (T(1) + T(0.75) / T(n) + T(2.25) / T(n * n));

    return std::make_tuple(A2, critical_values);
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
