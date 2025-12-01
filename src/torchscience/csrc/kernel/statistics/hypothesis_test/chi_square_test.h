#pragma once

#include <cmath>
#include <limits>
#include <tuple>

#include <c10/macros/Macros.h>

#include "../../probability/chi2_survival.h"

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Chi-square goodness-of-fit test.
 *
 * Tests whether observed frequencies differ from expected frequencies.
 *
 * chi2 = sum((observed - expected)^2 / expected)
 *
 * @param observed Pointer to observed frequencies
 * @param expected Pointer to expected frequencies (nullptr for uniform)
 * @param k Number of categories
 * @param ddof Delta degrees of freedom
 * @return Tuple of (chi-square statistic, p-value)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<T, T> chi_square_test(
    const T* observed,
    const T* expected,
    int64_t k,
    int64_t ddof
) {
    using torchscience::kernel::probability::chi2_survival;

    T nan = std::numeric_limits<T>::quiet_NaN();

    if (k < 1) {
        return std::make_tuple(nan, nan);
    }

    T df = T(k - 1 - ddof);
    if (df <= T(0)) {
        return std::make_tuple(nan, nan);
    }

    T chi2 = T(0);

    if (expected == nullptr) {
        // Uniform expected: each category has total/k expected frequency
        T total = T(0);
        for (int64_t i = 0; i < k; ++i) {
            total += observed[i];
        }
        T uniform_expected = total / T(k);

        if (uniform_expected <= T(0)) {
            return std::make_tuple(nan, nan);
        }

        for (int64_t i = 0; i < k; ++i) {
            T diff = observed[i] - uniform_expected;
            chi2 += (diff * diff) / uniform_expected;
        }
    } else {
        // Explicit expected frequencies
        for (int64_t i = 0; i < k; ++i) {
            if (expected[i] <= T(0)) {
                return std::make_tuple(nan, nan);
            }
            T diff = observed[i] - expected[i];
            chi2 += (diff * diff) / expected[i];
        }
    }

    T pvalue = chi2_survival(chi2, df);
    return std::make_tuple(chi2, pvalue);
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
