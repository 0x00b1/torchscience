#pragma once

#include <cmath>
#include <vector>
#include <limits>
#include <tuple>

#include <c10/macros/Macros.h>

#include "ranking.h"
#include "t_test_common.h"
#include "../../probability/normal_survival.h"
#include "../../probability/normal_cumulative_distribution.h"

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Mann-Whitney U test (Wilcoxon rank-sum test).
 *
 * Tests whether two independent samples come from the same distribution.
 *
 * @param x First sample
 * @param n1 Size of first sample
 * @param y Second sample
 * @param n2 Size of second sample
 * @param alternative Alternative hypothesis
 * @return Tuple of (U-statistic, p-value)
 */
template <typename T>
std::tuple<T, T> mann_whitney_u(
    const T* x,
    int64_t n1,
    const T* y,
    int64_t n2,
    Alternative alternative
) {
    using torchscience::kernel::probability::normal_survival;
    using torchscience::kernel::probability::normal_cumulative_distribution;

    T nan = std::numeric_limits<T>::quiet_NaN();

    if (n1 < 1 || n2 < 1) {
        return std::make_tuple(nan, nan);
    }

    int64_t n = n1 + n2;

    // Combine samples and compute ranks
    std::vector<T> combined(n);
    for (int64_t i = 0; i < n1; ++i) {
        combined[i] = x[i];
    }
    for (int64_t i = 0; i < n2; ++i) {
        combined[n1 + i] = y[i];
    }

    std::vector<T> ranks(n);
    compute_ranks(combined.data(), n, ranks.data());

    // Sum of ranks for first sample
    T R1 = T(0);
    for (int64_t i = 0; i < n1; ++i) {
        R1 += ranks[i];
    }

    // U statistic: U1 = R1 - n1*(n1+1)/2
    T U1 = R1 - T(n1) * T(n1 + 1) / T(2);
    T U2 = T(n1) * T(n2) - U1;

    // Use smaller U for two-sided test
    T U = (U1 < U2) ? U1 : U2;

    // Expected value and variance under null hypothesis
    T mu = T(n1) * T(n2) / T(2);

    // Variance with tie correction
    T tie_corr = tie_correction(ranks.data(), n);
    T sigma_sq = T(n1) * T(n2) * T(n + 1) / T(12) * tie_corr;

    if (sigma_sq <= T(0)) {
        return std::make_tuple(U1, nan);
    }

    T sigma = std::sqrt(sigma_sq);

    // Z-score (with continuity correction)
    T z;
    T pvalue;

    // Standard normal: loc=0, scale=1
    T loc = T(0);
    T scale = T(1);

    if (alternative == Alternative::TWO_SIDED) {
        z = (U - mu + T(0.5)) / sigma;  // continuity correction
        pvalue = T(2) * normal_cumulative_distribution(-std::abs(z), loc, scale);
    } else if (alternative == Alternative::LESS) {
        z = (U1 - mu + T(0.5)) / sigma;
        pvalue = normal_cumulative_distribution(z, loc, scale);
    } else {  // GREATER
        z = (U1 - mu - T(0.5)) / sigma;
        pvalue = normal_survival(z, loc, scale);
    }

    return std::make_tuple(U1, pvalue);
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
