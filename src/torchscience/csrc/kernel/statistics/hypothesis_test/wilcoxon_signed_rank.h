#pragma once

#include <cmath>
#include <vector>
#include <limits>
#include <tuple>
#include <algorithm>

#include <c10/macros/Macros.h>

#include "ranking.h"
#include "t_test_common.h"
#include "../../probability/normal_survival.h"
#include "../../probability/normal_cumulative_distribution.h"

namespace torchscience::kernel::statistics::hypothesis_test {

enum class ZeroMethod { Wilcox, Pratt, Zsplit };

/**
 * Compute tie term for Wilcoxon signed-rank test.
 * Returns sum(t^3 - t) where t is the count of each tie group.
 */
template <typename T>
T wilcoxon_tie_term(const T* abs_d, int64_t n) {
    // Sort absolute differences to find ties
    std::vector<T> sorted_abs(abs_d, abs_d + n);
    std::sort(sorted_abs.begin(), sorted_abs.end());

    T tie_term = T(0);
    int64_t i = 0;
    while (i < n) {
        int64_t count = 1;
        while (i + count < n && sorted_abs[i] == sorted_abs[i + count]) {
            ++count;
        }
        if (count > 1) {
            T t = T(count);
            tie_term += t * t * t - t;
        }
        i += count;
    }
    return tie_term;
}

/**
 * Wilcoxon signed-rank test.
 *
 * Tests whether the median of a sample (or differences) is zero.
 *
 * @param d Differences array (x - y, or just x for one-sample)
 * @param n Number of elements
 * @param alternative Alternative hypothesis
 * @param zero_method How to handle zero differences
 * @return Tuple of (W-statistic, p-value)
 */
template <typename T>
std::tuple<T, T> wilcoxon_signed_rank(
    const T* d,
    int64_t n,
    Alternative alternative,
    ZeroMethod zero_method
) {
    using torchscience::kernel::probability::normal_survival;
    using torchscience::kernel::probability::normal_cumulative_distribution;

    T nan = std::numeric_limits<T>::quiet_NaN();

    if (n < 1) {
        return std::make_tuple(nan, nan);
    }

    // Handle zero differences based on zero_method
    std::vector<T> abs_d;
    std::vector<int> signs;  // +1 for positive, -1 for negative

    int64_t n_zero = 0;

    for (int64_t i = 0; i < n; ++i) {
        if (d[i] == T(0)) {
            ++n_zero;
            if (zero_method == ZeroMethod::Pratt) {
                // Include zeros with sign 0 (they get rank but don't contribute to W)
                abs_d.push_back(T(0));
                signs.push_back(0);
            }
            // Wilcox: exclude zeros entirely
            // Zsplit: not implemented (split between + and -)
        } else {
            abs_d.push_back(std::abs(d[i]));
            signs.push_back(d[i] > T(0) ? 1 : -1);
        }
    }

    int64_t n_nonzero = static_cast<int64_t>(abs_d.size());

    if (n_nonzero < 1) {
        // All differences are zero
        return std::make_tuple(T(0), T(1));
    }

    // Compute ranks of absolute differences
    std::vector<T> ranks(n_nonzero);
    compute_ranks(abs_d.data(), n_nonzero, ranks.data());

    // Compute W+ (sum of ranks for positive differences)
    T W_plus = T(0);
    T W_minus = T(0);

    for (int64_t i = 0; i < n_nonzero; ++i) {
        if (signs[i] > 0) {
            W_plus += ranks[i];
        } else if (signs[i] < 0) {
            W_minus += ranks[i];
        }
        // signs[i] == 0 (Pratt zeros) don't contribute
    }

    T W_min = std::min(W_plus, W_minus);

    // Count non-zero differences for variance calculation
    int64_t n_r = 0;
    for (int64_t i = 0; i < n_nonzero; ++i) {
        if (signs[i] != 0) {
            ++n_r;
        }
    }

    if (n_r < 1) {
        return std::make_tuple(W_min, T(1));
    }

    // Expected value under null hypothesis
    T mu = T(n_r) * T(n_r + 1) / T(4);

    // Variance with tie correction (subtractive formula)
    // sigma^2 = n*(n+1)*(2*n+1)/24 - tie_term/48
    T tie_term = wilcoxon_tie_term(abs_d.data(), n_nonzero);
    T sigma_sq = T(n_r) * T(n_r + 1) * T(2 * n_r + 1) / T(24) - tie_term / T(48);

    if (sigma_sq <= T(0)) {
        return std::make_tuple(W_min, nan);
    }

    T sigma = std::sqrt(sigma_sq);

    // Z-score (with continuity correction)
    T z;
    T pvalue;

    // Standard normal: loc=0, scale=1
    T loc = T(0);
    T scale = T(1);

    if (alternative == Alternative::TWO_SIDED) {
        // For two-sided, use W_min and compute two-tailed p-value
        z = (W_min - mu) / sigma;
        pvalue = T(2) * normal_cumulative_distribution(-std::abs(z), loc, scale);
    } else if (alternative == Alternative::LESS) {
        // Test if median < 0 means W+ should be small
        z = (W_plus - mu) / sigma;
        pvalue = normal_cumulative_distribution(z, loc, scale);
    } else {  // GREATER
        // Test if median > 0 means W+ should be large
        z = (W_plus - mu) / sigma;
        pvalue = normal_survival(z, loc, scale);
    }

    // Return W_min as the statistic (matches scipy)
    return std::make_tuple(W_min, pvalue);
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
