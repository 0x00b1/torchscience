// src/torchscience/csrc/kernel/statistics/hypothesis_test/kruskal_wallis.h
#pragma once

#include <cmath>
#include <vector>
#include <limits>
#include <tuple>

#include <c10/macros/Macros.h>

#include "ranking.h"
#include "../../probability/chi2_survival.h"

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Kruskal-Wallis H test.
 *
 * Non-parametric test for comparing distributions of k independent samples.
 *
 * @param data Concatenated data from all groups
 * @param n Total number of observations
 * @param group_sizes Array of group sizes (length k)
 * @param k Number of groups
 * @return Tuple of (H-statistic, p-value)
 */
template <typename T>
std::tuple<T, T> kruskal_wallis(
    const T* data,
    int64_t n,
    const int64_t* group_sizes,
    int64_t k
) {
    using torchscience::kernel::probability::chi2_survival;

    T nan = std::numeric_limits<T>::quiet_NaN();

    if (n < 2 || k < 2) {
        return std::make_tuple(nan, nan);
    }

    // Verify total matches sum of group sizes
    int64_t total = 0;
    for (int64_t i = 0; i < k; ++i) {
        total += group_sizes[i];
    }
    if (total != n) {
        return std::make_tuple(nan, nan);
    }

    // Compute ranks of all data
    std::vector<T> ranks(n);
    compute_ranks(data, n, ranks.data());

    // Compute sum of ranks for each group
    std::vector<T> rank_sums(k, T(0));
    int64_t offset = 0;
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < group_sizes[i]; ++j) {
            rank_sums[i] += ranks[offset + j];
        }
        offset += group_sizes[i];
    }

    // Compute H statistic
    // H = (12 / (n*(n+1))) * sum(R_i^2 / n_i) - 3*(n+1)
    T sum_term = T(0);
    for (int64_t i = 0; i < k; ++i) {
        if (group_sizes[i] > 0) {
            sum_term += (rank_sums[i] * rank_sums[i]) / T(group_sizes[i]);
        }
    }

    T H = (T(12) / (T(n) * T(n + 1))) * sum_term - T(3) * T(n + 1);

    // Apply tie correction
    // H_corrected = H / (1 - sum(t^3 - t) / (n^3 - n))
    T tie_corr = tie_correction(ranks.data(), n);
    if (tie_corr > T(0)) {
        H = H / tie_corr;
    }

    // P-value from chi-squared distribution with k-1 degrees of freedom
    T df = T(k - 1);
    T pvalue = chi2_survival(H, df);

    return std::make_tuple(H, pvalue);
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
