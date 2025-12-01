#pragma once

#include <algorithm>
#include <vector>
#include <cmath>

#include <c10/macros/Macros.h>

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Compute ranks with average tie handling.
 *
 * For tied values, assigns the average of the ranks that would have been
 * assigned to each of them.
 *
 * @param data Input array (will be sorted in place conceptually via indices)
 * @param n Number of elements
 * @param ranks Output array for ranks (1-indexed)
 * @return Number of unique values (for tie correction)
 */
template <typename T>
inline int64_t compute_ranks(const T* data, int64_t n, T* ranks) {
    // Create index array and sort by data values
    std::vector<int64_t> indices(n);
    for (int64_t i = 0; i < n; ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
        return data[a] < data[b];
    });

    int64_t n_unique = 0;
    int64_t i = 0;

    while (i < n) {
        // Find all tied values
        int64_t j = i + 1;
        while (j < n && data[indices[j]] == data[indices[i]]) {
            ++j;
        }

        // Average rank for this tie group
        // Ranks are 1-indexed: if positions are i+1 to j, average = (i+1 + j) / 2
        T avg_rank = T(i + 1 + j) / T(2);

        for (int64_t k = i; k < j; ++k) {
            ranks[indices[k]] = avg_rank;
        }

        ++n_unique;
        i = j;
    }

    return n_unique;
}

/**
 * Compute tie correction factor for ranked data.
 *
 * The correction factor is: 1 - sum(t_i^3 - t_i) / (n^3 - n)
 * where t_i is the number of ties in each group.
 *
 * @param ranks Array of ranks
 * @param n Number of elements
 * @return Tie correction factor
 */
template <typename T>
inline T tie_correction(const T* ranks, int64_t n) {
    // Count tie groups by finding unique ranks
    std::vector<T> sorted_ranks(ranks, ranks + n);
    std::sort(sorted_ranks.begin(), sorted_ranks.end());

    T sum_t_cubed_minus_t = T(0);
    int64_t i = 0;

    while (i < n) {
        int64_t count = 1;
        while (i + count < n && sorted_ranks[i] == sorted_ranks[i + count]) {
            ++count;
        }
        if (count > 1) {
            T t = T(count);
            sum_t_cubed_minus_t += t * t * t - t;
        }
        i += count;
    }

    T n_cubed_minus_n = T(n) * T(n) * T(n) - T(n);
    if (n_cubed_minus_n == T(0)) {
        return T(1);
    }

    return T(1) - sum_t_cubed_minus_t / n_cubed_minus_n;
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
