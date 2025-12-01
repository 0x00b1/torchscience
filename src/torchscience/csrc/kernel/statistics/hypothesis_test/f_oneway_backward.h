#pragma once

#include <cmath>

#include <c10/macros/Macros.h>

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Backward pass for f_oneway F-statistic.
 *
 * Computes gradient of F w.r.t. each data point.
 *
 * The F-statistic is: F = MS_between / MS_within
 *
 * Gradient derivation:
 *   dF/dx_ij = (1/MS_within) * (dMS_between/dx_ij - F * dMS_within/dx_ij)
 *
 * where for x_ij in group g:
 *   dMS_between/dx_ij = 2 * (mean_g - grand_mean) / (k-1)
 *   dMS_within/dx_ij = 2 * (x_ij - mean_g) / (N-k)
 *
 * @param grad_statistic Gradient w.r.t. F-statistic (scalar)
 * @param data Pointer to data array (groups concatenated)
 * @param group_sizes Array of group sizes
 * @param k Number of groups
 * @param grad_data Output gradient array
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void f_oneway_backward(
    T grad_statistic,
    const T* data,
    const int64_t* group_sizes,
    int64_t k,
    T* grad_data
) {
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
        // Fill with zeros for invalid case
        for (int64_t i = 0; i < N; ++i) {
            grad_data[i] = T(0);
        }
        return;
    }

    T grand_mean = grand_sum / T(N);

    // Calculate group means and statistics
    T ss_between = T(0);
    T ss_within = T(0);

    // First pass: compute group means and ss values
    std::vector<T> group_means(k);
    offset = 0;
    for (int64_t g = 0; g < k; ++g) {
        int64_t n_g = group_sizes[g];

        T group_sum = T(0);
        for (int64_t i = 0; i < n_g; ++i) {
            group_sum += data[offset + i];
        }
        T group_mean = group_sum / T(n_g);
        group_means[g] = group_mean;

        T diff = group_mean - grand_mean;
        ss_between += T(n_g) * diff * diff;

        for (int64_t i = 0; i < n_g; ++i) {
            T d = data[offset + i] - group_mean;
            ss_within += d * d;
        }

        offset += n_g;
    }

    T df_between = T(k - 1);
    T df_within = T(N - k);

    if (df_within <= T(0) || ss_within <= T(0)) {
        for (int64_t i = 0; i < N; ++i) {
            grad_data[i] = T(0);
        }
        return;
    }

    T ms_between = ss_between / df_between;
    T ms_within = ss_within / df_within;
    T F = ms_between / ms_within;

    // Second pass: compute gradients
    offset = 0;
    for (int64_t g = 0; g < k; ++g) {
        int64_t n_g = group_sizes[g];
        T group_mean = group_means[g];

        // dMS_between/dx_ij = 2 * (mean_g - grand_mean) / (k-1)
        T dMS_between = T(2) * (group_mean - grand_mean) / df_between;

        for (int64_t i = 0; i < n_g; ++i) {
            T x_ij = data[offset + i];

            // dMS_within/dx_ij = 2 * (x_ij - mean_g) / (N-k)
            T dMS_within = T(2) * (x_ij - group_mean) / df_within;

            // dF/dx_ij = (1/ms_within) * (dMS_between - F * dMS_within)
            T dF = (dMS_between - F * dMS_within) / ms_within;

            grad_data[offset + i] = grad_statistic * dF;
        }

        offset += n_g;
    }
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
