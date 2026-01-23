#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute total correlation (multi-information) from an N-dimensional joint distribution.
 *
 * TC(X_1, ..., X_n) = sum_i H(X_i) - H(X_1, ..., X_n)
 *                   = sum_i [-sum_{x_i} p(x_i) log p(x_i)] - [-sum_{x} p(x) log p(x)]
 *
 * For a joint distribution represented as a flat array with dimensions [d_1, ..., d_n],
 * we compute:
 * 1. Joint entropy H(X_1, ..., X_n) = -sum p(x) log p(x)
 * 2. Marginal entropies H(X_i) for each dimension
 * 3. TC = sum_i H(X_i) - H(joint)
 *
 * @param joint Pointer to joint distribution (flattened)
 * @param sizes Array of dimension sizes [d_1, ..., d_n]
 * @param ndims Number of dimensions
 * @param marginals Pointer to pre-computed marginals (concatenated: p(x_1), p(x_2), ...)
 * @param log_base_scale Scale factor for log base conversion
 * @return Total correlation value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T total_correlation_kernel(
    const T* joint,
    const int64_t* sizes,
    int64_t ndims,
    const T* marginals,
    T log_base_scale
) {
    T eps = get_eps<T>();

    // Compute total number of elements
    int64_t total_elements = 1;
    for (int64_t d = 0; d < ndims; ++d) {
        total_elements *= sizes[d];
    }

    // Compute joint entropy H(X_1, ..., X_n) = -sum p(x) log p(x)
    T joint_entropy = T(0);
    for (int64_t i = 0; i < total_elements; ++i) {
        T p = joint[i];
        if (p > eps) {
            joint_entropy -= p * std::log(p);
        }
    }

    // Compute sum of marginal entropies
    // Marginals are stored concatenated: first sizes[0] elements are p(x_0),
    // next sizes[1] elements are p(x_1), etc.
    T sum_marginal_entropy = T(0);
    int64_t marginal_offset = 0;
    for (int64_t d = 0; d < ndims; ++d) {
        for (int64_t i = 0; i < sizes[d]; ++i) {
            T p = marginals[marginal_offset + i];
            if (p > eps) {
                sum_marginal_entropy -= p * std::log(p);
            }
        }
        marginal_offset += sizes[d];
    }

    // TC = sum_i H(X_i) - H(joint)
    T tc = sum_marginal_entropy - joint_entropy;

    return tc * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
