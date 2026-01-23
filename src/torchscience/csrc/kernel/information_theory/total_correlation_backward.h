#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of total correlation w.r.t. joint distribution.
 *
 * TC = sum_i H(X_i) - H(X_1, ..., X_n)
 *    = -sum_i sum_{x_i} p(x_i) log p(x_i) + sum_x p(x) log p(x)
 *
 * The gradient dTC/dp(x) involves:
 * 1. Direct contribution from joint entropy: d/dp[sum_x p(x) log p(x)] = log p(x) + 1
 * 2. Indirect contributions through each marginal:
 *    d/dp(x)[-sum_i sum_{x_i} p(x_i) log p(x_i)] = -sum_i (log p(x_i) + 1)
 *
 * Total gradient:
 * dTC/dp(x) = (log p(x) + 1) - sum_i (log p(x_i) + 1)
 *           = log p(x) - sum_i log p(x_i) + (1 - n)
 *
 * @param grad_output Upstream gradient
 * @param joint Pointer to joint distribution (flattened)
 * @param sizes Array of dimension sizes
 * @param ndims Number of dimensions
 * @param marginals Pointer to pre-computed marginals (concatenated)
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_joint Output gradient tensor (flattened)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void total_correlation_backward_kernel(
    T grad_output,
    const T* joint,
    const int64_t* sizes,
    int64_t ndims,
    const T* marginals,
    T log_base_scale,
    T* grad_joint
) {
    T eps = get_eps<T>();

    // Compute total number of elements
    int64_t total_elements = 1;
    for (int64_t d = 0; d < ndims; ++d) {
        total_elements *= sizes[d];
    }

    // Compute strides for index calculation
    std::vector<int64_t> strides(ndims);
    strides[ndims - 1] = 1;
    for (int64_t d = ndims - 2; d >= 0; --d) {
        strides[d] = strides[d + 1] * sizes[d + 1];
    }

    // Compute marginal offsets
    std::vector<int64_t> marginal_offsets(ndims);
    marginal_offsets[0] = 0;
    for (int64_t d = 1; d < ndims; ++d) {
        marginal_offsets[d] = marginal_offsets[d - 1] + sizes[d - 1];
    }

    // Compute gradient for each element
    for (int64_t i = 0; i < total_elements; ++i) {
        T p_joint = joint[i] > eps ? joint[i] : eps;

        // Compute multi-index from flat index
        int64_t remaining = i;
        T sum_log_marginals = T(0);

        for (int64_t d = 0; d < ndims; ++d) {
            int64_t idx_d = remaining / strides[d];
            remaining = remaining % strides[d];

            T p_marginal = marginals[marginal_offsets[d] + idx_d];
            p_marginal = p_marginal > eps ? p_marginal : eps;
            sum_log_marginals += std::log(p_marginal);
        }

        // dTC/dp(x) = log p(x) - sum_i log p(x_i) + (1 - n)
        T grad = std::log(p_joint) - sum_log_marginals + T(1 - ndims);

        grad_joint[i] = grad_output * grad * log_base_scale;
    }
}

}  // namespace torchscience::kernel::information_theory
