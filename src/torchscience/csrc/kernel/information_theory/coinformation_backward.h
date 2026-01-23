#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of coinformation w.r.t. joint distribution.
 *
 * CI = sum_{S non-empty} (-1)^{|S|+1} H(X_S)
 *    = sum_{S} sign_S * [-sum_{x_S} p(x_S) log p(x_S)]
 *
 * where sign_S = (-1)^{|S|+1} = +1 for |S| odd, -1 for |S| even.
 *
 * For each element p(x) in the joint distribution, it contributes to the
 * marginal p(x_S) for every subset S. The gradient is:
 *
 * dCI/dp(x) = sum_{S} sign_S * d H(X_S) / dp(x)
 *
 * For each subset S, the contribution is:
 * d H(X_S) / dp(x) = -(log p(x_S) + 1)
 *
 * where x_S is the projection of x onto the dimensions in S.
 *
 * Total gradient:
 * dCI/dp(x) = sum_{S non-empty} (-1)^{|S|+1} * [-(log p(x_S) + 1)]
 *           = -sum_{S non-empty} (-1)^{|S|+1} * (log p(x_S) + 1)
 *
 * @param grad_output Upstream gradient
 * @param joint Pointer to joint distribution (flattened)
 * @param sizes Array of dimension sizes
 * @param ndims Number of dimensions
 * @param num_subsets Number of non-empty subsets (2^n - 1)
 * @param subset_masks Bitmask for each subset
 * @param subset_marginals Pre-computed marginal distributions for each subset (flattened)
 * @param subset_offsets Offset into subset_marginals for each subset
 * @param subset_sizes Size of marginal for each subset
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_joint Output gradient tensor (flattened)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void coinformation_backward_kernel(
    T grad_output,
    const T* joint,
    const int64_t* sizes,
    int64_t ndims,
    int64_t num_subsets,
    const int64_t* subset_masks,
    const T* subset_marginals,
    const int64_t* subset_offsets,
    const int64_t* subset_sizes,
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

    // For each subset, compute the strides for projecting onto that marginal
    // This maps a full multi-index to the marginal index
    std::vector<std::vector<int64_t>> subset_strides(num_subsets);
    for (int64_t s = 0; s < num_subsets; ++s) {
        subset_strides[s].resize(ndims, 0);
        int64_t mask = subset_masks[s];
        int64_t stride = 1;

        // Build strides from right to left for dimensions in the mask
        for (int64_t d = ndims - 1; d >= 0; --d) {
            if (mask & (1LL << d)) {
                subset_strides[s][d] = stride;
                stride *= sizes[d];
            }
        }
    }

    // Compute signs for each subset
    // Sign is (-1)^(k+1) = +1 for odd k, -1 for even k
    std::vector<T> subset_signs(num_subsets);
    for (int64_t s = 0; s < num_subsets; ++s) {
        int64_t mask = subset_masks[s];
        int64_t k = 0;
        int64_t temp = mask;
        while (temp > 0) {
            k += temp & 1;
            temp >>= 1;
        }
        subset_signs[s] = (k % 2 == 1) ? T(1) : T(-1);
    }

    // Compute gradient for each element
    for (int64_t i = 0; i < total_elements; ++i) {
        // Compute multi-index from flat index
        std::vector<int64_t> indices(ndims);
        int64_t remaining = i;
        for (int64_t d = 0; d < ndims; ++d) {
            indices[d] = remaining / strides[d];
            remaining = remaining % strides[d];
        }

        // Sum contributions from all subsets
        T grad = T(0);
        for (int64_t s = 0; s < num_subsets; ++s) {
            // Compute index into this marginal
            int64_t marginal_idx = 0;
            for (int64_t d = 0; d < ndims; ++d) {
                marginal_idx += indices[d] * subset_strides[s][d];
            }

            T p_marginal = subset_marginals[subset_offsets[s] + marginal_idx];
            p_marginal = p_marginal > eps ? p_marginal : eps;

            // Contribution: -sign * (log p(x_S) + 1)
            grad -= subset_signs[s] * (std::log(p_marginal) + T(1));
        }

        grad_joint[i] = grad_output * grad * log_base_scale;
    }
}

}  // namespace torchscience::kernel::information_theory
