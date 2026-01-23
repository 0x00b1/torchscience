#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute coinformation (N-variable generalization of interaction information)
 * from an N-dimensional joint distribution using inclusion-exclusion.
 *
 * CI(X_1, ..., X_n) = -sum_{S subset {1,...,n}, S non-empty} (-1)^|S| H(X_S)
 *                   = sum_{S non-empty} (-1)^{|S|+1} H(X_S)
 *
 * Or equivalently using alternating signs by subset size:
 * CI = sum_{k=1}^{n} (-1)^{k+1} * [sum of all k-marginal entropies]
 *
 * For n=2: CI(X;Y) = H(X) + H(Y) - H(X,Y) = I(X;Y)
 * For n=3: CI(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z) = I(X;Y;Z)
 *
 * The sign for a subset S with |S| = k is (-1)^(k+1):
 * - k=1 (single variables): sign = +1
 * - k=2 (pairs): sign = -1
 * - k=3 (triples): sign = +1
 * - etc.
 *
 * @param joint Pointer to joint distribution (flattened)
 * @param sizes Array of dimension sizes [d_1, ..., d_n]
 * @param ndims Number of dimensions
 * @param num_subsets Number of non-empty subsets (2^n - 1)
 * @param subset_masks Bitmask for each subset
 * @param subset_entropies Pre-computed entropy for each subset
 * @param log_base_scale Scale factor for log base conversion
 * @return Coinformation value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T coinformation_kernel(
    const T* joint,
    const int64_t* sizes,
    int64_t ndims,
    int64_t num_subsets,
    const int64_t* subset_masks,
    const T* subset_entropies,
    T log_base_scale
) {
    // Apply inclusion-exclusion formula
    // For subset S with |S| = k, the sign is (-1)^(k+1)
    T ci = T(0);

    for (int64_t s = 0; s < num_subsets; ++s) {
        int64_t mask = subset_masks[s];

        // Count bits to get subset size k
        int64_t k = 0;
        int64_t temp = mask;
        while (temp > 0) {
            k += temp & 1;
            temp >>= 1;
        }

        // Sign is (-1)^(k+1) = +1 for odd k, -1 for even k
        T sign = (k % 2 == 1) ? T(1) : T(-1);

        ci += sign * subset_entropies[s];
    }

    return ci * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
