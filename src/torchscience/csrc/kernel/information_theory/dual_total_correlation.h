#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute dual total correlation (binding information) from an N-dimensional joint distribution.
 *
 * DTC(X_1, ..., X_n) = H(X_1, ..., X_n) - sum_i H(X_i | X_{-i})
 *
 * Where H(X_i | X_{-i}) is the entropy of X_i conditioned on all other variables.
 *
 * Using the identity:
 * H(X_i | X_{-i}) = H(X_1, ..., X_n) - H(X_{-i})
 *
 * Where H(X_{-i}) is the entropy of the marginal over all variables except X_i.
 *
 * Therefore:
 * sum_i H(X_i | X_{-i}) = n * H(joint) - sum_i H(X_{-i})
 *
 * And:
 * DTC = H(joint) - [n * H(joint) - sum_i H(X_{-i})]
 *     = (1 - n) * H(joint) + sum_i H(X_{-i})
 *
 * @param joint Pointer to joint distribution (flattened)
 * @param sizes Array of dimension sizes [d_1, ..., d_n]
 * @param ndims Number of dimensions
 * @param complementary_marginals Pointer to pre-computed complementary marginals
 *        (marginals over all-but-one dimension, stored sequentially)
 * @param complementary_sizes Array of sizes for each complementary marginal
 * @param log_base_scale Scale factor for log base conversion
 * @return Dual total correlation value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T dual_total_correlation_kernel(
    const T* joint,
    const int64_t* sizes,
    int64_t ndims,
    const T* complementary_marginals,
    const int64_t* complementary_sizes,
    T log_base_scale
) {
    T eps = get_eps<T>();

    // Compute total number of elements in joint
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

    // Compute sum of complementary marginal entropies
    // H(X_{-i}) for each i, where X_{-i} is all variables except X_i
    T sum_complementary_entropy = T(0);
    int64_t comp_offset = 0;
    for (int64_t d = 0; d < ndims; ++d) {
        int64_t comp_size = complementary_sizes[d];
        for (int64_t i = 0; i < comp_size; ++i) {
            T p = complementary_marginals[comp_offset + i];
            if (p > eps) {
                sum_complementary_entropy -= p * std::log(p);
            }
        }
        comp_offset += comp_size;
    }

    // DTC = (1 - n) * H(joint) + sum_i H(X_{-i})
    T dtc = T(1 - ndims) * joint_entropy + sum_complementary_entropy;

    return dtc * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
