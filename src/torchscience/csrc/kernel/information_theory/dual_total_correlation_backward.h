#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of dual total correlation w.r.t. joint distribution.
 *
 * DTC = (1 - n) * H(joint) + sum_i H(X_{-i})
 *     = -(1 - n) * sum_x p(x) log p(x) - sum_i sum_{x_{-i}} p(x_{-i}) log p(x_{-i})
 *
 * The gradient dDTC/dp(x) involves:
 * 1. Direct contribution from joint entropy:
 *    d/dp(x)[-(1-n) * sum_x p(x) log p(x)] = -(1-n) * (log p(x) + 1)
 * 2. Indirect contributions through each complementary marginal p(x_{-i}):
 *    For each dimension i, p(x) contributes to p(x_{-i}) where x_{-i} is the
 *    projection of x onto all dimensions except i.
 *    d/dp(x)[-sum_{x_{-i}} p(x_{-i}) log p(x_{-i})] = -(log p(x_{-i}) + 1)
 *
 * Total gradient:
 * dDTC/dp(x) = -(1-n) * (log p(x) + 1) - sum_i (log p(x_{-i}) + 1)
 *            = (n-1) * (log p(x) + 1) - sum_i (log p(x_{-i}) + 1)
 *            = (n-1) * log p(x) - sum_i log p(x_{-i}) + (n-1) - n
 *            = (n-1) * log p(x) - sum_i log p(x_{-i}) - 1
 *
 * @param grad_output Upstream gradient
 * @param joint Pointer to joint distribution (flattened)
 * @param sizes Array of dimension sizes
 * @param ndims Number of dimensions
 * @param complementary_marginals Pointer to pre-computed complementary marginals
 * @param complementary_sizes Array of sizes for each complementary marginal
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_joint Output gradient tensor (flattened)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void dual_total_correlation_backward_kernel(
    T grad_output,
    const T* joint,
    const int64_t* sizes,
    int64_t ndims,
    const T* complementary_marginals,
    const int64_t* complementary_sizes,
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

    // Compute complementary strides for each dimension
    // complementary_strides[d][k] is the stride for dimension k in the complementary
    // marginal that excludes dimension d
    std::vector<std::vector<int64_t>> complementary_strides(ndims);
    for (int64_t d = 0; d < ndims; ++d) {
        // Complementary marginal excludes dimension d
        // Build strides for the remaining (ndims-1) dimensions
        std::vector<int64_t>& comp_strides = complementary_strides[d];
        comp_strides.resize(ndims);

        int64_t stride = 1;
        for (int64_t k = ndims - 1; k >= 0; --k) {
            if (k == d) {
                comp_strides[k] = 0;  // This dimension is summed out
            } else {
                comp_strides[k] = stride;
                stride *= sizes[k];
            }
        }
    }

    // Compute offsets for complementary marginals
    std::vector<int64_t> comp_offsets(ndims);
    comp_offsets[0] = 0;
    for (int64_t d = 1; d < ndims; ++d) {
        comp_offsets[d] = comp_offsets[d - 1] + complementary_sizes[d - 1];
    }

    // Compute gradient for each element
    for (int64_t i = 0; i < total_elements; ++i) {
        T p_joint = joint[i] > eps ? joint[i] : eps;

        // Compute multi-index from flat index
        std::vector<int64_t> indices(ndims);
        int64_t remaining = i;
        for (int64_t d = 0; d < ndims; ++d) {
            indices[d] = remaining / strides[d];
            remaining = remaining % strides[d];
        }

        // Sum log of complementary marginals
        T sum_log_complementary = T(0);
        for (int64_t d = 0; d < ndims; ++d) {
            // Compute index into complementary marginal d
            int64_t comp_idx = 0;
            for (int64_t k = 0; k < ndims; ++k) {
                comp_idx += indices[k] * complementary_strides[d][k];
            }

            T p_comp = complementary_marginals[comp_offsets[d] + comp_idx];
            p_comp = p_comp > eps ? p_comp : eps;
            sum_log_complementary += std::log(p_comp);
        }

        // dDTC/dp(x) = (n-1) * log p(x) - sum_i log p(x_{-i}) - 1
        T grad = T(ndims - 1) * std::log(p_joint) - sum_log_complementary - T(1);

        grad_joint[i] = grad_output * grad * log_base_scale;
    }
}

}  // namespace torchscience::kernel::information_theory
