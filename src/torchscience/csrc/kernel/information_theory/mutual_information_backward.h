#pragma once

#include <cmath>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of mutual information w.r.t. joint distribution.
 *
 * dI/dp(x,y) = log(p(x,y) / (p(x) * p(y))) - 1
 *
 * Derivation:
 * I(X;Y) = sum p(x,y) * [log p(x,y) - log p(x) - log p(y)]
 *
 * Taking derivative w.r.t. p(x',y'):
 * - Direct term: log p(x',y') + 1 - log p(x') - log p(y')
 * - Via p(x'): sum_y p(x',y) * (-1/p(x')) = -1
 * - Via p(y'): sum_x p(x,y') * (-1/p(y')) = -1
 *
 * Total: log(p(x,y)/(p(x)*p(y))) + 1 - 1 - 1 = log(p(x,y)/(p(x)*p(y))) - 1
 *
 * @param grad_output Upstream gradient
 * @param joint Pointer to joint distribution
 * @param p_x Pointer to marginal p(x)
 * @param p_y Pointer to marginal p(y)
 * @param size_x Size of X dimension
 * @param size_y Size of Y dimension
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_joint Output gradient tensor
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void mutual_information_backward_kernel(
    T grad_output,
    const T* joint,
    const T* p_x,
    const T* p_y,
    int64_t size_x,
    int64_t size_y,
    T log_base_scale,
    T* grad_joint
) {
    T eps = get_eps<T>();

    for (int64_t i = 0; i < size_x; ++i) {
        for (int64_t j = 0; j < size_y; ++j) {
            int64_t idx = i * size_y + j;
            T p_xy = joint[idx] > eps ? joint[idx] : eps;
            T marginal_prod = p_x[i] * p_y[j];

            T grad;
            if (marginal_prod > eps && p_xy > eps) {
                // dI/dp(x,y) = log(p(x,y) / (p(x) * p(y))) - 1
                grad = std::log(p_xy / marginal_prod) - T(1);
            } else {
                grad = T(0);
            }

            grad_joint[idx] = grad_output * grad * log_base_scale;
        }
    }
}

}  // namespace torchscience::kernel::information_theory
