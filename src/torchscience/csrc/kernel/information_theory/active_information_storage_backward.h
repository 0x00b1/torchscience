#pragma once

#include <cmath>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of active information storage w.r.t. joint distribution.
 *
 * A(X) = sum p(x_t, x_{t-1}) * [log p(x_t, x_{t-1}) - log p(x_t) - log p(x_{t-1})]
 *
 * Taking derivative w.r.t. p(x_t', x_{t-1}'):
 * - Direct term: log p(x_t', x_{t-1}') + 1 - log p(x_t') - log p(x_{t-1}')
 * - Via p(x_t'): sum_{x_{t-1}} p(x_t', x_{t-1}) * (-1/p(x_t')) = -1
 * - Via p(x_{t-1}'): sum_{x_t} p(x_t, x_{t-1}') * (-1/p(x_{t-1}')) = -1
 *
 * Total: log(p(x_t, x_{t-1})/(p(x_t)*p(x_{t-1}))) + 1 - 1 - 1
 *      = log(p(x_t, x_{t-1})/(p(x_t)*p(x_{t-1}))) - 1
 *
 * @param grad_output Upstream gradient
 * @param joint Pointer to joint distribution
 * @param p_curr Pointer to marginal p(x_t)
 * @param p_prev Pointer to marginal p(x_{t-1})
 * @param size_curr Size of current state dimension
 * @param size_prev Size of previous state dimension
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_joint Output gradient tensor
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void active_information_storage_backward_kernel(
    T grad_output,
    const T* joint,
    const T* p_curr,
    const T* p_prev,
    int64_t size_curr,
    int64_t size_prev,
    T log_base_scale,
    T* grad_joint
) {
    T eps = get_eps<T>();

    for (int64_t i = 0; i < size_curr; ++i) {
        for (int64_t j = 0; j < size_prev; ++j) {
            int64_t idx = i * size_prev + j;
            T p_xy = joint[idx] > eps ? joint[idx] : eps;
            T marginal_prod = p_curr[i] * p_prev[j];

            T grad;
            if (marginal_prod > eps && p_xy > eps) {
                // dA/dp(x_t, x_{t-1}) = log(p(x_t, x_{t-1}) / (p(x_t) * p(x_{t-1}))) - 1
                grad = std::log(p_xy / marginal_prod) - T(1);
            } else {
                grad = T(0);
            }

            grad_joint[idx] = grad_output * grad * log_base_scale;
        }
    }
}

}  // namespace torchscience::kernel::information_theory
