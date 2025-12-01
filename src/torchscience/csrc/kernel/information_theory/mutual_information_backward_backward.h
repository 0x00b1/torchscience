#pragma once

#include <cmath>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute second-order gradient of mutual information.
 *
 * d²I/dp(x,y)² = 1/p(x,y) - 1/p(x) - 1/p(y)
 *
 * For the backward of backward:
 * - grad_grad_output += gg_joint * (dI/dp(x,y)) * scale
 * - grad_joint += grad_output * gg_joint * (d²I/dp(x,y)²) * scale
 *
 * @param gg_joint Gradient w.r.t. grad_joint from upstream
 * @param grad_output Original upstream gradient
 * @param joint Pointer to joint distribution
 * @param p_x Pointer to marginal p(x)
 * @param p_y Pointer to marginal p(y)
 * @param size_x Size of X dimension
 * @param size_y Size of Y dimension
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_grad_output Output: gradient w.r.t. grad_output
 * @param grad_joint Output: gradient w.r.t. joint
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void mutual_information_backward_backward_kernel(
    const T* gg_joint,
    T grad_output,
    const T* joint,
    const T* p_x,
    const T* p_y,
    int64_t size_x,
    int64_t size_y,
    T log_base_scale,
    T& grad_grad_output,
    T* grad_joint
) {
    T eps = get_eps<T>();
    grad_grad_output = T(0);

    for (int64_t i = 0; i < size_x; ++i) {
        for (int64_t j = 0; j < size_y; ++j) {
            int64_t idx = i * size_y + j;
            T p_xy = joint[idx] > eps ? joint[idx] : eps;
            T marginal_prod = p_x[i] * p_y[j];

            T first_deriv;
            T second_deriv;

            if (marginal_prod > eps && p_xy > eps) {
                // First derivative: log(p(x,y) / (p(x) * p(y))) - 1
                first_deriv = std::log(p_xy / marginal_prod) - T(1);

                // Second derivative: 1/p(x,y) - 1/p(x) - 1/p(y)
                second_deriv = T(1) / p_xy - T(1) / p_x[i] - T(1) / p_y[j];
            } else {
                first_deriv = T(0);
                second_deriv = T(0);
            }

            T gg = gg_joint ? gg_joint[idx] : T(0);

            // Gradient w.r.t. grad_output
            grad_grad_output += gg * first_deriv * log_base_scale;

            // Gradient w.r.t. joint
            if (grad_joint) {
                grad_joint[idx] = grad_output * gg * second_deriv * log_base_scale;
            }
        }
    }
}

}  // namespace torchscience::kernel::information_theory
