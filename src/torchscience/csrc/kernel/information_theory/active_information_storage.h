#pragma once

#include <cmath>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute active information storage A(X) = I(X_{t-1}; X_t) from joint distribution.
 *
 * A(X) = sum_{x_t, x_{t-1}} p(x_t, x_{t-1}) * log(p(x_t, x_{t-1}) / (p(x_t) * p(x_{t-1})))
 *      = H(X_t) + H(X_{t-1}) - H(X_t, X_{t-1})
 *
 * This is the mutual information between consecutive time steps.
 *
 * @param joint Pointer to joint distribution p(x_t, x_{t-1}) of shape [size_curr, size_prev]
 * @param p_curr Pointer to marginal p(x_t) of shape [size_curr]
 * @param p_prev Pointer to marginal p(x_{t-1}) of shape [size_prev]
 * @param size_curr Size of current state dimension
 * @param size_prev Size of previous state dimension
 * @param log_base_scale Scale factor for log base conversion (1/log(base))
 * @return Active information storage value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T active_information_storage_kernel(
    const T* joint,
    const T* p_curr,
    const T* p_prev,
    int64_t size_curr,
    int64_t size_prev,
    T log_base_scale
) {
    T eps = get_eps<T>();
    T result = T(0);

    // Compute A(X) = sum p(x_t, x_{t-1}) log(p(x_t, x_{t-1}) / (p(x_t) * p(x_{t-1})))
    for (int64_t i = 0; i < size_curr; ++i) {
        for (int64_t j = 0; j < size_prev; ++j) {
            T p_xy = joint[i * size_prev + j];
            if (p_xy > eps) {
                T marginal_product = p_curr[i] * p_prev[j];
                if (marginal_product > eps) {
                    result += p_xy * std::log(p_xy / marginal_product);
                }
            }
        }
    }

    return result * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
