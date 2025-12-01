#pragma once

#include <cmath>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of pointwise mutual information w.r.t. joint distribution.
 *
 * For PMI(x,y) = log(p(x,y) / (p(x) * p(y))):
 *
 * d Loss / d p(x,y) = grad_output[x,y]/p(x,y) - g_x/p(x) - g_y/p(y)
 *
 * where:
 * - g_x = sum_{y'} grad_output[x,y']
 * - g_y = sum_{x'} grad_output[x',y]
 *
 * @param grad_output Upstream gradient tensor [size_x, size_y]
 * @param joint Joint distribution [size_x, size_y]
 * @param p_x Marginal p(x) [size_x]
 * @param p_y Marginal p(y) [size_y]
 * @param g_x Sum of grad_output over y for each x [size_x]
 * @param g_y Sum of grad_output over x for each y [size_y]
 * @param size_x Size of X dimension
 * @param size_y Size of Y dimension
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_joint Output gradient tensor [size_x, size_y]
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void pointwise_mutual_information_backward_kernel(
    const T* grad_output,
    const T* joint,
    const T* p_x,
    const T* p_y,
    const T* g_x,
    const T* g_y,
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
            T px = p_x[i] > eps ? p_x[i] : eps;
            T py = p_y[j] > eps ? p_y[j] : eps;

            // grad_joint[x,y] = grad_output[x,y]/p(x,y) - g_x/p(x) - g_y/p(y)
            T grad = grad_output[idx] / p_xy - g_x[i] / px - g_y[j] / py;

            grad_joint[idx] = grad * log_base_scale;
        }
    }
}

}  // namespace torchscience::kernel::information_theory
