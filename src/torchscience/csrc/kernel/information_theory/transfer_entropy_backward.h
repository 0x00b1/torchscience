#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of transfer entropy w.r.t. joint distribution.
 *
 * T(X -> Y) = sum p(y_t, y_prev, x_prev) * log[p(y_t, y_prev, x_prev) * p(y_prev) / (p(y_prev, x_prev) * p(y_t, y_prev))]
 *
 * The gradient dT/dp(y_t', y_prev', x_prev') involves:
 * 1. Direct term from the p() factor in the sum
 * 2. Contributions through the marginals in the log argument
 *
 * After working out the math (similar to conditional mutual information),
 * the gradient simplifies to:
 *
 * dT/dp(y_t', y_prev', x_prev') = log[p(y_t', y_prev', x_prev') * p(y_prev') / (p(y_prev', x_prev') * p(y_t', y_prev'))]
 *
 * @param grad_output Upstream gradient
 * @param joint Pointer to joint distribution
 * @param p_yprev_xprev Pointer to marginal p(y_prev, x_prev)
 * @param p_yt_yprev Pointer to marginal p(y_t, y_prev)
 * @param p_yprev Pointer to marginal p(y_prev)
 * @param size_yt, size_yprev, size_xprev Dimensions
 * @param log_base_scale Scale factor
 * @param grad_joint Output gradient tensor
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void transfer_entropy_backward_kernel(
    T grad_output,
    const T* joint,
    const T* p_yprev_xprev,
    const T* p_yt_yprev,
    const T* p_yprev,
    int64_t size_yt,
    int64_t size_yprev,
    int64_t size_xprev,
    T log_base_scale,
    T* grad_joint
) {
    T eps = get_eps<T>();

    // Compute gradients
    for (int64_t yt = 0; yt < size_yt; ++yt) {
        for (int64_t yp = 0; yp < size_yprev; ++yp) {
            for (int64_t xp = 0; xp < size_xprev; ++xp) {
                int64_t idx = (yt * size_yprev + yp) * size_xprev + xp;
                T p_joint = joint[idx] > eps ? joint[idx] : eps;
                T pyp = p_yprev[yp] > eps ? p_yprev[yp] : eps;
                T pyp_xp = p_yprev_xprev[yp * size_xprev + xp] > eps ?
                           p_yprev_xprev[yp * size_xprev + xp] : eps;
                T pyt_yp = p_yt_yprev[yt * size_yprev + yp] > eps ?
                           p_yt_yprev[yt * size_yprev + yp] : eps;

                // dT/dp(y_t, y_prev, x_prev) = log[p(y_t, y_prev, x_prev) * p(y_prev) / (p(y_prev, x_prev) * p(y_t, y_prev))]
                T log_ratio = std::log((p_joint * pyp) / (pyp_xp * pyt_yp));
                T grad = log_ratio;

                grad_joint[idx] = grad_output * grad * log_base_scale;
            }
        }
    }
}

}  // namespace torchscience::kernel::information_theory
