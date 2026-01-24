#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of directed information w.r.t. joint distribution.
 *
 * I(X -> Y) = sum p(y_t, y_prev, x_t) * log[p(y_t, y_prev, x_t) * p(y_prev) / (p(y_t, y_prev) * p(y_prev, x_t))]
 *
 * The gradient dI/dp(y_t', y_prev', x_t') involves:
 * 1. Direct term from the p() factor in the sum
 * 2. Contributions through the marginals in the log argument
 *
 * After working out the math (similar to conditional mutual information),
 * the gradient simplifies to:
 *
 * dI/dp(y_t', y_prev', x_t') = log[p(y_t', y_prev', x_t') * p(y_prev') / (p(y_t', y_prev') * p(y_prev', x_t'))]
 *
 * @param grad_output Upstream gradient
 * @param joint Pointer to joint distribution
 * @param p_yprev_xt Pointer to marginal p(y_prev, x_t)
 * @param p_yt_yprev Pointer to marginal p(y_t, y_prev)
 * @param p_yprev Pointer to marginal p(y_prev)
 * @param size_yt, size_yprev, size_xt Dimensions
 * @param log_base_scale Scale factor
 * @param grad_joint Output gradient tensor
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void directed_information_backward_kernel(
    T grad_output,
    const T* joint,
    const T* p_yprev_xt,
    const T* p_yt_yprev,
    const T* p_yprev,
    int64_t size_yt,
    int64_t size_yprev,
    int64_t size_xt,
    T log_base_scale,
    T* grad_joint
) {
    T eps = get_eps<T>();

    // Compute gradients
    for (int64_t yt = 0; yt < size_yt; ++yt) {
        for (int64_t yp = 0; yp < size_yprev; ++yp) {
            for (int64_t xt = 0; xt < size_xt; ++xt) {
                int64_t idx = (yt * size_yprev + yp) * size_xt + xt;
                T p_joint = joint[idx] > eps ? joint[idx] : eps;
                T pyp = p_yprev[yp] > eps ? p_yprev[yp] : eps;
                T pyp_xt = p_yprev_xt[yp * size_xt + xt] > eps ?
                           p_yprev_xt[yp * size_xt + xt] : eps;
                T pyt_yp = p_yt_yprev[yt * size_yprev + yp] > eps ?
                           p_yt_yprev[yt * size_yprev + yp] : eps;

                // dI/dp(y_t, y_prev, x_t) = log[p(y_t, y_prev, x_t) * p(y_prev) / (p(y_t, y_prev) * p(y_prev, x_t))]
                T log_ratio = std::log((p_joint * pyp) / (pyt_yp * pyp_xt));
                T grad = log_ratio;

                grad_joint[idx] = grad_output * grad * log_base_scale;
            }
        }
    }
}

}  // namespace torchscience::kernel::information_theory
