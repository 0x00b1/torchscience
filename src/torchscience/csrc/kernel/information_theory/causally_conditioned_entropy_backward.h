#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of causally conditioned entropy w.r.t. joint distribution.
 *
 * H(Y||X) = -sum p(y_t, y_prev, x_t) * log[p(y_t, y_prev, x_t) / p(y_prev, x_t)]
 *
 * Let L = H(Y||X).
 *
 * The gradient dL/dp(y_t', y_prev', x_t') involves:
 * 1. Direct term from the p() factor in the sum at index (y_t', y_prev', x_t')
 * 2. Contributions through p(y_prev, x_t) marginal
 *
 * For the direct term at (y_t', y_prev', x_t'):
 *   d/dp [-p * log(p / p_cond)] = -log(p / p_cond) - 1
 *
 * For the marginal contribution:
 *   p(y_prev', x_t') depends on p(y_t, y_prev', x_t') for all y_t
 *   When we change p(y_t', y_prev', x_t'), we also change p(y_prev', x_t')
 *
 * After working out the math:
 *   dL/dp(y_t', y_prev', x_t') = -log[p(y_t', y_prev', x_t') / p(y_prev', x_t')]
 *
 * This is because:
 *   dL/dp = -[log(p/p_cond) + 1] + sum_{y_t} p(y_t, y_prev', x_t') / p(y_prev', x_t')
 *         = -log(p/p_cond) - 1 + 1
 *         = -log(p/p_cond)
 *
 * @param grad_output Upstream gradient
 * @param joint Pointer to joint distribution
 * @param p_yprev_xt Pointer to marginal p(y_prev, x_t)
 * @param size_yt, size_yprev, size_xt Dimensions
 * @param log_base_scale Scale factor
 * @param grad_joint Output gradient tensor
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void causally_conditioned_entropy_backward_kernel(
    T grad_output,
    const T* joint,
    const T* p_yprev_xt,
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
                T pyp_xt = p_yprev_xt[yp * size_xt + xt] > eps ?
                           p_yprev_xt[yp * size_xt + xt] : eps;

                // dH/dp = -log(p / p_cond) = -log(p_joint / pyp_xt)
                T log_cond = std::log(p_joint / pyp_xt);
                T grad = -log_cond;

                grad_joint[idx] = grad_output * grad * log_base_scale;
            }
        }
    }
}

}  // namespace torchscience::kernel::information_theory
