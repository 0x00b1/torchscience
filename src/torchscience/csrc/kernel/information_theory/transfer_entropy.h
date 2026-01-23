#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute transfer entropy T(X -> Y) from a 3D joint distribution.
 *
 * T(X -> Y) = I(Y_t; X_{t-1} | Y_{t-1})
 *           = sum_{y_t, y_prev, x_prev} p(y_t, y_prev, x_prev) *
 *             log[p(y_t | y_prev, x_prev) / p(y_t | y_prev)]
 *
 * Using marginals:
 *   p(y_t | y_prev, x_prev) = p(y_t, y_prev, x_prev) / p(y_prev, x_prev)
 *   p(y_t | y_prev) = p(y_t, y_prev) / p(y_prev)
 *
 * So:
 *   T(X -> Y) = sum p(y_t, y_prev, x_prev) *
 *               [log p(y_t, y_prev, x_prev) - log p(y_prev, x_prev)
 *                - log p(y_t, y_prev) + log p(y_prev)]
 *
 * The input joint tensor has shape [size_yt, size_yprev, size_xprev]
 * representing p(y_t, y_{t-1}, x_{t-1}).
 *
 * @param joint Pointer to joint distribution p(y_t, y_prev, x_prev)
 * @param p_yprev_xprev Pointer to marginal p(y_prev, x_prev) shape [size_yprev, size_xprev]
 * @param p_yt_yprev Pointer to marginal p(y_t, y_prev) shape [size_yt, size_yprev]
 * @param p_yprev Pointer to marginal p(y_prev) shape [size_yprev]
 * @param size_yt Size of Y_t dimension
 * @param size_yprev Size of Y_{t-1} dimension
 * @param size_xprev Size of X_{t-1} dimension
 * @param log_base_scale Scale factor for log base conversion
 * @return Transfer entropy value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T transfer_entropy_kernel(
    const T* joint,
    const T* p_yprev_xprev,
    const T* p_yt_yprev,
    const T* p_yprev,
    int64_t size_yt,
    int64_t size_yprev,
    int64_t size_xprev,
    T log_base_scale
) {
    T eps = get_eps<T>();
    T result = T(0);

    // Compute T(X -> Y) = sum p(y_t, y_prev, x_prev) *
    //                     log[p(y_t, y_prev, x_prev) * p(y_prev) / (p(y_prev, x_prev) * p(y_t, y_prev))]
    for (int64_t yt = 0; yt < size_yt; ++yt) {
        for (int64_t yp = 0; yp < size_yprev; ++yp) {
            for (int64_t xp = 0; xp < size_xprev; ++xp) {
                T p_joint = joint[(yt * size_yprev + yp) * size_xprev + xp];
                if (p_joint > eps) {
                    T pyp = p_yprev[yp] > eps ? p_yprev[yp] : eps;
                    T pyp_xp = p_yprev_xprev[yp * size_xprev + xp] > eps ?
                               p_yprev_xprev[yp * size_xprev + xp] : eps;
                    T pyt_yp = p_yt_yprev[yt * size_yprev + yp] > eps ?
                               p_yt_yprev[yt * size_yprev + yp] : eps;

                    // T(X -> Y) += p(y_t, y_prev, x_prev) *
                    //              log[p(y_t, y_prev, x_prev) * p(y_prev) / (p(y_prev, x_prev) * p(y_t, y_prev))]
                    result += p_joint * std::log((p_joint * pyp) / (pyp_xp * pyt_yp));
                }
            }
        }
    }

    return result * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
