#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute causally conditioned entropy H(Y||X) = H(Y_t | Y_{t-1}, X_t) from a 3D joint distribution.
 *
 * H(Y_t | Y_{t-1}, X_t) = -sum_{y_t, y_prev, x_t} p(y_t, y_prev, x_t) * log p(y_t | y_prev, x_t)
 *
 * where:
 *   p(y_t | y_prev, x_t) = p(y_t, y_prev, x_t) / p(y_prev, x_t)
 *
 * So:
 *   H(Y||X) = -sum p(y_t, y_prev, x_t) * [log p(y_t, y_prev, x_t) - log p(y_prev, x_t)]
 *           = -sum p(y_t, y_prev, x_t) * log p(y_t, y_prev, x_t)
 *             + sum p(y_t, y_prev, x_t) * log p(y_prev, x_t)
 *           = H(Y_t, Y_{t-1}, X_t) - H(Y_{t-1}, X_t)  (entropies)
 *
 * The input joint tensor has shape [size_yt, size_yprev, size_xt]
 * representing p(y_t, y_{t-1}, x_t).
 *
 * @param joint Pointer to joint distribution p(y_t, y_prev, x_t)
 * @param p_yprev_xt Pointer to marginal p(y_prev, x_t) shape [size_yprev, size_xt]
 * @param size_yt Size of Y_t dimension
 * @param size_yprev Size of Y_{t-1} dimension
 * @param size_xt Size of X_t dimension
 * @param log_base_scale Scale factor for log base conversion
 * @return Causally conditioned entropy value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T causally_conditioned_entropy_kernel(
    const T* joint,
    const T* p_yprev_xt,
    int64_t size_yt,
    int64_t size_yprev,
    int64_t size_xt,
    T log_base_scale
) {
    T eps = get_eps<T>();
    T result = T(0);

    // Compute H(Y_t | Y_{t-1}, X_t) = -sum p(y_t, y_prev, x_t) * log[p(y_t, y_prev, x_t) / p(y_prev, x_t)]
    for (int64_t yt = 0; yt < size_yt; ++yt) {
        for (int64_t yp = 0; yp < size_yprev; ++yp) {
            for (int64_t xt = 0; xt < size_xt; ++xt) {
                T p_joint = joint[(yt * size_yprev + yp) * size_xt + xt];
                if (p_joint > eps) {
                    T pyp_xt = p_yprev_xt[yp * size_xt + xt] > eps ?
                               p_yprev_xt[yp * size_xt + xt] : eps;

                    // H = -sum p * log(p / p_cond) = -sum p * [log p - log p_cond]
                    // p_cond = p(y_t | y_prev, x_t) = p_joint / pyp_xt
                    // So: H = -sum p_joint * log(p_joint / pyp_xt)
                    result -= p_joint * std::log(p_joint / pyp_xt);
                }
            }
        }
    }

    return result * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
