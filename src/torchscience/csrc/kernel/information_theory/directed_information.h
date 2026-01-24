#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute directed information I(X -> Y) from a 3D joint distribution.
 *
 * I(X -> Y) = I(X_t; Y_t | Y_{t-1})
 *           = sum_{y_t, y_prev, x_t} p(y_t, y_prev, x_t) *
 *             log[p(y_t, y_prev, x_t) * p(y_prev) / (p(y_t, y_prev) * p(y_prev, x_t))]
 *
 * This measures the causal information flow from X to Y at the current time step,
 * conditioning on the past of Y.
 *
 * The input joint tensor has shape [size_yt, size_yprev, size_xt]
 * representing p(y_t, y_{t-1}, x_t).
 *
 * @param joint Pointer to joint distribution p(y_t, y_prev, x_t)
 * @param p_yprev_xt Pointer to marginal p(y_prev, x_t) shape [size_yprev, size_xt]
 * @param p_yt_yprev Pointer to marginal p(y_t, y_prev) shape [size_yt, size_yprev]
 * @param p_yprev Pointer to marginal p(y_prev) shape [size_yprev]
 * @param size_yt Size of Y_t dimension
 * @param size_yprev Size of Y_{t-1} dimension
 * @param size_xt Size of X_t dimension
 * @param log_base_scale Scale factor for log base conversion
 * @return Directed information value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T directed_information_kernel(
    const T* joint,
    const T* p_yprev_xt,
    const T* p_yt_yprev,
    const T* p_yprev,
    int64_t size_yt,
    int64_t size_yprev,
    int64_t size_xt,
    T log_base_scale
) {
    T eps = get_eps<T>();
    T result = T(0);

    // Compute I(X -> Y) = sum p(y_t, y_prev, x_t) *
    //                     log[p(y_t, y_prev, x_t) * p(y_prev) / (p(y_t, y_prev) * p(y_prev, x_t))]
    for (int64_t yt = 0; yt < size_yt; ++yt) {
        for (int64_t yp = 0; yp < size_yprev; ++yp) {
            for (int64_t xt = 0; xt < size_xt; ++xt) {
                T p_joint = joint[(yt * size_yprev + yp) * size_xt + xt];
                if (p_joint > eps) {
                    T pyp = p_yprev[yp] > eps ? p_yprev[yp] : eps;
                    T pyp_xt = p_yprev_xt[yp * size_xt + xt] > eps ?
                               p_yprev_xt[yp * size_xt + xt] : eps;
                    T pyt_yp = p_yt_yprev[yt * size_yprev + yp] > eps ?
                               p_yt_yprev[yt * size_yprev + yp] : eps;

                    // I(X -> Y) += p(y_t, y_prev, x_t) *
                    //              log[p(y_t, y_prev, x_t) * p(y_prev) / (p(y_t, y_prev) * p(y_prev, x_t))]
                    result += p_joint * std::log((p_joint * pyp) / (pyt_yp * pyp_xt));
                }
            }
        }
    }

    return result * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
