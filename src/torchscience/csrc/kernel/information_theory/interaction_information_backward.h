#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of interaction information w.r.t. joint distribution.
 *
 * I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z)
 *
 * The gradient d I(X;Y;Z) / dp(x,y,z) involves contributions from each term:
 *
 * For entropy terms: d[-sum p log p]/dp = -(log p + 1)
 *
 * d H(X) / dp(x,y,z) = -(log p(x) + 1)  [contributes to sum over y,z]
 * d H(Y) / dp(x,y,z) = -(log p(y) + 1)  [contributes to sum over x,z]
 * d H(Z) / dp(x,y,z) = -(log p(z) + 1)  [contributes to sum over x,y]
 * d H(X,Y) / dp(x,y,z) = -(log p(x,y) + 1)  [contributes to sum over z]
 * d H(X,Z) / dp(x,y,z) = -(log p(x,z) + 1)  [contributes to sum over y]
 * d H(Y,Z) / dp(x,y,z) = -(log p(y,z) + 1)  [contributes to sum over x]
 * d H(X,Y,Z) / dp(x,y,z) = -(log p(x,y,z) + 1)  [direct]
 *
 * Total gradient:
 * d II / dp(x,y,z) = -(log p(x) + 1) - (log p(y) + 1) - (log p(z) + 1)
 *                  + (log p(x,y) + 1) + (log p(x,z) + 1) + (log p(y,z) + 1)
 *                  - (log p(x,y,z) + 1)
 *                = -log p(x) - log p(y) - log p(z) + log p(x,y) + log p(x,z) + log p(y,z) - log p(x,y,z) - 1
 *
 * @param grad_output Upstream gradient
 * @param joint Pointer to joint distribution
 * @param p_x, p_y, p_z Marginal distributions
 * @param p_xy, p_xz, p_yz Pairwise marginals
 * @param size_x, size_y, size_z Dimensions
 * @param log_base_scale Scale factor
 * @param grad_joint Output gradient tensor
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void interaction_information_backward_kernel(
    T grad_output,
    const T* joint,
    const T* p_x,
    const T* p_y,
    const T* p_z,
    const T* p_xy,
    const T* p_xz,
    const T* p_yz,
    int64_t size_x,
    int64_t size_y,
    int64_t size_z,
    T log_base_scale,
    T* grad_joint
) {
    T eps = get_eps<T>();

    // Compute gradients
    for (int64_t x = 0; x < size_x; ++x) {
        for (int64_t y = 0; y < size_y; ++y) {
            for (int64_t z = 0; z < size_z; ++z) {
                int64_t idx = (x * size_y + y) * size_z + z;

                T px = p_x[x] > eps ? p_x[x] : eps;
                T py = p_y[y] > eps ? p_y[y] : eps;
                T pz = p_z[z] > eps ? p_z[z] : eps;
                T pxy = p_xy[x * size_y + y] > eps ? p_xy[x * size_y + y] : eps;
                T pxz = p_xz[x * size_z + z] > eps ? p_xz[x * size_z + z] : eps;
                T pyz = p_yz[y * size_z + z] > eps ? p_yz[y * size_z + z] : eps;
                T pxyz = joint[idx] > eps ? joint[idx] : eps;

                // d II / dp(x,y,z) = -log p(x) - log p(y) - log p(z)
                //                  + log p(x,y) + log p(x,z) + log p(y,z)
                //                  - log p(x,y,z) - 1
                T grad = -std::log(px) - std::log(py) - std::log(pz)
                       + std::log(pxy) + std::log(pxz) + std::log(pyz)
                       - std::log(pxyz) - T(1);

                grad_joint[idx] = grad_output * grad * log_base_scale;
            }
        }
    }
}

}  // namespace torchscience::kernel::information_theory
