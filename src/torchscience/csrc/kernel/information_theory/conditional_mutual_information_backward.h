#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of conditional mutual information w.r.t. joint distribution.
 *
 * I(X;Y|Z) = sum p(x,y,z) * log[p(x,y,z) * p(z) / (p(x,z) * p(y,z))]
 *
 * The gradient dI/dp(x',y',z') involves contributions from:
 * 1. Direct term from p(x',y',z') in the sum: log(...) + 1
 * 2. Contributions through p(z'): +1
 * 3. Contributions through p(x',z'): -1
 * 4. Contributions through p(y',z'): -1
 *
 * These sum to give:
 * dI/dp(x',y',z') = log[p(x',y',z') * p(z') / (p(x',z') * p(y',z'))]
 *
 * @param grad_output Upstream gradient
 * @param joint Pointer to joint distribution
 * @param p_xz Pointer to marginal p(x,z)
 * @param p_yz Pointer to marginal p(y,z)
 * @param p_z Pointer to marginal p(z)
 * @param size_x, size_y, size_z Dimensions
 * @param log_base_scale Scale factor
 * @param grad_joint Output gradient tensor
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void conditional_mutual_information_backward_kernel(
    T grad_output,
    const T* joint,
    const T* p_xz,
    const T* p_yz,
    const T* p_z,
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
                T p_xyz = joint[idx] > eps ? joint[idx] : eps;
                T pz = p_z[z] > eps ? p_z[z] : eps;
                T pxz = p_xz[x * size_z + z] > eps ? p_xz[x * size_z + z] : eps;
                T pyz = p_yz[y * size_z + z] > eps ? p_yz[y * size_z + z] : eps;

                // dI/dp(x,y,z) = log[p(x,y,z) * p(z) / (p(x,z) * p(y,z))]
                // The full derivative also has contributions from marginals that cancel out
                T log_ratio = std::log((p_xyz * pz) / (pxz * pyz));
                T grad = log_ratio;

                grad_joint[idx] = grad_output * grad * log_base_scale;
            }
        }
    }
}

}  // namespace torchscience::kernel::information_theory
