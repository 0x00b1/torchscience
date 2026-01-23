#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute conditional mutual information I(X;Y|Z) from a 3D joint distribution.
 *
 * I(X;Y|Z) = sum_{x,y,z} p(x,y,z) * log[p(x,y|z) / (p(x|z) * p(y|z))]
 *          = sum_{x,y,z} p(x,y,z) * [log p(x,y,z) - log p(x,z) - log p(y,z) + log p(z)]
 *
 * @param joint Pointer to joint distribution p(x,y,z) of shape [size_x, size_y, size_z]
 * @param p_xz Pointer to marginal p(x,z) of shape [size_x, size_z]
 * @param p_yz Pointer to marginal p(y,z) of shape [size_y, size_z]
 * @param p_z Pointer to marginal p(z) of shape [size_z]
 * @param size_x Size of X dimension
 * @param size_y Size of Y dimension
 * @param size_z Size of Z dimension
 * @param log_base_scale Scale factor for log base conversion
 * @return Conditional mutual information value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T conditional_mutual_information_kernel(
    const T* joint,
    const T* p_xz,
    const T* p_yz,
    const T* p_z,
    int64_t size_x,
    int64_t size_y,
    int64_t size_z,
    T log_base_scale
) {
    T eps = get_eps<T>();
    T result = T(0);

    // Compute I(X;Y|Z) = sum p(x,y,z) * log[p(x,y,z) * p(z) / (p(x,z) * p(y,z))]
    for (int64_t x = 0; x < size_x; ++x) {
        for (int64_t y = 0; y < size_y; ++y) {
            for (int64_t z = 0; z < size_z; ++z) {
                T p_xyz = joint[(x * size_y + y) * size_z + z];
                if (p_xyz > eps) {
                    T pz = p_z[z] > eps ? p_z[z] : eps;
                    T pxz = p_xz[x * size_z + z] > eps ? p_xz[x * size_z + z] : eps;
                    T pyz = p_yz[y * size_z + z] > eps ? p_yz[y * size_z + z] : eps;

                    // I(X;Y|Z) += p(x,y,z) * log[p(x,y,z) * p(z) / (p(x,z) * p(y,z))]
                    result += p_xyz * std::log((p_xyz * pz) / (pxz * pyz));
                }
            }
        }
    }

    return result * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
