#pragma once

#include <cmath>
#include <vector>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute interaction information I(X;Y;Z) from a 3D joint distribution.
 *
 * I(X;Y;Z) = I(X;Y) - I(X;Y|Z)
 *          = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z)
 *
 * Interaction information can be positive (redundancy) or negative (synergy).
 *
 * @param joint Pointer to joint distribution p(x,y,z) of shape [size_x, size_y, size_z]
 * @param p_x Pointer to marginal p(x) of shape [size_x]
 * @param p_y Pointer to marginal p(y) of shape [size_y]
 * @param p_z Pointer to marginal p(z) of shape [size_z]
 * @param p_xy Pointer to marginal p(x,y) of shape [size_x, size_y]
 * @param p_xz Pointer to marginal p(x,z) of shape [size_x, size_z]
 * @param p_yz Pointer to marginal p(y,z) of shape [size_y, size_z]
 * @param size_x Size of X dimension
 * @param size_y Size of Y dimension
 * @param size_z Size of Z dimension
 * @param log_base_scale Scale factor for log base conversion
 * @return Interaction information value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T interaction_information_kernel(
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
    T log_base_scale
) {
    T eps = get_eps<T>();

    // Compute H(X) = -sum p(x) log p(x)
    T h_x = T(0);
    for (int64_t x = 0; x < size_x; ++x) {
        if (p_x[x] > eps) {
            h_x -= p_x[x] * std::log(p_x[x]);
        }
    }

    // Compute H(Y) = -sum p(y) log p(y)
    T h_y = T(0);
    for (int64_t y = 0; y < size_y; ++y) {
        if (p_y[y] > eps) {
            h_y -= p_y[y] * std::log(p_y[y]);
        }
    }

    // Compute H(Z) = -sum p(z) log p(z)
    T h_z = T(0);
    for (int64_t z = 0; z < size_z; ++z) {
        if (p_z[z] > eps) {
            h_z -= p_z[z] * std::log(p_z[z]);
        }
    }

    // Compute H(X,Y) = -sum p(x,y) log p(x,y)
    T h_xy = T(0);
    for (int64_t x = 0; x < size_x; ++x) {
        for (int64_t y = 0; y < size_y; ++y) {
            T pxy = p_xy[x * size_y + y];
            if (pxy > eps) {
                h_xy -= pxy * std::log(pxy);
            }
        }
    }

    // Compute H(X,Z) = -sum p(x,z) log p(x,z)
    T h_xz = T(0);
    for (int64_t x = 0; x < size_x; ++x) {
        for (int64_t z = 0; z < size_z; ++z) {
            T pxz = p_xz[x * size_z + z];
            if (pxz > eps) {
                h_xz -= pxz * std::log(pxz);
            }
        }
    }

    // Compute H(Y,Z) = -sum p(y,z) log p(y,z)
    T h_yz = T(0);
    for (int64_t y = 0; y < size_y; ++y) {
        for (int64_t z = 0; z < size_z; ++z) {
            T pyz = p_yz[y * size_z + z];
            if (pyz > eps) {
                h_yz -= pyz * std::log(pyz);
            }
        }
    }

    // Compute H(X,Y,Z) = -sum p(x,y,z) log p(x,y,z)
    T h_xyz = T(0);
    for (int64_t x = 0; x < size_x; ++x) {
        for (int64_t y = 0; y < size_y; ++y) {
            for (int64_t z = 0; z < size_z; ++z) {
                T pxyz = joint[(x * size_y + y) * size_z + z];
                if (pxyz > eps) {
                    h_xyz -= pxyz * std::log(pxyz);
                }
            }
        }
    }

    // I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z)
    T ii = h_x + h_y + h_z - h_xy - h_xz - h_yz + h_xyz;

    return ii * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
