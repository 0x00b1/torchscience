#pragma once

#include <cmath>
#include <algorithm>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute specific information I_spec(X=x; Z) = sum_z p(z|x) * log[p(z|x) / p(z)]
 *
 * This is the information that a specific value x provides about Z.
 *
 * @param p_z_given_x Pointer to p(z|x) for this specific x, shape [size_z]
 * @param p_z Pointer to marginal p(z), shape [size_z]
 * @param size_z Size of Z dimension
 * @return Specific information for this x value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T specific_information_kernel(
    const T* p_z_given_x,
    const T* p_z,
    int64_t size_z
) {
    T eps = get_eps<T>();
    T result = T(0);

    for (int64_t z = 0; z < size_z; ++z) {
        T pzx = p_z_given_x[z];
        T pz = p_z[z] > eps ? p_z[z] : eps;

        if (pzx > eps) {
            result += pzx * std::log(pzx / pz);
        }
    }

    return result;
}

/**
 * Compute Williams-Beer Imin redundancy.
 *
 * The redundancy R(X,Y;Z) using the Imin measure (Williams & Beer 2010):
 *
 * For each z, compute the minimum specific information from either source:
 *   min_info(z) = p(z) * min(I_spec_X(z), I_spec_Y(z))
 *
 * where:
 *   I_spec_X(z) = sum_x p(x|z) * log[p(z|x) / p(z)]
 *   I_spec_Y(z) = sum_y p(y|z) * log[p(z|y) / p(z)]
 *
 * Redundancy = sum_z min_info(z)
 *
 * @param joint Pointer to joint distribution p(x, y, z) of shape [size_x, size_y, size_z]
 * @param p_x Pointer to marginal p(x) of shape [size_x]
 * @param p_y Pointer to marginal p(y) of shape [size_y]
 * @param p_z Pointer to marginal p(z) of shape [size_z]
 * @param p_xz Pointer to joint p(x, z) of shape [size_x, size_z]
 * @param p_yz Pointer to joint p(y, z) of shape [size_y, size_z]
 * @param size_x Size of X dimension
 * @param size_y Size of Y dimension
 * @param size_z Size of Z dimension
 * @param log_base_scale Scale factor for log base conversion
 * @return Imin redundancy value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T imin_redundancy_kernel(
    const T* joint,
    const T* p_x,
    const T* p_y,
    const T* p_z,
    const T* p_xz,
    const T* p_yz,
    int64_t size_x,
    int64_t size_y,
    int64_t size_z,
    T log_base_scale
) {
    T eps = get_eps<T>();
    T redundancy = T(0);

    // For each z, compute min(I_spec_X(z), I_spec_Y(z))
    for (int64_t z = 0; z < size_z; ++z) {
        T pz = p_z[z];
        if (pz < eps) continue;

        // Compute I_spec_X(z) = sum_x p(x|z) * log[p(z|x) / p(z)]
        // = sum_x [p(x,z)/p(z)] * log[p(x,z)/(p(x)*p(z))]
        T i_spec_x = T(0);
        for (int64_t x = 0; x < size_x; ++x) {
            T pxz = p_xz[x * size_z + z];
            T px = p_x[x];
            if (pxz > eps && px > eps) {
                T p_x_given_z = pxz / pz;
                T p_z_given_x = pxz / px;
                // I_spec_X(z) = sum_x p(x|z) * log[p(z|x) / p(z)]
                i_spec_x += p_x_given_z * std::log(p_z_given_x / pz);
            }
        }

        // Compute I_spec_Y(z) = sum_y p(y|z) * log[p(z|y) / p(z)]
        T i_spec_y = T(0);
        for (int64_t y = 0; y < size_y; ++y) {
            T pyz = p_yz[y * size_z + z];
            T py = p_y[y];
            if (pyz > eps && py > eps) {
                T p_y_given_z = pyz / pz;
                T p_z_given_y = pyz / py;
                i_spec_y += p_y_given_z * std::log(p_z_given_y / pz);
            }
        }

        // Redundancy contribution from this z
        // Use min of the two specific information values
        T min_spec = std::min(i_spec_x, i_spec_y);
        // Weight by p(z)
        redundancy += pz * min_spec;
    }

    return redundancy * log_base_scale;
}

/**
 * Compute mutual information I(X;Z) from marginals.
 *
 * I(X;Z) = sum_{x,z} p(x,z) * log[p(x,z) / (p(x)*p(z))]
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T mi_xz_kernel(
    const T* p_xz,
    const T* p_x,
    const T* p_z,
    int64_t size_x,
    int64_t size_z,
    T log_base_scale
) {
    T eps = get_eps<T>();
    T mi = T(0);

    for (int64_t x = 0; x < size_x; ++x) {
        for (int64_t z = 0; z < size_z; ++z) {
            T pxz = p_xz[x * size_z + z];
            T px = p_x[x];
            T pz = p_z[z];
            if (pxz > eps && px > eps && pz > eps) {
                mi += pxz * std::log(pxz / (px * pz));
            }
        }
    }

    return mi * log_base_scale;
}

/**
 * Compute mutual information I(X,Y;Z) between joint source (X,Y) and target Z.
 *
 * I(X,Y;Z) = sum_{x,y,z} p(x,y,z) * log[p(x,y,z) / (p(x,y)*p(z))]
 *          = H(Z) + H(X,Y) - H(X,Y,Z)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T mi_xyz_kernel(
    const T* joint,
    const T* p_xy,
    const T* p_z,
    int64_t size_x,
    int64_t size_y,
    int64_t size_z,
    T log_base_scale
) {
    T eps = get_eps<T>();
    T mi = T(0);

    for (int64_t x = 0; x < size_x; ++x) {
        for (int64_t y = 0; y < size_y; ++y) {
            for (int64_t z = 0; z < size_z; ++z) {
                T pxyz = joint[(x * size_y + y) * size_z + z];
                T pxy = p_xy[x * size_y + y];
                T pz = p_z[z];
                if (pxyz > eps && pxy > eps && pz > eps) {
                    mi += pxyz * std::log(pxyz / (pxy * pz));
                }
            }
        }
    }

    return mi * log_base_scale;
}

/**
 * Compute full PID decomposition.
 *
 * Given joint p(x,y,z), compute:
 * - Redundancy R(X,Y;Z): Information both X and Y share about Z (Imin measure)
 * - Unique(X): I(X;Z) - Redundancy
 * - Unique(Y): I(Y;Z) - Redundancy
 * - Synergy: I(X,Y;Z) - I(X;Z) - I(Y;Z) + Redundancy
 * - Mutual Information: I(X,Y;Z)
 *
 * These satisfy: I(X,Y;Z) = Redundancy + Unique(X) + Unique(Y) + Synergy
 *
 * @param joint Pointer to joint distribution p(x, y, z) of shape [size_x, size_y, size_z]
 * @param p_x Pointer to marginal p(x) of shape [size_x]
 * @param p_y Pointer to marginal p(y) of shape [size_y]
 * @param p_z Pointer to marginal p(z) of shape [size_z]
 * @param p_xy Pointer to joint p(x, y) of shape [size_x, size_y]
 * @param p_xz Pointer to joint p(x, z) of shape [size_x, size_z]
 * @param p_yz Pointer to joint p(y, z) of shape [size_y, size_z]
 * @param size_x, size_y, size_z Dimensions
 * @param log_base_scale Scale factor for log base conversion
 * @param redundancy Output redundancy
 * @param unique_x Output unique information from X
 * @param unique_y Output unique information from Y
 * @param synergy Output synergistic information
 * @param mutual_info Output I(X,Y;Z)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void partial_information_decomposition_kernel(
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
    T* redundancy,
    T* unique_x,
    T* unique_y,
    T* synergy,
    T* mutual_info
) {
    // Compute I(X;Z) and I(Y;Z)
    T i_xz = mi_xz_kernel(p_xz, p_x, p_z, size_x, size_z, T(1));
    T i_yz = mi_xz_kernel(p_yz, p_y, p_z, size_y, size_z, T(1));

    // Compute I(X,Y;Z)
    T i_xyz = mi_xyz_kernel(joint, p_xy, p_z, size_x, size_y, size_z, T(1));

    // Compute Imin redundancy
    T red = imin_redundancy_kernel(joint, p_x, p_y, p_z, p_xz, p_yz,
                                    size_x, size_y, size_z, T(1));

    // Compute other components
    T uniq_x = i_xz - red;
    T uniq_y = i_yz - red;
    T syn = i_xyz - i_xz - i_yz + red;

    // Apply log base scaling
    *redundancy = red * log_base_scale;
    *unique_x = uniq_x * log_base_scale;
    *unique_y = uniq_y * log_base_scale;
    *synergy = syn * log_base_scale;
    *mutual_info = i_xyz * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
