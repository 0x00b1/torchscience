#pragma once

#include <cmath>
#include <algorithm>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of PID components with respect to the joint distribution.
 *
 * This is complex due to the min operation in redundancy. We use a subgradient
 * approach: for each z, we take the gradient through whichever specific
 * information term (X or Y) achieved the minimum.
 *
 * For the mutual information terms, the gradients are straightforward:
 * - dI(X,Y;Z)/dp(x,y,z) = log[p(x,y,z)/(p(x,y)*p(z))]
 * - dI(X;Z)/dp(x,y,z) contributes through the marginals
 * - dI(Y;Z)/dp(x,y,z) contributes through the marginals
 *
 * @param grad_redundancy Upstream gradient for redundancy
 * @param grad_unique_x Upstream gradient for unique_x
 * @param grad_unique_y Upstream gradient for unique_y
 * @param grad_synergy Upstream gradient for synergy
 * @param grad_mutual_info Upstream gradient for mutual_info
 * @param joint Pointer to joint distribution p(x, y, z)
 * @param p_x, p_y, p_z Marginals
 * @param p_xy, p_xz, p_yz Joint marginals
 * @param size_x, size_y, size_z Dimensions
 * @param log_base_scale Scale factor
 * @param grad_joint Output gradient tensor
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void partial_information_decomposition_backward_kernel(
    T grad_redundancy,
    T grad_unique_x,
    T grad_unique_y,
    T grad_synergy,
    T grad_mutual_info,
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

    // Total gradient through the decomposition:
    // redundancy contributes: grad_redundancy + grad_synergy (synergy = I_xyz - I_xz - I_yz + R)
    // unique_x = I_xz - R, so dR contributes: -grad_unique_x
    // unique_y = I_yz - R, so dR contributes: -grad_unique_y
    // synergy = I_xyz - I_xz - I_yz + R, so dR contributes: grad_synergy
    T total_grad_redundancy = (grad_redundancy - grad_unique_x - grad_unique_y + grad_synergy) * log_base_scale;

    // Gradient for I(X;Z): unique_x = I_xz - R, synergy = I_xyz - I_xz - I_yz + R
    // So dI_xz: grad_unique_x - grad_synergy
    T total_grad_i_xz = (grad_unique_x - grad_synergy) * log_base_scale;

    // Gradient for I(Y;Z): unique_y = I_yz - R, synergy = I_xyz - I_xz - I_yz + R
    // So dI_yz: grad_unique_y - grad_synergy
    T total_grad_i_yz = (grad_unique_y - grad_synergy) * log_base_scale;

    // Gradient for I(X,Y;Z): synergy = I_xyz - I_xz - I_yz + R
    // Plus direct grad_mutual_info
    T total_grad_i_xyz = (grad_synergy + grad_mutual_info) * log_base_scale;

    // Now compute gradients for each component

    // Pre-compute which source is minimum for each z (for redundancy subgradient)
    // We need to recompute I_spec_X(z) and I_spec_Y(z) for each z
    for (int64_t x = 0; x < size_x; ++x) {
        for (int64_t y = 0; y < size_y; ++y) {
            for (int64_t z = 0; z < size_z; ++z) {
                int64_t idx = (x * size_y + y) * size_z + z;
                T pxyz = joint[idx];
                T px = p_x[x] > eps ? p_x[x] : eps;
                T py = p_y[y] > eps ? p_y[y] : eps;
                T pz = p_z[z] > eps ? p_z[z] : eps;
                T pxy = p_xy[x * size_y + y] > eps ? p_xy[x * size_y + y] : eps;
                T pxz = p_xz[x * size_z + z] > eps ? p_xz[x * size_z + z] : eps;
                T pyz = p_yz[y * size_z + z] > eps ? p_yz[y * size_z + z] : eps;

                T grad = T(0);

                // Gradient from I(X,Y;Z) = sum p(x,y,z) * log[p(x,y,z)/(p(x,y)*p(z))]
                // dI_xyz/dp(x,y,z) = log[p(x,y,z)/(p(x,y)*p(z))]
                // Note: This is a simplified gradient that ignores the
                // contributions through the marginals for computational efficiency.
                // The full gradient would include terms from dp_xy, dp_z, etc.
                if (pxyz > eps) {
                    grad += total_grad_i_xyz * std::log(pxyz / (pxy * pz));
                }

                // Gradient from I(X;Z) = sum p(x,z) * log[p(x,z)/(p(x)*p(z))]
                // Contribution through p(x,z): dp(x,z)/dp(x,y,z) = 1
                // dI_xz/dp(x,y,z) = log[p(x,z)/(p(x)*p(z))]
                if (pxz > eps) {
                    grad += total_grad_i_xz * std::log(pxz / (px * pz));
                }

                // Gradient from I(Y;Z) = sum p(y,z) * log[p(y,z)/(p(y)*p(z))]
                // Contribution through p(y,z): dp(y,z)/dp(x,y,z) = 1
                // dI_yz/dp(x,y,z) = log[p(y,z)/(p(y)*p(z))]
                if (pyz > eps) {
                    grad += total_grad_i_yz * std::log(pyz / (py * pz));
                }

                // Gradient from redundancy (Imin)
                // This is complex due to the min. We use subgradient:
                // For this (x,y,z), the contribution to redundancy goes through
                // whichever specific information (X or Y) is smaller for this z.
                if (pz > eps) {
                    // Recompute I_spec_X(z) and I_spec_Y(z)
                    T i_spec_x = T(0);
                    for (int64_t xx = 0; xx < size_x; ++xx) {
                        T pxxz = p_xz[xx * size_z + z];
                        T pxx = p_x[xx];
                        if (pxxz > eps && pxx > eps) {
                            T p_x_given_z = pxxz / pz;
                            T p_z_given_x = pxxz / pxx;
                            i_spec_x += p_x_given_z * std::log(p_z_given_x / pz);
                        }
                    }

                    T i_spec_y = T(0);
                    for (int64_t yy = 0; yy < size_y; ++yy) {
                        T pyyz = p_yz[yy * size_z + z];
                        T pyy = p_y[yy];
                        if (pyyz > eps && pyy > eps) {
                            T p_y_given_z = pyyz / pz;
                            T p_z_given_y = pyyz / pyy;
                            i_spec_y += p_y_given_z * std::log(p_z_given_y / pz);
                        }
                    }

                    // Subgradient: take gradient through the smaller term
                    // If X is smaller (or equal), gradient goes through X terms
                    // If Y is smaller, gradient goes through Y terms
                    if (i_spec_x <= i_spec_y) {
                        // Gradient through I_spec_X(z)
                        // I_spec_X(z) = sum_x p(x|z) * log[p(z|x)/p(z)]
                        // dp(x,y,z) contributes to p(z) and p(x,z)
                        // This is the gradient of p(z) * I_spec_X(z)

                        // Contribution through p(z): d[p(z)*I_spec_X(z)]/dp(z) = I_spec_X(z) + ...
                        // Contribution through p(x,z):
                        // p(z) * d[sum_x p(x|z)*log(p(z|x)/p(z))]/dp(x,z)
                        // = p(z) * d[(p(x,z)/p(z))*log(p(x,z)/(p(x)*p(z)))]/dp(x,z)

                        // Simplified: gradient of redundancy contribution
                        if (pxz > eps) {
                            // The contribution from this z to redundancy is p(z) * I_spec_X(z)
                            // Taking derivative w.r.t. p(x,y,z):
                            // d/dp(x,y,z) [p(z) * I_spec_X(z)]
                            // = I_spec_X(z) + p(z) * dI_spec_X(z)/dp(x,z)
                            // = I_spec_X(z) + (log(p(x,z)/(p(x)*p(z))) - (1 - p(x|z))
                            // Simplified approximation:
                            grad += total_grad_redundancy * std::log(pxz / (px * pz));
                        }
                    } else {
                        // Gradient through I_spec_Y(z)
                        if (pyz > eps) {
                            grad += total_grad_redundancy * std::log(pyz / (py * pz));
                        }
                    }
                }

                grad_joint[idx] = grad;
            }
        }
    }
}

}  // namespace torchscience::kernel::information_theory
