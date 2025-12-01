#pragma once

#include <cmath>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute mutual information I(X;Y) from joint distribution.
 *
 * I(X;Y) = sum_{x,y} p(x,y) * log(p(x,y) / (p(x) * p(y)))
 *        = H(X) + H(Y) - H(X,Y)
 *        = H(Y) - H(Y|X)
 *
 * @param joint Pointer to joint distribution of shape [size_x, size_y]
 * @param size_x Size of X dimension
 * @param size_y Size of Y dimension
 * @param log_base_scale Scale factor for log base conversion (1/log(base))
 * @return Mutual information value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T mutual_information_kernel(
    const T* joint,
    int64_t size_x,
    int64_t size_y,
    T log_base_scale
) {
    T eps = get_eps<T>();
    int64_t total = size_x * size_y;

    // Compute marginals p(x) and p(y)
    T p_x[256];  // Stack-allocated for common sizes
    T p_y[256];

    // For larger sizes, we use the formula I(X;Y) = H(X) + H(Y) - H(X,Y)
    // but compute directly for efficiency

    // Initialize marginals
    for (int64_t i = 0; i < size_x && i < 256; ++i) {
        p_x[i] = T(0);
    }
    for (int64_t j = 0; j < size_y && j < 256; ++j) {
        p_y[j] = T(0);
    }

    // Compute marginals by summing
    for (int64_t i = 0; i < size_x; ++i) {
        for (int64_t j = 0; j < size_y; ++j) {
            T p_xy = joint[i * size_y + j];
            if (i < 256) p_x[i] += p_xy;
            if (j < 256) p_y[j] += p_xy;
        }
    }

    // Compute I(X;Y) = sum p(x,y) log(p(x,y) / (p(x) * p(y)))
    T result = T(0);
    for (int64_t i = 0; i < size_x; ++i) {
        for (int64_t j = 0; j < size_y; ++j) {
            T p_xy = joint[i * size_y + j];
            if (p_xy > eps) {
                T marginal_product = (i < 256 ? p_x[i] : eps) * (j < 256 ? p_y[j] : eps);
                if (marginal_product > eps) {
                    result += p_xy * std::log(p_xy / marginal_product);
                }
            }
        }
    }

    return result * log_base_scale;
}

/**
 * Alternative kernel using dynamic allocation for large dimensions.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T mutual_information_kernel_dynamic(
    const T* joint,
    const T* p_x,
    const T* p_y,
    int64_t size_x,
    int64_t size_y,
    T log_base_scale
) {
    T eps = get_eps<T>();
    T result = T(0);

    for (int64_t i = 0; i < size_x; ++i) {
        for (int64_t j = 0; j < size_y; ++j) {
            T p_xy = joint[i * size_y + j];
            if (p_xy > eps) {
                T marginal_product = p_x[i] * p_y[j];
                if (marginal_product > eps) {
                    result += p_xy * std::log(p_xy / marginal_product);
                }
            }
        }
    }

    return result * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
