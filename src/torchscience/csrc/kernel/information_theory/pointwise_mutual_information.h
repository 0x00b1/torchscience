#pragma once

#include <cmath>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute pointwise mutual information PMI(x,y) for a single element.
 *
 * PMI(x,y) = log(p(x,y) / (p(x) * p(y)))
 *
 * @param p_xy Joint probability p(x,y)
 * @param p_x Marginal probability p(x)
 * @param p_y Marginal probability p(y)
 * @param log_base_scale Scale factor for log base conversion (1/log(base))
 * @return Pointwise mutual information value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T pointwise_mutual_information_kernel(
    T p_xy,
    T p_x,
    T p_y,
    T log_base_scale
) {
    T eps = get_eps<T>();

    if (p_xy <= eps || p_x <= eps || p_y <= eps) {
        // Return -inf for zero probabilities (log(0) = -inf)
        // But clamp to large negative to avoid NaN in backward
        return T(-100) * log_base_scale;
    }

    T marginal_product = p_x * p_y;
    return std::log(p_xy / marginal_product) * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
