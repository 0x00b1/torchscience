#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute joint entropy of a 2D joint probability distribution.
 *
 * H(X, Y) = -sum_{x,y} p(x,y) * log(p(x,y))
 *
 * This is simply the Shannon entropy of the flattened joint distribution.
 *
 * @param joint Pointer to joint probability distribution (row-major, |X| x |Y|)
 * @param size_x Number of outcomes for X
 * @param size_y Number of outcomes for Y
 * @param log_base_scale Scale factor for log base conversion (1.0 for nats)
 * @return Joint entropy value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T joint_entropy_kernel(
    const T* joint,
    int64_t size_x,
    int64_t size_y,
    T log_base_scale
) {
  T eps = get_eps<T>();
  T result = T(0);
  int64_t total = size_x * size_y;

  for (int64_t i = 0; i < total; ++i) {
    T p_xy = joint[i] > eps ? joint[i] : eps;
    if (p_xy > eps) {
      result -= p_xy * std::log(p_xy);
    }
  }

  return result * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
