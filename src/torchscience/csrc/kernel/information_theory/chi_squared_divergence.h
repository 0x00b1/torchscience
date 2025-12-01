#pragma once

#include <cmath>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>()

namespace torchscience::kernel::information_theory {

/**
 * Compute chi-squared divergence between two probability distributions.
 *
 * χ²(P || Q) = sum_i (p_i - q_i)² / q_i
 *
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @return Chi-squared divergence value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T chi_squared_divergence_kernel(
    const T* p,
    const T* q,
    int64_t n
) {
  T eps = get_eps<T>();
  T result = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i];
    T q_i = q[i] > eps ? q[i] : eps;
    T diff = p_i - q_i;
    result += diff * diff / q_i;
  }

  return result;
}

}  // namespace torchscience::kernel::information_theory
