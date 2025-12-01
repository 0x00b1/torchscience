#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute Tsallis entropy of order q.
 *
 * S_q(P) = 1/(q-1) * (1 - sum_i p_i^q)
 *
 * Note: Different normalization than Renyi.
 * As q -> 1, S_q -> Shannon entropy.
 *
 * Properties:
 * - Non-extensive: S_q(A+B) = S_q(A) + S_q(B) + (1-q)*S_q(A)*S_q(B)
 * - q < 1: Favors rare events (sub-extensive)
 * - q > 1: Favors common events (super-extensive)
 *
 * @param p Pointer to probability distribution
 * @param n Size of distribution
 * @param q Order of Tsallis entropy (must be != 1)
 * @return Tsallis entropy value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T tsallis_entropy_kernel(
    const T* p,
    int64_t n,
    T q
) {
  T eps = get_eps<T>();

  T sum_p_q = T(0);
  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    sum_p_q += std::pow(p_i, q);
  }

  return (T(1) - sum_p_q) / (q - T(1));
}

}  // namespace torchscience::kernel::information_theory
