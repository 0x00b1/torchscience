#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of Tsallis entropy w.r.t. input probabilities.
 *
 * S_q = (1 - sum_i p_i^q) / (q - 1)
 *
 * dS_q/dp_j = -q * p_j^(q-1) / (q - 1)
 *
 * @param grad_output Upstream gradient (scalar)
 * @param p Pointer to probability distribution
 * @param n Size of distribution
 * @param q Order of Tsallis entropy
 * @param grad_p Output: gradient w.r.t. p
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void tsallis_entropy_backward_kernel(
    T grad_output,
    const T* p,
    int64_t n,
    T q,
    T* grad_p
) {
  T eps = get_eps<T>();
  T coeff = -grad_output * q / (q - T(1));

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    grad_p[i] = coeff * std::pow(p_i, q - T(1));
  }
}

}  // namespace torchscience::kernel::information_theory
