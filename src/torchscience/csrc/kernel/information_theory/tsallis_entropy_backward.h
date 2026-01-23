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

/**
 * Compute second-order gradient of Tsallis entropy.
 *
 * g_j = grad_output * (-q/(q-1)) * p_j^(q-1)
 *
 * d(g_j)/d(grad_output) = (-q/(q-1)) * p_j^(q-1)
 * d(g_j)/d(p_j) = grad_output * (-q) * p_j^(q-2)
 *
 * @param gg_p Upstream gradient w.r.t. grad_p
 * @param grad_output Original upstream gradient
 * @param p Pointer to probability distribution
 * @param n Size of distribution
 * @param q Order of Tsallis entropy
 * @param grad_grad_output Output: gradient w.r.t. grad_output
 * @param grad_p Output: second-order gradient w.r.t. p
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void tsallis_entropy_backward_backward_kernel(
    const T* gg_p,
    T grad_output,
    const T* p,
    int64_t n,
    T q,
    T& grad_grad_output,
    T* grad_p
) {
  T eps = get_eps<T>();
  T coeff = -q / (q - T(1));

  grad_grad_output = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T gg_p_i = gg_p ? gg_p[i] : T(0);

    // d(grad_p_i)/d(grad_output) = coeff * p_i^(q-1)
    grad_grad_output += gg_p_i * coeff * std::pow(p_i, q - T(1));

    // d(grad_p_i)/dp_i = grad_output * (-q) * p_i^(q-2)
    grad_p[i] = gg_p_i * grad_output * (-q) * std::pow(p_i, q - T(2));
  }
}

}  // namespace torchscience::kernel::information_theory
