#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of Renyi entropy w.r.t. input probabilities.
 *
 * H_alpha = 1/(1-alpha) * log(sum_i p_i^alpha)
 *
 * dH_alpha/dp_j = 1/(1-alpha) * (alpha * p_j^(alpha-1)) / sum_i p_i^alpha
 *               = alpha / ((1-alpha) * sum_i p_i^alpha) * p_j^(alpha-1)
 *
 * @param grad_output Upstream gradient (scalar)
 * @param p Pointer to probability distribution
 * @param n Size of distribution
 * @param alpha Order of Renyi entropy
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_p Output: gradient w.r.t. p
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void renyi_entropy_backward_kernel(
    T grad_output,
    const T* p,
    int64_t n,
    T alpha,
    T log_base_scale,
    T* grad_p
) {
  T eps = get_eps<T>();

  // Handle special case: alpha = 0 (no gradient since it only depends on support)
  if (alpha < eps) {
    for (int64_t i = 0; i < n; ++i) {
      grad_p[i] = T(0);
    }
    return;
  }

  // Handle special case: alpha -> infinity (min-entropy)
  if (alpha > T(100)) {
    // Gradient only at the maximum element
    T max_p = T(0);
    int64_t max_idx = 0;
    for (int64_t i = 0; i < n; ++i) {
      if (p[i] > max_p) {
        max_p = p[i];
        max_idx = i;
      }
    }
    for (int64_t i = 0; i < n; ++i) {
      grad_p[i] = T(0);
    }
    grad_p[max_idx] = -grad_output * log_base_scale / (max_p > eps ? max_p : eps);
    return;
  }

  // General case
  T sum_p_alpha = T(0);
  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    sum_p_alpha += std::pow(p_i, alpha);
  }

  T coeff = grad_output * log_base_scale * alpha / ((T(1) - alpha) * sum_p_alpha);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    grad_p[i] = coeff * std::pow(p_i, alpha - T(1));
  }
}

}  // namespace torchscience::kernel::information_theory
