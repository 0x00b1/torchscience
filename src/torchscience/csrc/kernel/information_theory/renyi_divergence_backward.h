#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of Renyi divergence w.r.t. input probabilities.
 *
 * D_alpha = 1/(alpha-1) * log(sum_i p_i^alpha * q_i^(1-alpha))
 *
 * Let S = sum_i p_i^alpha * q_i^(1-alpha)
 *
 * dD_alpha/dp_j = alpha / ((alpha-1) * S) * p_j^(alpha-1) * q_j^(1-alpha)
 * dD_alpha/dq_j = -1/S * p_j^alpha * q_j^(-alpha)
 *
 * @param grad_output Upstream gradient (scalar)
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @param alpha Order of Renyi divergence
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_p Output: gradient w.r.t. p
 * @param grad_q Output: gradient w.r.t. q
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void renyi_divergence_backward_kernel(
    T grad_output,
    const T* p,
    const T* q,
    int64_t n,
    T alpha,
    T log_base_scale,
    T* grad_p,
    T* grad_q
) {
  T eps = get_eps<T>();

  // Compute S = sum_i p_i^alpha * q_i^(1-alpha)
  T S = T(0);
  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;
    S += std::pow(p_i, alpha) * std::pow(q_i, T(1) - alpha);
  }

  T coeff_p = grad_output * log_base_scale * alpha / ((alpha - T(1)) * S);
  T coeff_q = -grad_output * log_base_scale / S;

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;

    // dD/dp_j = alpha / ((alpha-1) * S) * p_j^(alpha-1) * q_j^(1-alpha)
    grad_p[i] = coeff_p * std::pow(p_i, alpha - T(1)) * std::pow(q_i, T(1) - alpha);

    // dD/dq_j = -1/S * p_j^alpha * q_j^(-alpha)
    grad_q[i] = coeff_q * std::pow(p_i, alpha) * std::pow(q_i, -alpha);
  }
}

}  // namespace torchscience::kernel::information_theory
