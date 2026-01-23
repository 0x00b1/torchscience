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

/**
 * Compute second-order gradient of Renyi entropy.
 *
 * H_α = (1/(1-α)) · log(S)  where S = Σ p_i^α
 * g_j = grad_output · log_base_scale · α / ((1-α)·S) · p_j^(α-1)
 *
 * @param gg_p Upstream gradient w.r.t. grad_p
 * @param grad_output Original upstream gradient
 * @param p Pointer to probability distribution
 * @param n Size of distribution
 * @param alpha Order of Renyi entropy
 * @param log_base_scale Scale factor for log base
 * @param grad_grad_output Output: gradient w.r.t. grad_output
 * @param grad_p Output: second-order gradient w.r.t. p
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void renyi_entropy_backward_backward_kernel(
    const T* gg_p,
    T grad_output,
    const T* p,
    int64_t n,
    T alpha,
    T log_base_scale,
    T& grad_grad_output,
    T* grad_p
) {
  T eps = get_eps<T>();

  // Handle special cases (alpha=0 or alpha->inf have zero gradients)
  if (alpha < eps || alpha > T(100)) {
    grad_grad_output = T(0);
    for (int64_t i = 0; i < n; ++i) {
      grad_p[i] = T(0);
    }
    return;
  }

  // Compute S = Σ p_i^α
  T S = T(0);
  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    S += std::pow(p_i, alpha);
  }

  T coeff = log_base_scale * alpha / (T(1) - alpha);

  grad_grad_output = T(0);

  // First pass: compute grad_grad_output and accumulate gg_dot_p_alpha_m1
  T gg_dot_p_alpha_m1 = T(0);  // Σ gg_p_i · p_i^(α-1)
  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T gg_p_i = gg_p ? gg_p[i] : T(0);
    T p_alpha_m1 = std::pow(p_i, alpha - T(1));

    // ∂g_i/∂grad_output = coeff / S · p_i^(α-1)
    grad_grad_output += gg_p_i * coeff / S * p_alpha_m1;

    gg_dot_p_alpha_m1 += gg_p_i * p_alpha_m1;
  }

  // Second pass: compute grad_p
  // grad_p[i] = Σ_j gg_p_j · ∂g_j/∂p_i
  // = coeff/S · [gg_p_i·(α-1)·p_i^(α-2) - (Σ_j gg_p_j·p_j^(α-1))·α·p_i^(α-1)/S]
  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T gg_p_i = gg_p ? gg_p[i] : T(0);

    T diag_term = gg_p_i * (alpha - T(1)) * std::pow(p_i, alpha - T(2));
    T off_diag_term = gg_dot_p_alpha_m1 * alpha * std::pow(p_i, alpha - T(1)) / S;

    grad_p[i] = grad_output * coeff / S * (diag_term - off_diag_term);
  }
}

}  // namespace torchscience::kernel::information_theory
