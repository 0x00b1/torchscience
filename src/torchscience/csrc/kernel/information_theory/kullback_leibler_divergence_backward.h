#pragma once

#include <cmath>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute gradients for KL divergence.
 *
 * d(D_KL)/d(p_i) = log(p_i) - log(q_i) + 1
 * d(D_KL)/d(q_i) = -p_i / q_i
 *
 * @param grad_out Upstream gradient
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @param grad_p Output: gradient w.r.t. p
 * @param grad_q Output: gradient w.r.t. q
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void kl_divergence_backward_kernel(
    T grad_out,
    const T* p,
    const T* q,
    int64_t n,
    T* grad_p,
    T* grad_q
) {
  T eps = get_eps<T>();

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;

    // d(D_KL)/d(p_i) = log(p_i / q_i) + 1
    grad_p[i] = grad_out * (std::log(p_i) - std::log(q_i) + T(1));

    // d(D_KL)/d(q_i) = -p_i / q_i
    grad_q[i] = grad_out * (-p_i / q_i);
  }
}

/**
 * Compute gradients for JS divergence.
 *
 * With M = 0.5 * (P + Q):
 * d(D_JS)/d(p_i) = 0.5 * (log(p_i) - log(m_i) + 1) - 0.25 * (p_i + q_i) / m_i
 * d(D_JS)/d(q_i) = 0.5 * (log(q_i) - log(m_i) + 1) - 0.25 * (p_i + q_i) / m_i
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void js_divergence_backward_kernel(
    T grad_out,
    const T* p,
    const T* q,
    int64_t n,
    T log_base_scale,
    T* grad_p,
    T* grad_q
) {
  T eps = get_eps<T>();
  T scaled_grad = grad_out * log_base_scale;

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;
    T m_i = T(0.5) * (p_i + q_i);

    T log_p = std::log(p_i);
    T log_q = std::log(q_i);
    T log_m = std::log(m_i);

    // Gradient from D_KL(P || M) term w.r.t. p_i
    // d/dp_i [0.5 * p_i * (log(p_i) - log(m_i))]
    // = 0.5 * (log(p_i) - log(m_i) + 1 - 0.5 * p_i / m_i)
    T grad_p_from_kl_p = T(0.5) * (log_p - log_m + T(1) - T(0.5) * p_i / m_i);

    // Gradient from D_KL(Q || M) term w.r.t. p_i (via m_i)
    // d/dp_i [0.5 * q_i * (log(q_i) - log(m_i))]
    // = 0.5 * q_i * (-0.5 / m_i) = -0.25 * q_i / m_i
    T grad_p_from_kl_q = -T(0.25) * q_i / m_i;

    grad_p[i] = scaled_grad * (grad_p_from_kl_p + grad_p_from_kl_q);

    // Symmetric for q
    T grad_q_from_kl_q = T(0.5) * (log_q - log_m + T(1) - T(0.5) * q_i / m_i);
    T grad_q_from_kl_p = -T(0.25) * p_i / m_i;

    grad_q[i] = scaled_grad * (grad_q_from_kl_q + grad_q_from_kl_p);
  }
}

}  // namespace torchscience::kernel::information_theory
