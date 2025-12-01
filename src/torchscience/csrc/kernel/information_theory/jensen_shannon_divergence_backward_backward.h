#pragma once

#include <cmath>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute second-order gradients for JS divergence.
 * Implementation follows the same pattern as KL but with JS-specific derivatives.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void js_divergence_backward_backward_kernel(
    const T* gg_p,
    const T* gg_q,
    T grad_out,
    const T* p,
    const T* q,
    int64_t n,
    T log_base_scale,
    T& grad_grad_out,
    T* grad_p,
    T* grad_q
) {
  T eps = get_eps<T>();
  T scaled_grad = grad_out * log_base_scale;
  grad_grad_out = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;
    T m_i = T(0.5) * (p_i + q_i);

    T gg_p_i = gg_p != nullptr ? gg_p[i] : T(0);
    T gg_q_i = gg_q != nullptr ? gg_q[i] : T(0);

    T log_p = std::log(p_i);
    T log_q = std::log(q_i);
    T log_m = std::log(m_i);

    // First derivatives (from backward)
    T grad_p_from_kl_p = T(0.5) * (log_p - log_m + T(1) - T(0.5) * p_i / m_i);
    T grad_p_from_kl_q = -T(0.25) * q_i / m_i;
    T d_dp = grad_p_from_kl_p + grad_p_from_kl_q;

    T grad_q_from_kl_q = T(0.5) * (log_q - log_m + T(1) - T(0.5) * q_i / m_i);
    T grad_q_from_kl_p = -T(0.25) * p_i / m_i;
    T d_dq = grad_q_from_kl_q + grad_q_from_kl_p;

    // Gradient of grad_out
    grad_grad_out += (gg_p_i * d_dp + gg_q_i * d_dq) * log_base_scale;

    // Second derivatives (simplified)
    // d^2/dp^2 = 0.5/p - 0.25/m + 0.125*(p+q)/m^2
    T d2_dp2 = T(0.5) / p_i - T(0.25) / m_i + T(0.125) * (p_i + q_i) / (m_i * m_i);

    // d^2/dq^2 (symmetric)
    T d2_dq2 = T(0.5) / q_i - T(0.25) / m_i + T(0.125) * (p_i + q_i) / (m_i * m_i);

    // d^2/dpdq
    T d2_dpdq = T(0.125) * (p_i + q_i) / (m_i * m_i) - T(0.25) / m_i;

    grad_p[i] = scaled_grad * (gg_p_i * d2_dp2 + gg_q_i * d2_dpdq);
    grad_q[i] = scaled_grad * (gg_p_i * d2_dpdq + gg_q_i * d2_dq2);
  }
}

}  // namespace torchscience::kernel::information_theory
