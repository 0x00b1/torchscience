#pragma once

#include <cmath>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute second-order gradients for KL divergence.
 *
 * d^2(D_KL)/d(p_i)^2 = 1 / p_i
 * d^2(D_KL)/d(q_i)^2 = p_i / q_i^2
 * d^2(D_KL)/d(p_i)d(q_i) = -1 / q_i
 *
 * @param gg_p Upstream gradient w.r.t. grad_p
 * @param gg_q Upstream gradient w.r.t. grad_q
 * @param grad_out Original upstream gradient
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @param grad_grad_out Output: gradient w.r.t. grad_out
 * @param grad_p Output: gradient w.r.t. p
 * @param grad_q Output: gradient w.r.t. q
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void kl_divergence_backward_backward_kernel(
    const T* gg_p,
    const T* gg_q,
    T grad_out,
    const T* p,
    const T* q,
    int64_t n,
    T& grad_grad_out,
    T* grad_p,
    T* grad_q
) {
  T eps = get_eps<T>();
  grad_grad_out = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;

    T gg_p_i = gg_p != nullptr ? gg_p[i] : T(0);
    T gg_q_i = gg_q != nullptr ? gg_q[i] : T(0);

    // Gradient of grad_out from backward pass
    // grad_p[i] = grad_out * (log(p_i/q_i) + 1)
    // grad_q[i] = grad_out * (-p_i/q_i)
    grad_grad_out += gg_p_i * (std::log(p_i) - std::log(q_i) + T(1));
    grad_grad_out += gg_q_i * (-p_i / q_i);

    // Second derivative w.r.t. p
    // d/dp_i [grad_out * (log(p_i) - log(q_i) + 1)] = grad_out / p_i
    // d/dp_i [grad_out * (-p_i / q_i)] = -grad_out / q_i
    grad_p[i] = gg_p_i * grad_out / p_i + gg_q_i * (-grad_out / q_i);

    // Second derivative w.r.t. q
    // d/dq_i [grad_out * (log(p_i) - log(q_i) + 1)] = -grad_out / q_i
    // d/dq_i [grad_out * (-p_i / q_i)] = grad_out * p_i / q_i^2
    grad_q[i] = gg_p_i * (-grad_out / q_i) + gg_q_i * (grad_out * p_i / (q_i * q_i));
  }
}

}  // namespace torchscience::kernel::information_theory
