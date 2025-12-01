#pragma once

#include <cmath>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>()

namespace torchscience::kernel::information_theory {

/**
 * Compute backward pass for chi-squared divergence.
 *
 * χ²(P || Q) = sum_i (p_i - q_i)² / q_i
 *
 * Gradients:
 *   dχ²/dp_i = 2(p_i - q_i) / q_i
 *   dχ²/dq_i = -(p_i² - q_i²) / q_i²
 *
 * @param grad_output Upstream gradient
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @param grad_p Output: gradient w.r.t. p
 * @param grad_q Output: gradient w.r.t. q
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void chi_squared_divergence_backward_kernel(
    T grad_output,
    const T* p,
    const T* q,
    int64_t n,
    T* grad_p,
    T* grad_q
) {
  T eps = get_eps<T>();

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i];
    T q_i = q[i] > eps ? q[i] : eps;
    T diff = p_i - q_i;

    // dχ²/dp_i = 2(p_i - q_i) / q_i
    grad_p[i] = grad_output * T(2) * diff / q_i;

    // dχ²/dq_i = -(p_i² - q_i²) / q_i²
    grad_q[i] = grad_output * (-(p_i * p_i - q_i * q_i) / (q_i * q_i));
  }
}

/**
 * Compute second-order backward pass for chi-squared divergence.
 *
 * Second-order derivatives (used for computing gradients of first-order grads):
 *   d(grad_p_i)/dp_i = grad_output * 2 / q_i
 *   d(grad_p_i)/dq_i = grad_output * (-2p_i / q_i²)
 *   d(grad_q_i)/dp_i = grad_output * (-2p_i / q_i²)
 *   d(grad_q_i)/dq_i = grad_output * 2p_i² / q_i³
 *
 * @param gg_p Upstream gradient w.r.t. grad_p
 * @param gg_q Upstream gradient w.r.t. grad_q
 * @param grad_output Original upstream gradient
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @param grad_grad_output Output: gradient w.r.t. grad_output
 * @param grad_p Output: gradient w.r.t. p
 * @param grad_q Output: gradient w.r.t. q
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void chi_squared_divergence_backward_backward_kernel(
    const T* gg_p,
    const T* gg_q,
    T grad_output,
    const T* p,
    const T* q,
    int64_t n,
    T& grad_grad_output,
    T* grad_p,
    T* grad_q
) {
  T eps = get_eps<T>();
  grad_grad_output = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i];
    T q_i = q[i] > eps ? q[i] : eps;
    T diff = p_i - q_i;
    T gg_p_i = gg_p ? gg_p[i] : T(0);
    T gg_q_i = gg_q ? gg_q[i] : T(0);

    // d(grad_p_i)/d(grad_output) = 2(p_i - q_i) / q_i
    // d(grad_q_i)/d(grad_output) = -(p_i² - q_i²) / q_i²
    grad_grad_output += gg_p_i * (T(2) * diff / q_i)
                      + gg_q_i * (-(p_i * p_i - q_i * q_i) / (q_i * q_i));

    // d(grad_p_i)/dp_i = grad_output * 2 / q_i
    // d(grad_q_i)/dp_i = grad_output * (-2 * p_i / q_i²)
    grad_p[i] = gg_p_i * (grad_output * T(2) / q_i)
              + gg_q_i * (grad_output * (-T(2) * p_i / (q_i * q_i)));

    // d(grad_p_i)/dq_i = grad_output * (-2p_i / q_i²)
    // d(grad_q_i)/dq_i = grad_output * 2p_i² / q_i³
    grad_q[i] = gg_p_i * (grad_output * (-T(2) * p_i / (q_i * q_i)))
              + gg_q_i * (grad_output * T(2) * (p_i * p_i) / (q_i * q_i * q_i));
  }
}

}  // namespace torchscience::kernel::information_theory
