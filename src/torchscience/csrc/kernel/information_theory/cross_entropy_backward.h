#pragma once

#include <cmath>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>()

namespace torchscience::kernel::information_theory {

/**
 * Compute backward pass for cross-entropy.
 *
 * H(P, Q) = -sum_i p_i * log(q_i)
 *
 * Gradients:
 *   dH/dp_i = -log(q_i) * log_base_scale
 *   dH/dq_i = -p_i / q_i * log_base_scale
 *
 * @param grad_output Upstream gradient
 * @param p Pointer to true/target probability distribution
 * @param q Pointer to predicted probability distribution
 * @param n Size of distributions
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_p Output: gradient w.r.t. p
 * @param grad_q Output: gradient w.r.t. q
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void cross_entropy_backward_kernel(
    T grad_output,
    const T* p,
    const T* q,
    int64_t n,
    T log_base_scale,
    T* grad_p,
    T* grad_q
) {
  T eps = get_eps<T>();

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i];
    T q_i = q[i] > eps ? q[i] : eps;

    // dH/dp_i = -log(q_i) * log_base_scale
    grad_p[i] = grad_output * (-std::log(q_i)) * log_base_scale;

    // dH/dq_i = -p_i / q_i * log_base_scale
    grad_q[i] = grad_output * (-p_i / q_i) * log_base_scale;
  }
}

/**
 * Compute second-order backward pass for cross-entropy.
 *
 * Second-order derivatives:
 *   d²H/dp_i² = 0
 *   d²H/dq_i² = p_i / q_i² * log_base_scale
 *   d²H/dp_i dq_i = -1/q_i * log_base_scale
 *
 * @param gg_p Upstream gradient w.r.t. grad_p
 * @param gg_q Upstream gradient w.r.t. grad_q
 * @param grad_output Original upstream gradient
 * @param p Pointer to true/target probability distribution
 * @param q Pointer to predicted probability distribution
 * @param n Size of distributions
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_grad_output Output: gradient w.r.t. grad_output
 * @param grad_p Output: gradient w.r.t. p
 * @param grad_q Output: gradient w.r.t. q
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void cross_entropy_backward_backward_kernel(
    const T* gg_p,
    const T* gg_q,
    T grad_output,
    const T* p,
    const T* q,
    int64_t n,
    T log_base_scale,
    T& grad_grad_output,
    T* grad_p,
    T* grad_q
) {
  T eps = get_eps<T>();
  grad_grad_output = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i];
    T q_i = q[i] > eps ? q[i] : eps;
    T gg_p_i = gg_p ? gg_p[i] : T(0);
    T gg_q_i = gg_q ? gg_q[i] : T(0);

    // d(grad_p_i)/d(grad_output) = -log(q_i) * log_base_scale
    // d(grad_q_i)/d(grad_output) = -p_i / q_i * log_base_scale
    grad_grad_output += gg_p_i * (-std::log(q_i) * log_base_scale)
                      + gg_q_i * (-p_i / q_i * log_base_scale);

    // d(grad_p_i)/dp_i = 0
    // d(grad_q_i)/dp_i = -grad_output / q_i * log_base_scale
    grad_p[i] = gg_q_i * (-grad_output / q_i * log_base_scale);

    // d(grad_p_i)/dq_i = -grad_output / q_i * log_base_scale
    // d(grad_q_i)/dq_i = grad_output * p_i / q_i² * log_base_scale
    grad_q[i] = gg_p_i * (-grad_output / q_i * log_base_scale)
              + gg_q_i * (grad_output * p_i / (q_i * q_i) * log_base_scale);
  }
}

}  // namespace torchscience::kernel::information_theory
