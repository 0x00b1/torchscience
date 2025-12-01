#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of Shannon entropy w.r.t. input probabilities.
 *
 * H(P) = -sum_i p_i * log(p_i)
 * dH/dp_i = -(log(p_i) + 1)
 *
 * @param grad_output Upstream gradient (scalar)
 * @param p Pointer to probability distribution
 * @param n Size of distribution
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_p Output: gradient w.r.t. p
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void shannon_entropy_backward_kernel(
    T grad_output,
    const T* p,
    int64_t n,
    T log_base_scale,
    T* grad_p
) {
  T eps = get_eps<T>();
  T scaled_grad = grad_output * log_base_scale;

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;

    // dH/dp_i = -(log(p_i) + 1)
    grad_p[i] = -scaled_grad * (std::log(p_i) + T(1));
  }
}

/**
 * Compute second-order gradient of Shannon entropy.
 *
 * d²H/dp_i² = -1/p_i
 *
 * @param gg_p Upstream gradient w.r.t. grad_p
 * @param grad_output Original upstream gradient
 * @param p Pointer to probability distribution
 * @param n Size of distribution
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_grad_output Output: gradient w.r.t. grad_output
 * @param grad_p Output: second-order gradient w.r.t. p
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void shannon_entropy_backward_backward_kernel(
    const T* gg_p,
    T grad_output,
    const T* p,
    int64_t n,
    T log_base_scale,
    T& grad_grad_output,
    T* grad_p
) {
  T eps = get_eps<T>();

  grad_grad_output = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T gg_p_i = gg_p ? gg_p[i] : T(0);

    // d(grad_p_i)/d(grad_output) = -(log(p_i) + 1) * log_base_scale
    grad_grad_output += gg_p_i * (-(std::log(p_i) + T(1)) * log_base_scale);

    // d(grad_p_i)/dp_i = -grad_output * log_base_scale / p_i
    grad_p[i] = -grad_output * log_base_scale * gg_p_i / p_i;
  }
}

}  // namespace torchscience::kernel::information_theory
