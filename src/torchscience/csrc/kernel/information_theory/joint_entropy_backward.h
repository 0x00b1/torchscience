#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of joint entropy w.r.t. joint distribution.
 *
 * H(X,Y) = -sum_{x,y} p(x,y) * log(p(x,y))
 * dH/dp(x,y) = -(log(p(x,y)) + 1)
 *
 * @param grad_output Upstream gradient (scalar)
 * @param joint Pointer to joint probability distribution
 * @param size_x Number of outcomes for X
 * @param size_y Number of outcomes for Y
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_joint Output: gradient w.r.t. joint
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void joint_entropy_backward_kernel(
    T grad_output,
    const T* joint,
    int64_t size_x,
    int64_t size_y,
    T log_base_scale,
    T* grad_joint
) {
  T eps = get_eps<T>();
  T scaled_grad = grad_output * log_base_scale;
  int64_t total = size_x * size_y;

  for (int64_t i = 0; i < total; ++i) {
    T p_xy = joint[i] > eps ? joint[i] : eps;
    grad_joint[i] = -scaled_grad * (std::log(p_xy) + T(1));
  }
}

/**
 * Compute second-order gradient of joint entropy.
 *
 * d²H/dp(x,y)² = -1/p(x,y)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void joint_entropy_backward_backward_kernel(
    const T* gg_joint,
    T grad_output,
    const T* joint,
    int64_t size_x,
    int64_t size_y,
    T log_base_scale,
    T& grad_grad_output,
    T* grad_joint
) {
  T eps = get_eps<T>();
  int64_t total = size_x * size_y;

  grad_grad_output = T(0);

  for (int64_t i = 0; i < total; ++i) {
    T p_xy = joint[i] > eps ? joint[i] : eps;
    T gg = gg_joint ? gg_joint[i] : T(0);

    grad_grad_output += gg * (-(std::log(p_xy) + T(1)) * log_base_scale);
    grad_joint[i] = -grad_output * log_base_scale * gg / p_xy;
  }
}

}  // namespace torchscience::kernel::information_theory
