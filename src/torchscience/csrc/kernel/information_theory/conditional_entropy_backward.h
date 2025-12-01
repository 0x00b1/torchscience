#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of conditional entropy H(Y|X) w.r.t. joint distribution.
 *
 * H(Y|X) = -sum_{x,y} p(x,y) * log(p(x,y) / p(x))
 *        = -sum_{x,y} p(x,y) * [log(p(x,y)) - log(p(x))]
 *
 * dH/dp(x,y) = -[log(p(x,y)) - log(p(x)) + 1] + 1/p(x) * sum_y' p(x,y')
 *            = -log(p(x,y)) + log(p(x)) - 1 + 1 = -log(p(y|x))
 *
 * But we need to account for the implicit constraint sum p(x,y) = 1.
 * Actually the derivative is:
 * dH/dp(x,y) = -[log(p(x,y)/p(x)) + 1 - p(x,y)/(p(x))]
 *            = -log(p(y|x)) - 1 + p(y|x)
 *
 * @param grad_output Upstream gradient (scalar)
 * @param joint Pointer to joint probability distribution
 * @param size_x Number of outcomes for X
 * @param size_y Number of outcomes for Y
 * @param condition_dim 0 if X is rows, 1 if X is cols
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_joint Output: gradient w.r.t. joint
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void conditional_entropy_backward_kernel(
    T grad_output,
    const T* joint,
    int64_t size_x,
    int64_t size_y,
    int64_t condition_dim,
    T log_base_scale,
    T* grad_joint
) {
  T eps = get_eps<T>();
  T scaled_grad = grad_output * log_base_scale;

  if (condition_dim == 0) {
    // Condition on X (rows)
    for (int64_t x = 0; x < size_x; ++x) {
      // Compute marginal p(x)
      T p_x = T(0);
      for (int64_t y = 0; y < size_y; ++y) {
        p_x += joint[x * size_y + y];
      }
      p_x = p_x > eps ? p_x : eps;

      for (int64_t y = 0; y < size_y; ++y) {
        T p_xy = joint[x * size_y + y] > eps ? joint[x * size_y + y] : eps;
        T p_y_given_x = p_xy / p_x;

        // dH/dp(x,y) = -log(p(y|x)) - 1 + p(y|x)
        // But simpler: using chain rule H(Y|X) = H(X,Y) - H(X)
        // dH(Y|X)/dp(x,y) = dH(X,Y)/dp(x,y) - dH(X)/dp(x,y)
        //                 = -(log(p(x,y)) + 1) - [-(log(p(x)) + 1) * (1/1)]
        // Actually need to be more careful...

        // Direct computation:
        // H(Y|X) = -sum p(x,y) log(p(x,y)/p(x))
        // dH/dp(x,y) = -[log(p(x,y)/p(x)) + 1 - sum_y' p(x,y')/p(x)]
        //            = -log(p(y|x)) - 1 + 1 = -log(p(y|x))
        grad_joint[x * size_y + y] = -scaled_grad * std::log(p_y_given_x);
      }
    }
  } else {
    // Condition on Y (cols): H(X|Y)
    for (int64_t y = 0; y < size_y; ++y) {
      T p_y = T(0);
      for (int64_t x = 0; x < size_x; ++x) {
        p_y += joint[x * size_y + y];
      }
      p_y = p_y > eps ? p_y : eps;

      for (int64_t x = 0; x < size_x; ++x) {
        T p_xy = joint[x * size_y + y] > eps ? joint[x * size_y + y] : eps;
        T p_x_given_y = p_xy / p_y;

        grad_joint[x * size_y + y] = -scaled_grad * std::log(p_x_given_y);
      }
    }
  }
}

/**
 * Compute second-order gradient of conditional entropy.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void conditional_entropy_backward_backward_kernel(
    const T* gg_joint,
    T grad_output,
    const T* joint,
    int64_t size_x,
    int64_t size_y,
    int64_t condition_dim,
    T log_base_scale,
    T& grad_grad_output,
    T* grad_joint
) {
  T eps = get_eps<T>();

  grad_grad_output = T(0);

  if (condition_dim == 0) {
    for (int64_t x = 0; x < size_x; ++x) {
      T p_x = T(0);
      for (int64_t y = 0; y < size_y; ++y) {
        p_x += joint[x * size_y + y];
      }
      p_x = p_x > eps ? p_x : eps;

      for (int64_t y = 0; y < size_y; ++y) {
        T p_xy = joint[x * size_y + y] > eps ? joint[x * size_y + y] : eps;
        T p_y_given_x = p_xy / p_x;
        T gg = gg_joint ? gg_joint[x * size_y + y] : T(0);

        // d/dp(x,y) of (-log(p(y|x))) = -1/p(y|x) * d(p(y|x))/dp(x,y)
        // d(p(y|x))/dp(x,y) = 1/p(x) - p(x,y)/p(x)^2 = (p(x) - p(x,y))/p(x)^2
        //                   = (1 - p(y|x))/p(x)
        // So d(-log(p(y|x)))/dp(x,y) = -1/p(y|x) * (1 - p(y|x))/p(x)
        //                            = -(1 - p(y|x))/(p(y|x) * p(x))
        //                            = -(1/p(x,y) - 1/p(x))

        grad_grad_output += gg * (-std::log(p_y_given_x) * log_base_scale);

        T d2 = -(T(1) / p_xy - T(1) / p_x);
        grad_joint[x * size_y + y] = grad_output * log_base_scale * gg * d2;
      }
    }
  } else {
    for (int64_t y = 0; y < size_y; ++y) {
      T p_y = T(0);
      for (int64_t x = 0; x < size_x; ++x) {
        p_y += joint[x * size_y + y];
      }
      p_y = p_y > eps ? p_y : eps;

      for (int64_t x = 0; x < size_x; ++x) {
        T p_xy = joint[x * size_y + y] > eps ? joint[x * size_y + y] : eps;
        T p_x_given_y = p_xy / p_y;
        T gg = gg_joint ? gg_joint[x * size_y + y] : T(0);

        grad_grad_output += gg * (-std::log(p_x_given_y) * log_base_scale);

        T d2 = -(T(1) / p_xy - T(1) / p_y);
        grad_joint[x * size_y + y] = grad_output * log_base_scale * gg * d2;
      }
    }
  }
}

}  // namespace torchscience::kernel::information_theory
