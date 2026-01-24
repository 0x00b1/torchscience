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
 *
 * The first gradient is: g(x,y) = -log(p(y|x)) * scale
 *
 * Second-order derivatives:
 * - d²H/dp(x,y)² = (-1/p(x,y) + 1/p(x)) * scale
 * - d²H/dp(x,y)dp(x,y') = 1/p(x) * scale  for y' != y (same row cross-term)
 * - d²H/dp(x,y)dp(x',y') = 0  for x' != x (different rows are independent)
 *
 * The backward_backward computes:
 * - grad_grad_output = sum_{x,y} gg_joint(x,y) * dH/dp(x,y)
 * - grad_joint(x,y) = sum_{x',y'} gg_joint(x',y') * grad_output * d²H/dp(x',y')dp(x,y)
 *                   = grad_output * [gg_joint(x,y) * (-1/p(x,y) + 1/p(x))
 *                                  + sum_{y'!=y} gg_joint(x,y') * 1/p(x)]
 *                   = grad_output * [gg_joint(x,y) * (-1/p(x,y))
 *                                  + sum_{y'} gg_joint(x,y') * 1/p(x)]
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
  T scaled_grad = grad_output * log_base_scale;

  grad_grad_output = T(0);

  if (condition_dim == 0) {
    // Condition on X (rows): H(Y|X)
    for (int64_t x = 0; x < size_x; ++x) {
      // Compute marginal p(x) and sum of gg for this row
      T p_x = T(0);
      T gg_row_sum = T(0);
      for (int64_t y = 0; y < size_y; ++y) {
        p_x += joint[x * size_y + y];
        if (gg_joint) {
          gg_row_sum += gg_joint[x * size_y + y];
        }
      }
      p_x = p_x > eps ? p_x : eps;

      // Cross-term contribution: (1/p(x)) * sum_y' gg(x,y')
      T cross_term = gg_row_sum / p_x;

      for (int64_t y = 0; y < size_y; ++y) {
        T p_xy = joint[x * size_y + y] > eps ? joint[x * size_y + y] : eps;
        T p_y_given_x = p_xy / p_x;
        T gg = gg_joint ? gg_joint[x * size_y + y] : T(0);

        // grad_grad_output contribution
        grad_grad_output += gg * (-std::log(p_y_given_x) * log_base_scale);

        // grad_joint: diagonal term (-1/p(x,y)) + cross term (1/p(x) from all y' in same row)
        // = gg * (-1/p(x,y)) + cross_term
        // But we scale by grad_output and log_base_scale
        grad_joint[x * size_y + y] = scaled_grad * (-gg / p_xy + cross_term);
      }
    }
  } else {
    // Condition on Y (cols): H(X|Y)
    for (int64_t y = 0; y < size_y; ++y) {
      // Compute marginal p(y) and sum of gg for this column
      T p_y = T(0);
      T gg_col_sum = T(0);
      for (int64_t x = 0; x < size_x; ++x) {
        p_y += joint[x * size_y + y];
        if (gg_joint) {
          gg_col_sum += gg_joint[x * size_y + y];
        }
      }
      p_y = p_y > eps ? p_y : eps;

      // Cross-term contribution: (1/p(y)) * sum_x' gg(x',y)
      T cross_term = gg_col_sum / p_y;

      for (int64_t x = 0; x < size_x; ++x) {
        T p_xy = joint[x * size_y + y] > eps ? joint[x * size_y + y] : eps;
        T p_x_given_y = p_xy / p_y;
        T gg = gg_joint ? gg_joint[x * size_y + y] : T(0);

        // grad_grad_output contribution
        grad_grad_output += gg * (-std::log(p_x_given_y) * log_base_scale);

        // grad_joint: diagonal term + cross term
        grad_joint[x * size_y + y] = scaled_grad * (-gg / p_xy + cross_term);
      }
    }
  }
}

}  // namespace torchscience::kernel::information_theory
