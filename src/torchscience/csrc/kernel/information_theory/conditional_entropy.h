#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute conditional entropy H(Y|X) from a 2D joint probability distribution.
 *
 * H(Y|X) = -sum_{x,y} p(x,y) * log(p(y|x))
 *        = -sum_{x,y} p(x,y) * log(p(x,y) / p(x))
 *        = H(X, Y) - H(X)
 *
 * @param joint Pointer to joint probability distribution P(X,Y) (row-major, |X| x |Y|)
 * @param size_x Number of outcomes for X (conditioning variable)
 * @param size_y Number of outcomes for Y (target variable)
 * @param condition_dim 0 if X is rows (condition on rows), 1 if X is cols
 * @param log_base_scale Scale factor for log base conversion (1.0 for nats)
 * @return Conditional entropy value H(Y|X)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T conditional_entropy_kernel(
    const T* joint,
    int64_t size_x,
    int64_t size_y,
    int64_t condition_dim,
    T log_base_scale
) {
  T eps = get_eps<T>();
  T result = T(0);

  if (condition_dim == 0) {
    // Condition on X (rows): H(Y|X) where X indexes rows
    // p(x) = sum_y p(x,y), then p(y|x) = p(x,y) / p(x)
    for (int64_t x = 0; x < size_x; ++x) {
      // Compute marginal p(x)
      T p_x = T(0);
      for (int64_t y = 0; y < size_y; ++y) {
        p_x += joint[x * size_y + y];
      }
      p_x = p_x > eps ? p_x : eps;

      // Compute contribution to conditional entropy
      for (int64_t y = 0; y < size_y; ++y) {
        T p_xy = joint[x * size_y + y];
        if (p_xy > eps) {
          T p_y_given_x = p_xy / p_x;
          result -= p_xy * std::log(p_y_given_x);
        }
      }
    }
  } else {
    // Condition on Y (cols): H(X|Y) where Y indexes cols
    // p(y) = sum_x p(x,y), then p(x|y) = p(x,y) / p(y)
    for (int64_t y = 0; y < size_y; ++y) {
      // Compute marginal p(y)
      T p_y = T(0);
      for (int64_t x = 0; x < size_x; ++x) {
        p_y += joint[x * size_y + y];
      }
      p_y = p_y > eps ? p_y : eps;

      // Compute contribution to conditional entropy
      for (int64_t x = 0; x < size_x; ++x) {
        T p_xy = joint[x * size_y + y];
        if (p_xy > eps) {
          T p_x_given_y = p_xy / p_y;
          result -= p_xy * std::log(p_x_given_y);
        }
      }
    }
  }

  return result * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
