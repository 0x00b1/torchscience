#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

#include "inverse_regularized_gamma_p.h"
#include "regularized_gamma_p.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> inverse_regularized_gamma_p_backward_backward(
    T gg_a, T gg_y, T grad_output, T a, T y) {
  // Second-order derivatives for inverse_regularized_gamma_p
  //
  // Let x = P^{-1}(a, y) where P(a, x) = y
  //
  // Using implicit differentiation on P(a, x(a, y)) = y
  //
  // First derivatives:
  //   dx/dy = 1 / P_x  where P_x = dP/dx = x^(a-1) * e^(-x) / Gamma(a)
  //   dx/da = -P_a / P_x  where P_a = dP/da (numerical)
  //
  // For second derivatives, differentiate dx/dy and dx/da
  //
  // The second-order derivatives are complex for inverse gamma functions.
  // We use a hybrid approach: analytical where possible, numerical for stability.

  // Handle edge cases
  if (y <= T(0) || y >= T(1) || a <= T(0)) {
    return std::make_tuple(T(0), T(0), T(0));
  }

  // Compute x = P^{-1}(a, y)
  T x = inverse_regularized_gamma_p(a, y);

  if (x <= T(0) || !std::isfinite(x)) {
    return std::make_tuple(T(0), T(0), T(0));
  }

  // Compute P_x = dP/dx = x^(a-1) * e^(-x) / Gamma(a)
  T log_gamma_a = std::lgamma(a);
  T log_Px = (a - T(1)) * std::log(x) - x - log_gamma_a;
  T Px = std::exp(log_Px);

  if (Px < std::numeric_limits<T>::min() * T(1e10)) {
    // Derivative too small, return zeros
    return std::make_tuple(T(0), T(0), T(0));
  }

  // dx/dy = 1 / P_x
  T dxdy = T(1) / Px;

  // Compute P_a numerically
  T eps_a = std::sqrt(std::numeric_limits<T>::epsilon()) *
            std::max(T(1), std::abs(a));
  T P_plus = regularized_gamma_p(a + eps_a, x);
  T P_minus = regularized_gamma_p(a - eps_a, x);
  T Pa = (P_plus - P_minus) / (T(2) * eps_a);

  // dx/da = -P_a / P_x
  T dxda = -Pa / Px;

  // For second derivatives, use numerical differentiation with appropriate step sizes
  // d^2x/dy^2:
  // From dx/dy = 1/P_x, we get d(dx/dy)/dy = d(1/P_x)/dy = -P_xx * (dx/dy) / P_x^2
  // where P_xx = d^2P/dx^2 = ((a-1)/x - 1) * P_x
  T Pxx = ((a - T(1)) / x - T(1)) * Px;
  T d2xdy2 = -Pxx * dxdy / (Px * Px);

  // For cross and second derivatives w.r.t. a, use numerical differentiation
  // This is more stable than analytical formulas for complex derivatives
  T eps_y = std::sqrt(std::numeric_limits<T>::epsilon()) *
            std::max(T(1e-6), std::min(y, T(1) - y));

  // Compute dx/da at (a + eps_a, y) and (a - eps_a, y)
  T x_ap = inverse_regularized_gamma_p(a + eps_a, y);
  T x_am = inverse_regularized_gamma_p(a - eps_a, y);
  T dxda_ap = T(0), dxda_am = T(0);

  // Compute dxda at perturbed a values
  if (x_ap > T(0)) {
    T log_Px_ap = (a + eps_a - T(1)) * std::log(x_ap) - x_ap - std::lgamma(a + eps_a);
    T Px_ap = std::exp(log_Px_ap);
    if (Px_ap > std::numeric_limits<T>::min() * T(1e10)) {
      T P_plus_ap = regularized_gamma_p(a + T(2) * eps_a, x_ap);
      T P_minus_ap = regularized_gamma_p(a, x_ap);
      T Pa_ap = (P_plus_ap - P_minus_ap) / (T(2) * eps_a);
      dxda_ap = -Pa_ap / Px_ap;
    }
  }

  if (x_am > T(0)) {
    T log_Px_am = (a - eps_a - T(1)) * std::log(x_am) - x_am - std::lgamma(a - eps_a);
    T Px_am = std::exp(log_Px_am);
    if (Px_am > std::numeric_limits<T>::min() * T(1e10)) {
      T P_plus_am = regularized_gamma_p(a, x_am);
      T P_minus_am = regularized_gamma_p(a - T(2) * eps_a, x_am);
      T Pa_am = (P_plus_am - P_minus_am) / (T(2) * eps_a);
      dxda_am = -Pa_am / Px_am;
    }
  }

  // d^2x/da^2 using central difference
  T d2xda2 = (dxda_ap - T(2) * dxda + dxda_am) / (eps_a * eps_a);

  // For d^2x/dady, compute dx/da at different y values
  T y_plus = std::min(y + eps_y, T(1) - std::numeric_limits<T>::epsilon() * T(10));
  T y_minus = std::max(y - eps_y, std::numeric_limits<T>::epsilon() * T(10));

  T x_yp = inverse_regularized_gamma_p(a, y_plus);
  T x_ym = inverse_regularized_gamma_p(a, y_minus);
  T dxda_yp = T(0), dxda_ym = T(0);

  if (x_yp > T(0)) {
    T log_Px_yp = (a - T(1)) * std::log(x_yp) - x_yp - log_gamma_a;
    T Px_yp = std::exp(log_Px_yp);
    if (Px_yp > std::numeric_limits<T>::min() * T(1e10)) {
      T P_plus_yp = regularized_gamma_p(a + eps_a, x_yp);
      T P_minus_yp = regularized_gamma_p(a - eps_a, x_yp);
      T Pa_yp = (P_plus_yp - P_minus_yp) / (T(2) * eps_a);
      dxda_yp = -Pa_yp / Px_yp;
    }
  }

  if (x_ym > T(0)) {
    T log_Px_ym = (a - T(1)) * std::log(x_ym) - x_ym - log_gamma_a;
    T Px_ym = std::exp(log_Px_ym);
    if (Px_ym > std::numeric_limits<T>::min() * T(1e10)) {
      T P_plus_ym = regularized_gamma_p(a + eps_a, x_ym);
      T P_minus_ym = regularized_gamma_p(a - eps_a, x_ym);
      T Pa_ym = (P_plus_ym - P_minus_ym) / (T(2) * eps_a);
      dxda_ym = -Pa_ym / Px_ym;
    }
  }

  T d2xdady = (dxda_yp - dxda_ym) / (y_plus - y_minus);

  // Bound the second derivatives to avoid numerical instability
  T max_deriv = T(1e6);
  d2xdy2 = std::max(-max_deriv, std::min(max_deriv, d2xdy2));
  d2xda2 = std::max(-max_deriv, std::min(max_deriv, d2xda2));
  d2xdady = std::max(-max_deriv, std::min(max_deriv, d2xdady));

  // Backward_backward computation
  // grad_grad_output = gg_a * dx/da + gg_y * dx/dy
  T grad_grad_output = gg_a * dxda + gg_y * dxdy;

  // grad_a = grad_output * (gg_a * d^2x/da^2 + gg_y * d^2x/dady)
  T grad_a_out = grad_output * (gg_a * d2xda2 + gg_y * d2xdady);

  // grad_y = grad_output * (gg_a * d^2x/dady + gg_y * d^2x/dy^2)
  T grad_y_out = grad_output * (gg_a * d2xdady + gg_y * d2xdy2);

  return std::make_tuple(grad_grad_output, grad_a_out, grad_y_out);
}

} // namespace torchscience::kernel::special_functions
