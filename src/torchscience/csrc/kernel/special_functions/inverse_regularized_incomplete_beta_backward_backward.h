#pragma once

#include <cmath>
#include <tuple>
#include <limits>

#include "inverse_regularized_incomplete_beta.h"
#include "incomplete_beta.h"
#include "log_beta.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T, T> inverse_regularized_incomplete_beta_backward_backward(
    T gg_a, T gg_b, T gg_y, T grad_output, T a, T b, T y) {
  // Second-order derivatives for inverse_regularized_incomplete_beta
  //
  // Let x = I^{-1}(a, b, y) where I(x, a, b) = y
  //
  // Using implicit differentiation, we compute second derivatives numerically
  // since the analytical formulas are complex and numerically unstable.

  // Handle edge cases
  if (y <= T(0) || y >= T(1) || a <= T(0) || b <= T(0)) {
    return std::make_tuple(T(0), T(0), T(0), T(0));
  }

  // Compute x = I^{-1}(a, b, y)
  T x = inverse_regularized_incomplete_beta(a, b, y);

  if (x <= T(0) || x >= T(1) || !std::isfinite(x)) {
    return std::make_tuple(T(0), T(0), T(0), T(0));
  }

  // Compute log B(a, b)
  T log_beta_ab = log_beta(a, b);

  // Compute dI/dx = x^(a-1) * (1-x)^(b-1) / B(a, b)
  T log_dIdx = (a - T(1)) * std::log(x) + (b - T(1)) * std::log(T(1) - x) - log_beta_ab;
  T dIdx = std::exp(log_dIdx);

  if (dIdx < std::numeric_limits<T>::min() * T(1e10)) {
    return std::make_tuple(T(0), T(0), T(0), T(0));
  }

  // First derivatives
  T dxdy = T(1) / dIdx;

  // Numerical differentiation for dI/da, dI/db
  T eps_a = std::sqrt(std::numeric_limits<T>::epsilon()) *
            std::max(T(1), std::abs(a));
  T eps_b = std::sqrt(std::numeric_limits<T>::epsilon()) *
            std::max(T(1), std::abs(b));

  T I_a_plus = incomplete_beta(x, a + eps_a, b);
  T I_a_minus = incomplete_beta(x, a - eps_a, b);
  T dIda = (I_a_plus - I_a_minus) / (T(2) * eps_a);

  T I_b_plus = incomplete_beta(x, a, b + eps_b);
  T I_b_minus = incomplete_beta(x, a, b - eps_b);
  T dIdb = (I_b_plus - I_b_minus) / (T(2) * eps_b);

  T dxda = -dIda / dIdx;
  T dxdb = -dIdb / dIdx;

  // Second derivative d^2x/dy^2:
  // From dx/dy = 1/dIdx, we get:
  // d(dx/dy)/dy = d(1/dIdx)/dy = -dIdx' * (dx/dy) / dIdx^2
  // where dIdx' = d(dI/dx)/dx = d^2I/dx^2
  // d^2I/dx^2 = [(a-1)/x - (b-1)/(1-x)] * dI/dx
  T d2Idx2 = ((a - T(1)) / x - (b - T(1)) / (T(1) - x)) * dIdx;
  T d2xdy2 = -d2Idx2 * dxdy / (dIdx * dIdx);

  // For cross and second derivatives w.r.t. a, b, use numerical differentiation
  // of the first derivatives
  T eps_y = std::sqrt(std::numeric_limits<T>::epsilon()) *
            std::max(T(1e-6), std::min(y, T(1) - y));

  // Compute dx/da at perturbed a values for d^2x/da^2
  T x_ap = inverse_regularized_incomplete_beta(a + eps_a, b, y);
  T x_am = inverse_regularized_incomplete_beta(a - eps_a, b, y);

  T dxda_ap = T(0), dxda_am = T(0);
  T dxdb_ap = T(0), dxdb_am = T(0);

  // Compute dxda at a + eps_a
  if (x_ap > T(0) && x_ap < T(1)) {
    T log_dIdx_ap = (a + eps_a - T(1)) * std::log(x_ap) + (b - T(1)) * std::log(T(1) - x_ap) - log_beta(a + eps_a, b);
    T dIdx_ap = std::exp(log_dIdx_ap);
    if (dIdx_ap > std::numeric_limits<T>::min()) {
      T I_plus = incomplete_beta(x_ap, a + T(2) * eps_a, b);
      T I_minus = incomplete_beta(x_ap, a, b);
      T dIda_ap = (I_plus - I_minus) / (T(2) * eps_a);
      dxda_ap = -dIda_ap / dIdx_ap;
    }
  }

  // Compute dxda at a - eps_a
  if (x_am > T(0) && x_am < T(1)) {
    T log_dIdx_am = (a - eps_a - T(1)) * std::log(x_am) + (b - T(1)) * std::log(T(1) - x_am) - log_beta(a - eps_a, b);
    T dIdx_am = std::exp(log_dIdx_am);
    if (dIdx_am > std::numeric_limits<T>::min()) {
      T I_plus = incomplete_beta(x_am, a, b);
      T I_minus = incomplete_beta(x_am, a - T(2) * eps_a, b);
      T dIda_am = (I_plus - I_minus) / (T(2) * eps_a);
      dxda_am = -dIda_am / dIdx_am;
    }
  }

  T d2xda2 = (dxda_ap - T(2) * dxda + dxda_am) / (eps_a * eps_a);

  // Compute dx/db at perturbed b values for d^2x/db^2
  T x_bp = inverse_regularized_incomplete_beta(a, b + eps_b, y);
  T x_bm = inverse_regularized_incomplete_beta(a, b - eps_b, y);

  T dxdb_bp = T(0), dxdb_bm = T(0);

  if (x_bp > T(0) && x_bp < T(1)) {
    T log_dIdx_bp = (a - T(1)) * std::log(x_bp) + (b + eps_b - T(1)) * std::log(T(1) - x_bp) - log_beta(a, b + eps_b);
    T dIdx_bp = std::exp(log_dIdx_bp);
    if (dIdx_bp > std::numeric_limits<T>::min()) {
      T I_plus = incomplete_beta(x_bp, a, b + T(2) * eps_b);
      T I_minus = incomplete_beta(x_bp, a, b);
      T dIdb_bp = (I_plus - I_minus) / (T(2) * eps_b);
      dxdb_bp = -dIdb_bp / dIdx_bp;
    }
  }

  if (x_bm > T(0) && x_bm < T(1)) {
    T log_dIdx_bm = (a - T(1)) * std::log(x_bm) + (b - eps_b - T(1)) * std::log(T(1) - x_bm) - log_beta(a, b - eps_b);
    T dIdx_bm = std::exp(log_dIdx_bm);
    if (dIdx_bm > std::numeric_limits<T>::min()) {
      T I_plus = incomplete_beta(x_bm, a, b);
      T I_minus = incomplete_beta(x_bm, a, b - T(2) * eps_b);
      T dIdb_bm = (I_plus - I_minus) / (T(2) * eps_b);
      dxdb_bm = -dIdb_bm / dIdx_bm;
    }
  }

  T d2xdb2 = (dxdb_bp - T(2) * dxdb + dxdb_bm) / (eps_b * eps_b);

  // Cross derivatives d^2x/dady, d^2x/dbdy
  T y_plus = std::min(y + eps_y, T(1) - std::numeric_limits<T>::epsilon() * T(10));
  T y_minus = std::max(y - eps_y, std::numeric_limits<T>::epsilon() * T(10));

  T x_yp = inverse_regularized_incomplete_beta(a, b, y_plus);
  T x_ym = inverse_regularized_incomplete_beta(a, b, y_minus);

  T dxda_yp = T(0), dxda_ym = T(0);
  T dxdb_yp = T(0), dxdb_ym = T(0);

  if (x_yp > T(0) && x_yp < T(1)) {
    T log_dIdx_yp = (a - T(1)) * std::log(x_yp) + (b - T(1)) * std::log(T(1) - x_yp) - log_beta_ab;
    T dIdx_yp = std::exp(log_dIdx_yp);
    if (dIdx_yp > std::numeric_limits<T>::min()) {
      T I_plus = incomplete_beta(x_yp, a + eps_a, b);
      T I_minus = incomplete_beta(x_yp, a - eps_a, b);
      T dIda_yp = (I_plus - I_minus) / (T(2) * eps_a);
      dxda_yp = -dIda_yp / dIdx_yp;

      I_plus = incomplete_beta(x_yp, a, b + eps_b);
      I_minus = incomplete_beta(x_yp, a, b - eps_b);
      T dIdb_yp = (I_plus - I_minus) / (T(2) * eps_b);
      dxdb_yp = -dIdb_yp / dIdx_yp;
    }
  }

  if (x_ym > T(0) && x_ym < T(1)) {
    T log_dIdx_ym = (a - T(1)) * std::log(x_ym) + (b - T(1)) * std::log(T(1) - x_ym) - log_beta_ab;
    T dIdx_ym = std::exp(log_dIdx_ym);
    if (dIdx_ym > std::numeric_limits<T>::min()) {
      T I_plus = incomplete_beta(x_ym, a + eps_a, b);
      T I_minus = incomplete_beta(x_ym, a - eps_a, b);
      T dIda_ym = (I_plus - I_minus) / (T(2) * eps_a);
      dxda_ym = -dIda_ym / dIdx_ym;

      I_plus = incomplete_beta(x_ym, a, b + eps_b);
      I_minus = incomplete_beta(x_ym, a, b - eps_b);
      T dIdb_ym = (I_plus - I_minus) / (T(2) * eps_b);
      dxdb_ym = -dIdb_ym / dIdx_ym;
    }
  }

  T d2xdady = (dxda_yp - dxda_ym) / (y_plus - y_minus);
  T d2xdbdy = (dxdb_yp - dxdb_ym) / (y_plus - y_minus);

  // Cross derivative d^2x/dadb
  T dxda_bp = T(0), dxda_bm = T(0);

  if (x_bp > T(0) && x_bp < T(1)) {
    T log_dIdx_bp = (a - T(1)) * std::log(x_bp) + (b + eps_b - T(1)) * std::log(T(1) - x_bp) - log_beta(a, b + eps_b);
    T dIdx_bp = std::exp(log_dIdx_bp);
    if (dIdx_bp > std::numeric_limits<T>::min()) {
      T I_plus = incomplete_beta(x_bp, a + eps_a, b + eps_b);
      T I_minus = incomplete_beta(x_bp, a - eps_a, b + eps_b);
      T dIda_bp = (I_plus - I_minus) / (T(2) * eps_a);
      dxda_bp = -dIda_bp / dIdx_bp;
    }
  }

  if (x_bm > T(0) && x_bm < T(1)) {
    T log_dIdx_bm = (a - T(1)) * std::log(x_bm) + (b - eps_b - T(1)) * std::log(T(1) - x_bm) - log_beta(a, b - eps_b);
    T dIdx_bm = std::exp(log_dIdx_bm);
    if (dIdx_bm > std::numeric_limits<T>::min()) {
      T I_plus = incomplete_beta(x_bm, a + eps_a, b - eps_b);
      T I_minus = incomplete_beta(x_bm, a - eps_a, b - eps_b);
      T dIda_bm = (I_plus - I_minus) / (T(2) * eps_a);
      dxda_bm = -dIda_bm / dIdx_bm;
    }
  }

  T d2xdadb = (dxda_bp - dxda_bm) / (T(2) * eps_b);

  // Bound the second derivatives to avoid numerical instability
  T max_deriv = T(1e6);
  d2xdy2 = std::max(-max_deriv, std::min(max_deriv, d2xdy2));
  d2xda2 = std::max(-max_deriv, std::min(max_deriv, d2xda2));
  d2xdb2 = std::max(-max_deriv, std::min(max_deriv, d2xdb2));
  d2xdady = std::max(-max_deriv, std::min(max_deriv, d2xdady));
  d2xdbdy = std::max(-max_deriv, std::min(max_deriv, d2xdbdy));
  d2xdadb = std::max(-max_deriv, std::min(max_deriv, d2xdadb));

  // Backward_backward computation
  // grad_grad_output = gg_a * dx/da + gg_b * dx/db + gg_y * dx/dy
  T grad_grad_output = gg_a * dxda + gg_b * dxdb + gg_y * dxdy;

  // grad_a = grad_output * (gg_a * d^2x/da^2 + gg_b * d^2x/dadb + gg_y * d^2x/dady)
  T grad_a_out = grad_output * (gg_a * d2xda2 + gg_b * d2xdadb + gg_y * d2xdady);

  // grad_b = grad_output * (gg_a * d^2x/dadb + gg_b * d^2x/db^2 + gg_y * d^2x/dbdy)
  T grad_b_out = grad_output * (gg_a * d2xdadb + gg_b * d2xdb2 + gg_y * d2xdbdy);

  // grad_y = grad_output * (gg_a * d^2x/dady + gg_b * d^2x/dbdy + gg_y * d^2x/dy^2)
  T grad_y_out = grad_output * (gg_a * d2xdady + gg_b * d2xdbdy + gg_y * d2xdy2);

  return std::make_tuple(grad_grad_output, grad_a_out, grad_b_out, grad_y_out);
}

} // namespace torchscience::kernel::special_functions
