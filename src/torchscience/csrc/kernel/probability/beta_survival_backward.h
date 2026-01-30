#pragma once

#include <cmath>
#include <tuple>

#include "beta_probability_density.h"
#include "../special_functions/incomplete_beta_backward.h"

namespace torchscience::kernel::probability {

// Beta survival function gradient
// S(x; a, b) = I_{1-x}(b, a)
//
// dS/dx = -pdf(x; a, b)  [since S = 1 - CDF and dCDF/dx = pdf]
// dS/da = -d(CDF)/da
// dS/db = -d(CDF)/db
//
// Using the incomplete beta backward with transformed arguments:
// Let y = 1-x, p = b, q = a
// S = I_y(p, q)
// dS/dy = pdf at y with params (p, q) = pdf(y; b, a)
// dS/dp = grad w.r.t. first param = grad_a_eff from incomplete_beta_backward(gradient, y, p, q)
// dS/dq = grad w.r.t. second param = grad_b_eff from incomplete_beta_backward(gradient, y, p, q)
//
// Then chain rule:
// dS/dx = dS/dy * dy/dx = -dS/dy = -pdf(y; b, a) = -pdf(1-x; b, a)
// But we also know dS/dx = -pdf(x; a, b), which is the same due to symmetry of the beta density
// dS/da = dS/dq (since a maps to q in the transformed call)
// dS/db = dS/dp (since b maps to p in the transformed call)
template <typename T>
std::tuple<T, T, T> beta_survival_backward(T gradient, T x, T a, T b) {
  if (x <= T(0)) {
    return {T(0), T(0), T(0)};
  }

  if (x >= T(1)) {
    return {T(0), T(0), T(0)};
  }

  // Compute y = 1 - x for the transformed incomplete beta
  T y = T(1) - x;

  // Get gradients of I_y(b, a) with respect to y, b (first param), and a (second param)
  // incomplete_beta_backward returns (grad_x, grad_a, grad_b) for incomplete_beta(x, a, b)
  // So incomplete_beta_backward(gradient, y, b, a) returns (grad_y, grad_b, grad_a)
  auto [grad_y, grad_b_raw, grad_a_raw] = special_functions::incomplete_beta_backward(gradient, y, b, a);

  // dS/dx = dS/dy * dy/dx = grad_y * (-1) = -grad_y
  // But note: grad_y already equals gradient * pdf(y; b, a) which equals the pdf at that point
  // Since we want dS/dx = -pdf(x; a, b) and pdf(y; b, a) = pdf(x; a, b) due to symmetry,
  // we have dS/dx = -grad_y
  T grad_x = -grad_y;

  // dS/da: Since a is the second parameter in I_y(b, a), this is grad_a_raw
  T grad_a = grad_a_raw;

  // dS/db: Since b is the first parameter in I_y(b, a), this is grad_b_raw
  T grad_b = grad_b_raw;

  return {grad_x, grad_a, grad_b};
}

}  // namespace torchscience::kernel::probability
