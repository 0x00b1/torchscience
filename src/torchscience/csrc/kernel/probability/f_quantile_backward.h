#pragma once

#include <cmath>
#include <tuple>
#include "f_quantile.h"
#include "f_probability_density.h"

namespace torchscience::kernel::probability {

// Gradient of f_quantile with respect to p:
// By implicit differentiation: d(ppf)/dp = 1 / pdf(ppf(p))
template <typename T>
T f_quantile_grad_p(T p, T d1, T d2) {
  if (p <= T(0) || p >= T(1)) return T(0);
  T x = f_quantile(p, d1, d2);
  T pdf = f_probability_density(x, d1, d2);
  if (pdf < T(1e-15)) return T(0);
  return T(1) / pdf;
}

// Gradient of f_quantile with respect to d1:
template <typename T>
T f_quantile_grad_d1(T p, T d1, T d2) {
  if (p <= T(0) || p >= T(1)) return T(0);
  T eps = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), std::abs(d1));
  T ppf_plus = f_quantile(p, d1 + eps, d2);
  T ppf_minus = f_quantile(p, d1 - eps, d2);
  return (ppf_plus - ppf_minus) / (T(2) * eps);
}

// Gradient of f_quantile with respect to d2:
template <typename T>
T f_quantile_grad_d2(T p, T d1, T d2) {
  if (p <= T(0) || p >= T(1)) return T(0);
  T eps = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), std::abs(d2));
  T ppf_plus = f_quantile(p, d1, d2 + eps);
  T ppf_minus = f_quantile(p, d1, d2 - eps);
  return (ppf_plus - ppf_minus) / (T(2) * eps);
}

template <typename T>
std::tuple<T, T, T> f_quantile_backward(T grad_output, T p, T d1, T d2) {
  return {
    grad_output * f_quantile_grad_p(p, d1, d2),
    grad_output * f_quantile_grad_d1(p, d1, d2),
    grad_output * f_quantile_grad_d2(p, d1, d2)
  };
}

}  // namespace torchscience::kernel::probability
