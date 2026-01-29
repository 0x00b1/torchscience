#pragma once

#include <cmath>
#include <tuple>
#include "f_survival.h"
#include "f_probability_density.h"

namespace torchscience::kernel::probability {

// Gradient of f_survival with respect to f:
// d(SF)/df = -pdf(f; d1, d2)
template <typename T>
T f_survival_grad_f(T f, T d1, T d2) {
  if (f <= T(0)) return T(0);
  return -f_probability_density(f, d1, d2);
}

// Gradient of f_survival with respect to d1:
template <typename T>
T f_survival_grad_d1(T f, T d1, T d2) {
  if (f <= T(0)) return T(0);
  T eps = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), static_cast<T>(std::abs(d1)));
  T sf_plus = f_survival(f, d1 + eps, d2);
  T sf_minus = f_survival(f, d1 - eps, d2);
  return (sf_plus - sf_minus) / (T(2) * eps);
}

// Gradient of f_survival with respect to d2:
template <typename T>
T f_survival_grad_d2(T f, T d1, T d2) {
  if (f <= T(0)) return T(0);
  T eps = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), static_cast<T>(std::abs(d2)));
  T sf_plus = f_survival(f, d1, d2 + eps);
  T sf_minus = f_survival(f, d1, d2 - eps);
  return (sf_plus - sf_minus) / (T(2) * eps);
}

template <typename T>
std::tuple<T, T, T> f_survival_backward(T grad_output, T f, T d1, T d2) {
  return {
    grad_output * f_survival_grad_f(f, d1, d2),
    grad_output * f_survival_grad_d1(f, d1, d2),
    grad_output * f_survival_grad_d2(f, d1, d2)
  };
}

}  // namespace torchscience::kernel::probability
