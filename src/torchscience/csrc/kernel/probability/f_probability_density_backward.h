#pragma once

#include <cmath>
#include <tuple>
#include "f_probability_density.h"

namespace torchscience::kernel::probability {

// Gradient of f_probability_density with respect to x:
// Uses numerical differentiation for stability
template <typename T>
T f_probability_density_grad_x(T x, T d1, T d2) {
  if (x <= T(0)) return T(0);
  T eps = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), static_cast<T>(std::abs(x)));
  T pdf_plus = f_probability_density(x + eps, d1, d2);
  T pdf_minus = f_probability_density(x - eps, d1, d2);
  return (pdf_plus - pdf_minus) / (T(2) * eps);
}

// Gradient of f_probability_density with respect to d1:
template <typename T>
T f_probability_density_grad_d1(T x, T d1, T d2) {
  if (x <= T(0)) return T(0);
  T eps = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), static_cast<T>(std::abs(d1)));
  T pdf_plus = f_probability_density(x, d1 + eps, d2);
  T pdf_minus = f_probability_density(x, d1 - eps, d2);
  return (pdf_plus - pdf_minus) / (T(2) * eps);
}

// Gradient of f_probability_density with respect to d2:
template <typename T>
T f_probability_density_grad_d2(T x, T d1, T d2) {
  if (x <= T(0)) return T(0);
  T eps = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), static_cast<T>(std::abs(d2)));
  T pdf_plus = f_probability_density(x, d1, d2 + eps);
  T pdf_minus = f_probability_density(x, d1, d2 - eps);
  return (pdf_plus - pdf_minus) / (T(2) * eps);
}

template <typename T>
std::tuple<T, T, T> f_probability_density_backward(T grad_output, T x, T d1, T d2) {
  return {
    grad_output * f_probability_density_grad_x(x, d1, d2),
    grad_output * f_probability_density_grad_d1(x, d1, d2),
    grad_output * f_probability_density_grad_d2(x, d1, d2)
  };
}

}  // namespace torchscience::kernel::probability
