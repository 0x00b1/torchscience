#pragma once

#include <cmath>
#include <tuple>
#include "f_cumulative_distribution.h"
#include "f_probability_density.h"

namespace torchscience::kernel::probability {

// Gradient of f_cumulative_distribution with respect to f:
// d(CDF)/df = pdf(f; d1, d2)
template <typename T>
T f_cumulative_distribution_grad_f(T f, T d1, T d2) {
  if (f <= T(0)) return T(0);
  return f_probability_density(f, d1, d2);
}

// Gradient of f_cumulative_distribution with respect to d1 (numerator df):
// Uses numerical differentiation
template <typename T>
T f_cumulative_distribution_grad_d1(T f, T d1, T d2) {
  if (f <= T(0)) return T(0);
  T eps = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), std::abs(d1));
  T cdf_plus = f_cumulative_distribution(f, d1 + eps, d2);
  T cdf_minus = f_cumulative_distribution(f, d1 - eps, d2);
  return (cdf_plus - cdf_minus) / (T(2) * eps);
}

// Gradient of f_cumulative_distribution with respect to d2 (denominator df):
// Uses numerical differentiation
template <typename T>
T f_cumulative_distribution_grad_d2(T f, T d1, T d2) {
  if (f <= T(0)) return T(0);
  T eps = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), std::abs(d2));
  T cdf_plus = f_cumulative_distribution(f, d1, d2 + eps);
  T cdf_minus = f_cumulative_distribution(f, d1, d2 - eps);
  return (cdf_plus - cdf_minus) / (T(2) * eps);
}

template <typename T>
std::tuple<T, T, T> f_cumulative_distribution_backward(T grad_output, T f, T d1, T d2) {
  return {
    grad_output * f_cumulative_distribution_grad_f(f, d1, d2),
    grad_output * f_cumulative_distribution_grad_d1(f, d1, d2),
    grad_output * f_cumulative_distribution_grad_d2(f, d1, d2)
  };
}

}  // namespace torchscience::kernel::probability
