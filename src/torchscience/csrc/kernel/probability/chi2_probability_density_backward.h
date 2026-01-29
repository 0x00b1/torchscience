#pragma once

#include <cmath>
#include <tuple>
#include "chi2_probability_density.h"

namespace torchscience::kernel::probability {

// Gradient of chi2_probability_density with respect to x:
// d/dx [x^(k/2 - 1) * exp(-x/2)] * const = pdf * [(k/2 - 1)/x - 1/2]
template <typename T>
T chi2_probability_density_grad_x(T x, T df) {
  if (x <= T(0)) return T(0);
  T pdf = chi2_probability_density(x, df);
  T k_half = df / T(2);
  return pdf * ((k_half - T(1)) / x - T(0.5));
}

// Gradient of chi2_probability_density with respect to df:
// Uses numerical differentiation
template <typename T>
T chi2_probability_density_grad_df(T x, T df) {
  if (x <= T(0)) return T(0);
  T eps = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), static_cast<T>(std::abs(df)));
  T pdf_plus = chi2_probability_density(x, df + eps);
  T pdf_minus = chi2_probability_density(x, df - eps);
  return (pdf_plus - pdf_minus) / (T(2) * eps);
}

template <typename T>
std::tuple<T, T> chi2_probability_density_backward(T grad_output, T x, T df) {
  return {
    grad_output * chi2_probability_density_grad_x(x, df),
    grad_output * chi2_probability_density_grad_df(x, df)
  };
}

}  // namespace torchscience::kernel::probability
