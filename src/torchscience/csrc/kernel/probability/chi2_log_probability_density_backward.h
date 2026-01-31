#pragma once

#include <cmath>
#include <tuple>

#include "../special_functions/digamma.h"

namespace torchscience::kernel::probability {

// Gradients of chi-squared log probability density function
// logpdf(x; df) = (df/2 - 1) * log(x) - x/2 - (df/2) * log(2) - log_gamma(df/2)
//
// d(logpdf)/dx = (df/2 - 1)/x - 0.5
// d(logpdf)/ddf = 0.5 * log(x) - 0.5 * log(2) - 0.5 * digamma(df/2)
template <typename T>
std::tuple<T, T> chi2_log_probability_density_backward(T gradient, T x, T df) {
  if (x <= T(0)) {
    return {T(0), T(0)};
  }

  const T log_2 = T(0.6931471805599453);  // log(2)
  T k_half = df / T(2);

  // d(logpdf)/dx = (df/2 - 1)/x - 0.5
  T grad_x = gradient * ((k_half - T(1)) / x - T(0.5));

  // d(logpdf)/ddf = 0.5 * log(x) - 0.5 * log(2) - 0.5 * digamma(df/2)
  T grad_df = gradient * T(0.5) * (std::log(x) - log_2 - special_functions::digamma(k_half));

  return {grad_x, grad_df};
}

}  // namespace torchscience::kernel::probability
