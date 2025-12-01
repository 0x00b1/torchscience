#pragma once

#include <cmath>
#include <tuple>

#include "../special_functions/digamma.h"
#include "../special_functions/log_beta.h"
#include "beta_probability_density.h"

namespace torchscience::kernel::probability {

// Beta PDF gradient
// f(x; a, b) = x^(a-1) * (1-x)^(b-1) / B(a, b)
//
// df/dx = f * [(a-1)/x - (b-1)/(1-x)]
// df/da = f * [log(x) - (psi(a) - psi(a+b))]
// df/db = f * [log(1-x) - (psi(b) - psi(a+b))]
template <typename T>
std::tuple<T, T, T> beta_probability_density_backward(T gradient, T x, T a, T b) {
  if (x <= T(0) || x >= T(1)) {
    return {T(0), T(0), T(0)};
  }

  T pdf = beta_probability_density(x, a, b);

  // df/dx
  T df_dx = pdf * ((a - T(1)) / x - (b - T(1)) / (T(1) - x));

  // df/da and df/db
  T psi_a = special_functions::digamma(a);
  T psi_b = special_functions::digamma(b);
  T psi_ab = special_functions::digamma(a + b);

  T df_da = pdf * (std::log(x) - (psi_a - psi_ab));
  T df_db = pdf * (std::log(T(1) - x) - (psi_b - psi_ab));

  return {
    gradient * df_dx,
    gradient * df_da,
    gradient * df_db
  };
}

}  // namespace torchscience::kernel::probability
