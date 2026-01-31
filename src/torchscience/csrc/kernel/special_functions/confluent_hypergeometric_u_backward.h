#pragma once

#include <cmath>
#include <tuple>

#include "confluent_hypergeometric_u.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> confluent_hypergeometric_u_backward(T grad, T a, T b, T z) {
  using detail::hypu_epsilon;
  using detail::hypu_is_complex_v;
  using detail::hypu_real_type_t;

  // d/dz U(a, b, z) = -a * U(a+1, b+1, z)
  T dfdz = -a * confluent_hypergeometric_u(a + T(1), b + T(1), z);

  // For parameter derivatives (a and b), use finite differences
  // These are complex to derive analytically due to the gamma function ratios
  using real_t = hypu_real_type_t<T>;
  real_t eps_real = std::sqrt(hypu_epsilon<T>());
  T eps = T(eps_real);

  T f_a_plus = confluent_hypergeometric_u(a + eps, b, z);
  T f_a_minus = confluent_hypergeometric_u(a - eps, b, z);
  T dfda = (f_a_plus - f_a_minus) / (T(2) * eps);

  T f_b_plus = confluent_hypergeometric_u(a, b + eps, z);
  T f_b_minus = confluent_hypergeometric_u(a, b - eps, z);
  T dfdb = (f_b_plus - f_b_minus) / (T(2) * eps);

  // For complex types, PyTorch expects grad * conj(derivative)
  if constexpr (hypu_is_complex_v<T>) {
    return {
      grad * std::conj(dfda),
      grad * std::conj(dfdb),
      grad * std::conj(dfdz)
    };
  } else {
    return {grad * dfda, grad * dfdb, grad * dfdz};
  }
}

} // namespace torchscience::kernel::special_functions
