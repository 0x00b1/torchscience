#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "exponential_integral_e.h"

namespace torchscience::kernel::special_functions {

// Real backward: returns (grad_n, grad_x)
// d/dx E_n(x) = -E_{n-1}(x)
// d/dn E_n(x) = 0 (n is discrete, gradient w.r.t. n is zero)
template <typename T>
std::tuple<T, T> exponential_integral_e_backward(T grad_output, T n, T x) {
  const T eps = detail::exponential_integral_e_eps<T>();

  // grad_n is always 0 since n is a discrete parameter
  T grad_n = T(0);

  // Check if n is a valid non-negative integer
  T n_rounded = std::round(n);
  if (n < T(0) || std::abs(n - n_rounded) > eps) {
    return {T(0), std::numeric_limits<T>::quiet_NaN()};
  }

  int n_int = static_cast<int>(n_rounded);

  // grad_x: d/dx E_n(x) = -E_{n-1}(x)
  T grad_x;
  if (n_int == 0) {
    // E_0(x) = e^{-x} / x
    // d/dx E_0(x) = -e^{-x}/x - e^{-x}/x^2 = -e^{-x} * (1/x + 1/x^2) = -e^{-x} * (x+1)/x^2
    // But also: d/dx E_0(x) = -E_{-1}(x) which is undefined
    // Using the direct derivative: d/dx (e^{-x}/x) = -e^{-x}/x - e^{-x}/x^2 = -e^{-x}(x+1)/x^2
    if (x == T(0)) {
      grad_x = std::numeric_limits<T>::quiet_NaN();
    } else {
      T exp_neg_x = std::exp(-x);
      grad_x = -exp_neg_x * (x + T(1)) / (x * x);
    }
  } else {
    // d/dx E_n(x) = -E_{n-1}(x) for n >= 1
    T e_nm1 = exponential_integral_e(T(n_int - 1), x);
    grad_x = -e_nm1;
  }

  return {grad_n, grad_output * grad_x};
}

// Complex backward: returns (grad_n, grad_z)
// d/dz E_n(z) = -E_{n-1}(z)
// PyTorch convention: grad * conj(d/dz E_n(z)) for Wirtinger derivatives
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> exponential_integral_e_backward(
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
  using Complex = c10::complex<T>;
  const T eps = detail::exponential_integral_e_eps<T>();

  // grad_n is always 0 since n is discrete
  Complex grad_n(T(0), T(0));

  // Validate n is a real non-negative integer
  if (std::abs(n.imag()) > eps) {
    return {grad_n, Complex(std::numeric_limits<T>::quiet_NaN(),
                            std::numeric_limits<T>::quiet_NaN())};
  }

  T n_real = n.real();
  T n_rounded = std::round(n_real);
  if (n_real < T(0) || std::abs(n_real - n_rounded) > eps) {
    return {grad_n, Complex(std::numeric_limits<T>::quiet_NaN(),
                            std::numeric_limits<T>::quiet_NaN())};
  }

  int n_int = static_cast<int>(n_rounded);

  // d/dz E_n(z) = -E_{n-1}(z) for n >= 1
  // For n = 0: d/dz E_0(z) = d/dz (e^{-z}/z) = -e^{-z}/z - e^{-z}/z^2 = -e^{-z}(z+1)/z^2
  Complex dE_dz;
  if (n_int == 0) {
    if (std::abs(z) < eps) {
      dE_dz = Complex(std::numeric_limits<T>::quiet_NaN(),
                      std::numeric_limits<T>::quiet_NaN());
    } else {
      Complex exp_neg_z = std::exp(-z);
      Complex one(T(1), T(0));
      dE_dz = -exp_neg_z * (z + one) / (z * z);
    }
  } else {
    Complex n_m1(T(n_int - 1), T(0));
    Complex e_nm1 = exponential_integral_e(n_m1, z);
    dE_dz = -e_nm1;
  }

  // Wirtinger derivative: grad * conj(dE_dz)
  Complex grad_z = grad_output * std::conj(dE_dz);

  return {grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
