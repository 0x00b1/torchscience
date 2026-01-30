#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "exponential_integral_e.h"
#include "exponential_integral_e_backward.h"

namespace torchscience::kernel::special_functions {

// Real backward_backward: returns (grad_grad_output, grad_n, grad_x)
// Computes gradients of the backward pass w.r.t. (grad_output, n, x)
// given upstream gradients (gg_n, gg_x) for the outputs (grad_n, grad_x)
//
// Forward pass: y = E_n(x)
// Backward: grad_n = 0, grad_x = grad_output * (-E_{n-1}(x))
//
// Backward_backward:
// - d(grad_n)/d(anything) = 0 since grad_n = 0
// - d(grad_x)/d(grad_output) = -E_{n-1}(x)
// - d(grad_x)/d(n) = grad_output * (-d/dn E_{n-1}(x)) = 0 (n discrete)
// - d(grad_x)/d(x) = grad_output * (-d/dx E_{n-1}(x)) = grad_output * E_{n-2}(x)
template <typename T>
std::tuple<T, T, T> exponential_integral_e_backward_backward(
    T gg_n,       // upstream gradient for grad_n output
    T gg_x,       // upstream gradient for grad_x output
    T grad_output,
    T n,
    T x
) {
  const T eps = detail::exponential_integral_e_eps<T>();

  // Since grad_n = 0, gg_n does not contribute to any gradient
  (void)gg_n;

  // Validate n
  T n_rounded = std::round(n);
  if (n < T(0) || std::abs(n - n_rounded) > eps) {
    return {std::numeric_limits<T>::quiet_NaN(),
            T(0),
            std::numeric_limits<T>::quiet_NaN()};
  }

  int n_int = static_cast<int>(n_rounded);

  // grad_x = grad_output * dE_n/dx where dE_n/dx = -E_{n-1}(x)
  // So: d(grad_x)/d(grad_output) = dE_n/dx = -E_{n-1}(x)
  // And: d(grad_x)/d(x) = grad_output * d^2E_n/dx^2
  //                     = grad_output * (-d/dx E_{n-1}(x))
  //                     = grad_output * (E_{n-2}(x))

  T dE_n_dx;  // = -E_{n-1}(x)
  T d2E_n_dx2; // = E_{n-2}(x)

  if (n_int == 0) {
    // E_0(x) = e^{-x}/x
    // dE_0/dx = -e^{-x}(x+1)/x^2
    // d^2E_0/dx^2 = e^{-x} * (x^2 + 2x + 2) / x^3
    if (x == T(0)) {
      dE_n_dx = std::numeric_limits<T>::quiet_NaN();
      d2E_n_dx2 = std::numeric_limits<T>::quiet_NaN();
    } else {
      T exp_neg_x = std::exp(-x);
      dE_n_dx = -exp_neg_x * (x + T(1)) / (x * x);
      d2E_n_dx2 = exp_neg_x * (x * x + T(2) * x + T(2)) / (x * x * x);
    }
  } else if (n_int == 1) {
    // dE_1/dx = -E_0(x)
    // d^2E_1/dx^2 = -dE_0/dx = e^{-x}(x+1)/x^2
    T e_0 = exponential_integral_e(T(0), x);
    dE_n_dx = -e_0;
    if (x == T(0)) {
      d2E_n_dx2 = std::numeric_limits<T>::quiet_NaN();
    } else {
      T exp_neg_x = std::exp(-x);
      d2E_n_dx2 = exp_neg_x * (x + T(1)) / (x * x);
    }
  } else {
    // dE_n/dx = -E_{n-1}(x) for n >= 1
    // d^2E_n/dx^2 = E_{n-2}(x) for n >= 2
    T e_nm1 = exponential_integral_e(T(n_int - 1), x);
    T e_nm2 = exponential_integral_e(T(n_int - 2), x);
    dE_n_dx = -e_nm1;
    d2E_n_dx2 = e_nm2;
  }

  // grad_grad_output = gg_x * dE_n_dx (from grad_x = grad_output * dE_n_dx)
  T grad_grad_output = gg_x * dE_n_dx;

  // grad_n = 0 (n is discrete)
  T grad_n = T(0);

  // grad_x = gg_x * grad_output * d2E_n_dx2
  T grad_x = gg_x * grad_output * d2E_n_dx2;

  return {grad_grad_output, grad_n, grad_x};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
exponential_integral_e_backward_backward(
    c10::complex<T> gg_n,
    c10::complex<T> gg_z,
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
  using Complex = c10::complex<T>;
  const T eps = detail::exponential_integral_e_eps<T>();

  // Since grad_n = 0, gg_n does not contribute
  (void)gg_n;

  // Validate n
  if (std::abs(n.imag()) > eps) {
    return {Complex(std::numeric_limits<T>::quiet_NaN(),
                    std::numeric_limits<T>::quiet_NaN()),
            Complex(T(0), T(0)),
            Complex(std::numeric_limits<T>::quiet_NaN(),
                    std::numeric_limits<T>::quiet_NaN())};
  }

  T n_real = n.real();
  T n_rounded = std::round(n_real);
  if (n_real < T(0) || std::abs(n_real - n_rounded) > eps) {
    return {Complex(std::numeric_limits<T>::quiet_NaN(),
                    std::numeric_limits<T>::quiet_NaN()),
            Complex(T(0), T(0)),
            Complex(std::numeric_limits<T>::quiet_NaN(),
                    std::numeric_limits<T>::quiet_NaN())};
  }

  int n_int = static_cast<int>(n_rounded);

  Complex dE_n_dz;
  Complex d2E_n_dz2;
  Complex one(T(1), T(0));
  Complex two(T(2), T(0));

  if (n_int == 0) {
    if (std::abs(z) < eps) {
      dE_n_dz = Complex(std::numeric_limits<T>::quiet_NaN(),
                        std::numeric_limits<T>::quiet_NaN());
      d2E_n_dz2 = Complex(std::numeric_limits<T>::quiet_NaN(),
                          std::numeric_limits<T>::quiet_NaN());
    } else {
      Complex exp_neg_z = std::exp(-z);
      dE_n_dz = -exp_neg_z * (z + one) / (z * z);
      d2E_n_dz2 = exp_neg_z * (z * z + two * z + two) / (z * z * z);
    }
  } else if (n_int == 1) {
    Complex zero_c(T(0), T(0));
    Complex e_0 = exponential_integral_e(zero_c, z);
    dE_n_dz = -e_0;
    if (std::abs(z) < eps) {
      d2E_n_dz2 = Complex(std::numeric_limits<T>::quiet_NaN(),
                          std::numeric_limits<T>::quiet_NaN());
    } else {
      Complex exp_neg_z = std::exp(-z);
      d2E_n_dz2 = exp_neg_z * (z + one) / (z * z);
    }
  } else {
    Complex n_m1(T(n_int - 1), T(0));
    Complex n_m2(T(n_int - 2), T(0));
    Complex e_nm1 = exponential_integral_e(n_m1, z);
    Complex e_nm2 = exponential_integral_e(n_m2, z);
    dE_n_dz = -e_nm1;
    d2E_n_dz2 = e_nm2;
  }

  // Wirtinger derivatives
  Complex grad_grad_output = gg_z * std::conj(dE_n_dz);
  Complex grad_n(T(0), T(0));
  Complex grad_z = gg_z * grad_output * std::conj(d2E_n_dz2);

  return {grad_grad_output, grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
