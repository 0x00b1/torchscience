#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "polylogarithm_li.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Derivative with respect to z: d/dz Li_s(z) = Li_{s-1}(z) / z
// This follows from: Li_s(z) = sum z^k / k^s
// d/dz = sum k * z^{k-1} / k^s = sum z^{k-1} / k^{s-1} = (1/z) * sum z^k / k^{s-1} = Li_{s-1}(z) / z
template <typename T>
T polylogarithm_li_dz(T s, T z) {
  if (z == T(0)) {
    // d/dz Li_s(z) at z=0: the derivative is Li_{s-1}(0)/0 which is 0/0
    // Actually Li_{s-1}(z)/z as z->0 = 1 (since Li_{s-1}(z) ~ z for small z)
    return T(1);
  }
  if (std::abs(z) > T(1)) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  return polylogarithm_li(s - T(1), z) / z;
}

template <typename T>
c10::complex<T> polylogarithm_li_dz(c10::complex<T> s, c10::complex<T> z) {
  c10::complex<T> one(T(1), T(0));
  if (std::abs(z) < std::numeric_limits<T>::epsilon()) {
    return one;
  }
  if (std::abs(z) > T(1)) {
    return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(),
                           std::numeric_limits<T>::quiet_NaN());
  }
  return polylogarithm_li(s - one, z) / z;
}

// Derivative with respect to s: d/ds Li_s(z) = -sum ln(k) * z^k / k^s
template <typename T>
T polylogarithm_li_ds(T s, T z, int max_terms = 200) {
  if (z == T(0)) {
    return T(0);
  }
  if (std::abs(z) > T(1)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  T sum = T(0);
  T z_power = z;

  for (int k = 1; k <= max_terms; ++k) {
    T k_t = static_cast<T>(k);
    T ln_k = std::log(k_t);
    T term = -ln_k * z_power / std::pow(k_t, s);
    sum += term;

    if (k > 1 && std::abs(term) < std::numeric_limits<T>::epsilon() * std::abs(sum)) {
      break;
    }

    z_power *= z;
  }

  return sum;
}

template <typename T>
c10::complex<T> polylogarithm_li_ds(c10::complex<T> s, c10::complex<T> z, int max_terms = 200) {
  c10::complex<T> zero(T(0), T(0));
  if (std::abs(z) < std::numeric_limits<T>::epsilon()) {
    return zero;
  }
  if (std::abs(z) > T(1)) {
    return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(),
                           std::numeric_limits<T>::quiet_NaN());
  }

  c10::complex<T> sum = zero;
  c10::complex<T> z_power = z;

  for (int k = 1; k <= max_terms; ++k) {
    c10::complex<T> k_c(static_cast<T>(k), T(0));
    T ln_k = std::log(static_cast<T>(k));
    c10::complex<T> term = c10::complex<T>(-ln_k, T(0)) * z_power / std::pow(k_c, s);
    sum += term;

    if (k > 1 && std::abs(term) < std::numeric_limits<T>::epsilon() * std::abs(sum)) {
      break;
    }

    z_power *= z;
  }

  return sum;
}

} // namespace detail

// Backward pass for polylogarithm_li
// Returns (grad_s, grad_z)
template <typename T>
std::tuple<T, T> polylogarithm_li_backward(T gradient, T s, T z) {
  T grad_s = gradient * detail::polylogarithm_li_ds(s, z);
  T grad_z = gradient * detail::polylogarithm_li_dz(s, z);
  return {grad_s, grad_z};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> polylogarithm_li_backward(
    c10::complex<T> gradient,
    c10::complex<T> s,
    c10::complex<T> z) {
  c10::complex<T> ds = detail::polylogarithm_li_ds(s, z);
  c10::complex<T> dz = detail::polylogarithm_li_dz(s, z);
  // PyTorch convention: multiply by conjugate for holomorphic functions
  c10::complex<T> grad_s = gradient * std::conj(ds);
  c10::complex<T> grad_z = gradient * std::conj(dz);
  return {grad_s, grad_z};
}

} // namespace torchscience::kernel::special_functions
