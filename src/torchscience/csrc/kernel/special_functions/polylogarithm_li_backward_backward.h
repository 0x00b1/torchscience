#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "polylogarithm_li_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivatives of polylogarithm

// d^2/dz^2 Li_s(z) = d/dz [Li_{s-1}(z)/z]
//                  = [Li_{s-2}(z)/z - Li_{s-1}(z)] / z^2
//                  = Li_{s-2}(z)/z^2 - Li_{s-1}(z)/z^2
template <typename T>
T polylogarithm_li_d2z(T s, T z) {
  if (std::abs(z) < std::numeric_limits<T>::epsilon()) {
    // Limit as z->0
    return T(0);
  }
  if (std::abs(z) > T(1)) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  T li_sm2 = polylogarithm_li(s - T(2), z);
  T li_sm1 = polylogarithm_li(s - T(1), z);
  return (li_sm2 - li_sm1) / (z * z);
}

template <typename T>
c10::complex<T> polylogarithm_li_d2z(c10::complex<T> s, c10::complex<T> z) {
  if (std::abs(z) < std::numeric_limits<T>::epsilon()) {
    return c10::complex<T>(T(0), T(0));
  }
  if (std::abs(z) > T(1)) {
    return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(),
                           std::numeric_limits<T>::quiet_NaN());
  }
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));
  c10::complex<T> li_sm2 = polylogarithm_li(s - two, z);
  c10::complex<T> li_sm1 = polylogarithm_li(s - one, z);
  return (li_sm2 - li_sm1) / (z * z);
}

// d^2/ds^2 Li_s(z) = sum (ln k)^2 * z^k / k^s
template <typename T>
T polylogarithm_li_d2s(T s, T z, int max_terms = 200) {
  if (std::abs(z) < std::numeric_limits<T>::epsilon()) {
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
    T term = ln_k * ln_k * z_power / std::pow(k_t, s);
    sum += term;

    if (k > 1 && std::abs(term) < std::numeric_limits<T>::epsilon() * std::abs(sum)) {
      break;
    }

    z_power *= z;
  }

  return sum;
}

template <typename T>
c10::complex<T> polylogarithm_li_d2s(c10::complex<T> s, c10::complex<T> z, int max_terms = 200) {
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
    c10::complex<T> term = c10::complex<T>(ln_k * ln_k, T(0)) * z_power / std::pow(k_c, s);
    sum += term;

    if (k > 1 && std::abs(term) < std::numeric_limits<T>::epsilon() * std::abs(sum)) {
      break;
    }

    z_power *= z;
  }

  return sum;
}

// d^2/dsdz Li_s(z) = d/ds [Li_{s-1}(z)/z] = (d/ds Li_{s-1}(z)) / z
//                  = -sum ln(k) * z^{k-1} / k^{s-1}
//                  = (1/z) * d/ds Li_{s-1}(z)
template <typename T>
T polylogarithm_li_d2sz(T s, T z) {
  if (std::abs(z) < std::numeric_limits<T>::epsilon()) {
    return T(0);
  }
  if (std::abs(z) > T(1)) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  return polylogarithm_li_ds(s - T(1), z) / z;
}

template <typename T>
c10::complex<T> polylogarithm_li_d2sz(c10::complex<T> s, c10::complex<T> z) {
  if (std::abs(z) < std::numeric_limits<T>::epsilon()) {
    return c10::complex<T>(T(0), T(0));
  }
  if (std::abs(z) > T(1)) {
    return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(),
                           std::numeric_limits<T>::quiet_NaN());
  }
  c10::complex<T> one(T(1), T(0));
  return polylogarithm_li_ds(s - one, z) / z;
}

} // namespace detail

// Second-order backward pass for polylogarithm_li
// Returns (grad_gradient, grad_s, grad_z)
template <typename T>
std::tuple<T, T, T> polylogarithm_li_backward_backward(
    T grad_grad_s,
    T grad_grad_z,
    T gradient,
    T s,
    T z) {
  // First derivatives
  T ds = detail::polylogarithm_li_ds(s, z);
  T dz = detail::polylogarithm_li_dz(s, z);

  // Second derivatives
  T d2s = detail::polylogarithm_li_d2s(s, z);
  T d2z = detail::polylogarithm_li_d2z(s, z);
  T d2sz = detail::polylogarithm_li_d2sz(s, z);

  // grad_gradient: derivative of output w.r.t. input gradient
  T grad_gradient = grad_grad_s * ds + grad_grad_z * dz;

  // grad_s: derivative w.r.t. s
  T grad_s = grad_grad_s * gradient * d2s + grad_grad_z * gradient * d2sz;

  // grad_z: derivative w.r.t. z
  T grad_z = grad_grad_s * gradient * d2sz + grad_grad_z * gradient * d2z;

  return {grad_gradient, grad_s, grad_z};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> polylogarithm_li_backward_backward(
    c10::complex<T> grad_grad_s,
    c10::complex<T> grad_grad_z,
    c10::complex<T> gradient,
    c10::complex<T> s,
    c10::complex<T> z) {
  // First derivatives
  c10::complex<T> ds = detail::polylogarithm_li_ds(s, z);
  c10::complex<T> dz = detail::polylogarithm_li_dz(s, z);

  // Second derivatives
  c10::complex<T> d2s = detail::polylogarithm_li_d2s(s, z);
  c10::complex<T> d2z = detail::polylogarithm_li_d2z(s, z);
  c10::complex<T> d2sz = detail::polylogarithm_li_d2sz(s, z);

  // With conjugation for PyTorch convention
  c10::complex<T> grad_gradient = grad_grad_s * std::conj(ds) + grad_grad_z * std::conj(dz);
  c10::complex<T> grad_s = grad_grad_s * gradient * std::conj(d2s) +
                           grad_grad_z * gradient * std::conj(d2sz);
  c10::complex<T> grad_z = grad_grad_s * gradient * std::conj(d2sz) +
                           grad_grad_z * gradient * std::conj(d2z);

  return {grad_gradient, grad_s, grad_z};
}

} // namespace torchscience::kernel::special_functions
