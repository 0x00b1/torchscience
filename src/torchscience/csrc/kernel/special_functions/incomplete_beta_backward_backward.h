#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "digamma.h"
#include "incomplete_beta.h"
#include "incomplete_beta_backward.h"
#include "log_gamma.h"
#include "trigamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
T log_squared_weighted_beta_integral(T x, T a, T b, int weight_type) {
  if (x <= T(0)) return T(0);

  T log_beta_val = log_gamma(a) + log_gamma(b) - log_gamma(a + b);
  T beta_val = std::exp(log_beta_val);

  T eps = gauss_kronrod_eps<T>();
  T tol = eps;  // Tighter tolerance for gradgradcheck
  int max_depth = 15;  // Increased for better accuracy

  T lower = eps;
  T upper = std::min(x, T(1) - eps);

  if (lower >= upper) {
    return T(0);
  }

  // Use singularity transformation for weight_type 0 (log(t)^2) or 2 (log(t)*log(1-t)) when a < 1
  bool use_singularity_transform = (a < T(1)) && (weight_type == 0 || weight_type == 2);

  T integral;

  if (use_singularity_transform) {
    // Same transformation as in log_weighted_beta_integral:
    // t = s^(1/a), dt = (1/a) * s^(1/a - 1) ds
    // t^(a-1) dt = (1/a) ds
    // log(t) = log(s) / a
    // log(t)^2 = log(s)^2 / a^2

    T s_upper = std::pow(upper, a);
    T s_lower = std::pow(lower, a);
    if (s_lower < eps * T(0.01)) s_lower = eps * T(0.01);

    T inv_a = T(1) / a;
    T inv_a_sq = inv_a * inv_a;
    T inv_a_cu = inv_a_sq * inv_a;

    auto transformed_integrand = [a, b, inv_a, inv_a_sq, inv_a_cu, weight_type](T s) -> T {
      if (s <= T(0)) return T(0);

      T t = std::pow(s, inv_a);
      if (t >= T(1)) return T(0);

      T one_minus_t = T(1) - t;
      if (one_minus_t <= T(0)) return T(0);

      T one_minus_t_pow = std::pow(one_minus_t, b - T(1));
      T log_s = std::log(s);

      // weight_type 0: log(t)^2 * t^(a-1) * (1-t)^(b-1) dt
      //              = (log(s)/a)^2 * (1/a) * (1-t)^(b-1) ds
      //              = log(s)^2 / a^3 * (1-t)^(b-1) ds
      if (weight_type == 0) {
        return log_s * log_s * inv_a_cu * one_minus_t_pow;
      }
      // weight_type 2: log(t) * log(1-t) * t^(a-1) * (1-t)^(b-1) dt
      //              = (log(s)/a) * log(1-t) * (1/a) * (1-t)^(b-1) ds
      //              = log(s) / a^2 * log(1-t) * (1-t)^(b-1) ds
      else {
        T log_1mt = std::log(one_minus_t);
        return log_s * inv_a_sq * log_1mt * one_minus_t_pow;
      }
    };

    integral = adaptive_integrate(transformed_integrand, s_lower, s_upper, tol, max_depth);
  } else {
    auto integrand = [a, b, weight_type](T t) -> T {
      if (t <= T(0) || t >= T(1)) return T(0);

      T log_integrand = (a - T(1)) * std::log(t) + (b - T(1)) * std::log(T(1) - t);
      T base_val = std::exp(log_integrand);

      T log_t = std::log(t);
      T log_1mt = std::log(T(1) - t);

      switch (weight_type) {
        case 0: return log_t * log_t * base_val;
        case 1: return log_1mt * log_1mt * base_val;
        case 2: return log_t * log_1mt * base_val;
        default: return T(0);
      }
    };

    integral = adaptive_integrate(integrand, lower, upper, tol, max_depth);
  }

  return integral / beta_val;
}

} // namespace detail

template <typename T>
std::tuple<T, T, T, T> incomplete_beta_backward_backward(
    T gg_x, T gg_a, T gg_b,
    T gradient, T x, T a, T b) {

  T gg_out = T(0);
  T new_grad_x = T(0);
  T new_grad_a = T(0);
  T new_grad_b = T(0);

  if (x <= T(0) || x >= T(1)) {
    return {gg_out, new_grad_x, new_grad_a, new_grad_b};
  }

  // Check if a or b are invalid (non-positive)
  if (a <= T(0) || b <= T(0)) {
    T nan_val = std::numeric_limits<T>::quiet_NaN();
    return {nan_val, nan_val, nan_val, nan_val};
  }

  // Check if forward (and backward) used symmetry transformation
  T threshold = (a + T(1)) / (a + b + T(2));
  bool use_symmetry = (x > threshold);

  T x_eff, a_eff, b_eff;
  T gg_a_eff, gg_b_eff;
  if (use_symmetry) {
    x_eff = T(1) - x;
    a_eff = b;
    b_eff = a;
    // When using symmetry, grad_a = -grad_b_eff and grad_b = -grad_a_eff
    // So gg_a corresponds to -gg_b_eff and gg_b corresponds to -gg_a_eff
    gg_a_eff = -gg_b;
    gg_b_eff = -gg_a;
  } else {
    x_eff = x;
    a_eff = a;
    b_eff = b;
    gg_a_eff = gg_a;
    gg_b_eff = gg_b;
  }

  T log_beta_val = log_beta(a_eff, b_eff);

  T log_pdf = (a_eff - T(1)) * std::log(x_eff) + (b_eff - T(1)) * std::log(T(1) - x_eff) - log_beta_val;
  T pdf = std::exp(log_pdf);

  T I_eff = incomplete_beta(x_eff, a_eff, b_eff);

  T psi_a_eff = digamma(a_eff);
  T psi_b_eff = digamma(b_eff);
  T psi_ab_eff = digamma(a_eff + b_eff);
  T psi1_a_eff = trigamma(a_eff);
  T psi1_b_eff = trigamma(b_eff);
  T psi1_ab_eff = trigamma(a_eff + b_eff);

  // gg_x contributions
  // grad_x = gradient * pdf (same for both cases since pdf at effective point is correct)
  gg_out += gg_x * pdf;

  T d_pdf_dx_eff = pdf * ((a_eff - T(1)) / x_eff - (b_eff - T(1)) / (T(1) - x_eff));
  // When using symmetry, x_eff = 1-x, so dx_eff/dx = -1
  T dx_eff_dx = use_symmetry ? T(-1) : T(1);
  new_grad_x += gg_x * gradient * d_pdf_dx_eff * dx_eff_dx;

  T d_pdf_da_eff = pdf * (std::log(x_eff) - (psi_a_eff - psi_ab_eff));
  T d_pdf_db_eff = pdf * (std::log(T(1) - x_eff) - (psi_b_eff - psi_ab_eff));

  // gg_a_eff and gg_b_eff contributions
  T log_int_a_eff = detail::log_weighted_beta_integral(x_eff, a_eff, b_eff, true);
  T log_int_b_eff = detail::log_weighted_beta_integral(x_eff, a_eff, b_eff, false);
  T grad_a_over_g_eff = log_int_a_eff - I_eff * (psi_a_eff - psi_ab_eff);
  T grad_b_over_g_eff = log_int_b_eff - I_eff * (psi_b_eff - psi_ab_eff);

  gg_out += gg_a_eff * grad_a_over_g_eff;
  gg_out += gg_b_eff * grad_b_over_g_eff;

  // d(grad_a_eff)/dx_eff = gradient * (d(log_integral_a)/dx - d(I)/dx * (psi_a - psi_ab))
  //                      = gradient * (log(x) * pdf - pdf * (psi_a - psi_ab))
  // d(grad_b_eff)/dx_eff = gradient * (log(1-x) * pdf - pdf * (psi_b - psi_ab))
  // With symmetry we need dx_eff/dx = -1
  new_grad_x += gg_a_eff * gradient * (std::log(x_eff) - (psi_a_eff - psi_ab_eff)) * pdf * dx_eff_dx;
  new_grad_x += gg_b_eff * gradient * (std::log(T(1) - x_eff) - (psi_b_eff - psi_ab_eff)) * pdf * dx_eff_dx;

  T log_sq_int_aa = detail::log_squared_weighted_beta_integral(x_eff, a_eff, b_eff, 0);
  T log_sq_int_bb = detail::log_squared_weighted_beta_integral(x_eff, a_eff, b_eff, 1);
  T log_sq_int_ab = detail::log_squared_weighted_beta_integral(x_eff, a_eff, b_eff, 2);

  T d_grad_a_da_eff = log_sq_int_aa
                - T(2) * log_int_a_eff * (psi_a_eff - psi_ab_eff)
                + I_eff * (psi_a_eff - psi_ab_eff) * (psi_a_eff - psi_ab_eff)
                - I_eff * (psi1_a_eff - psi1_ab_eff);

  T d_grad_a_db_eff = log_sq_int_ab
                - log_int_a_eff * (psi_b_eff - psi_ab_eff)
                - log_int_b_eff * (psi_a_eff - psi_ab_eff)
                + I_eff * (psi_a_eff - psi_ab_eff) * (psi_b_eff - psi_ab_eff)
                + I_eff * psi1_ab_eff;

  T d_grad_b_da_eff = d_grad_a_db_eff;  // Symmetric

  T d_grad_b_db_eff = log_sq_int_bb
                - T(2) * log_int_b_eff * (psi_b_eff - psi_ab_eff)
                + I_eff * (psi_b_eff - psi_ab_eff) * (psi_b_eff - psi_ab_eff)
                - I_eff * (psi1_b_eff - psi1_ab_eff);

  // Compute new_grad_a_eff and new_grad_b_eff
  T new_grad_a_eff = gg_x * gradient * d_pdf_da_eff
                   + gg_a_eff * gradient * d_grad_a_da_eff
                   + gg_b_eff * gradient * d_grad_b_da_eff;

  T new_grad_b_eff = gg_x * gradient * d_pdf_db_eff
                   + gg_a_eff * gradient * d_grad_a_db_eff
                   + gg_b_eff * gradient * d_grad_b_db_eff;

  // Transform back to original parameters if using symmetry
  if (use_symmetry) {
    // a_eff = b, b_eff = a
    // So new_grad_a_eff is d/db and new_grad_b_eff is d/da
    new_grad_a = new_grad_b_eff;
    new_grad_b = new_grad_a_eff;
  } else {
    new_grad_a = new_grad_a_eff;
    new_grad_b = new_grad_b_eff;
  }

  return {gg_out, new_grad_x, new_grad_a, new_grad_b};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
incomplete_beta_backward_backward(
    c10::complex<T> gg_x, c10::complex<T> gg_a, c10::complex<T> gg_b,
    c10::complex<T> gradient, c10::complex<T> x, c10::complex<T> a, c10::complex<T> b) {

  c10::complex<T> zero(T(0), T(0));
  c10::complex<T> one(T(1), T(0));

  c10::complex<T> gg_out = zero;
  c10::complex<T> new_grad_x = zero;
  c10::complex<T> new_grad_a = zero;
  c10::complex<T> new_grad_b = zero;

  T eps = detail::beta_eps<T>();
  if (std::abs(x) < eps || std::abs(x - one) < eps) {
    return {gg_out, new_grad_x, new_grad_a, new_grad_b};
  }

  c10::complex<T> log_beta_val = log_beta(a, b);
  c10::complex<T> log_pdf = (a - one) * std::log(x) + (b - one) * std::log(one - x) - log_beta_val;
  c10::complex<T> pdf = std::exp(log_pdf);

  c10::complex<T> I_x = incomplete_beta(x, a, b);

  c10::complex<T> psi_a = digamma(a);
  c10::complex<T> psi_b = digamma(b);
  c10::complex<T> psi_ab = digamma(a + b);
  c10::complex<T> psi1_a = trigamma(a);
  c10::complex<T> psi1_b = trigamma(b);
  c10::complex<T> psi1_ab = trigamma(a + b);

  // For complex holomorphic functions, we need to conjugate derivatives
  // to match PyTorch's Wirtinger derivative convention.

  // gg_x contributions: d(backward_x)/d* needs conjugation
  gg_out += gg_x * std::conj(pdf);

  c10::complex<T> d_pdf_dx = pdf * ((a - one) / x - (b - one) / (one - x));
  new_grad_x += gg_x * gradient * std::conj(d_pdf_dx);

  c10::complex<T> d_pdf_da = pdf * (std::log(x) - (psi_a - psi_ab));
  new_grad_a += gg_x * gradient * std::conj(d_pdf_da);

  c10::complex<T> d_pdf_db = pdf * (std::log(one - x) - (psi_b - psi_ab));
  new_grad_b += gg_x * gradient * std::conj(d_pdf_db);

  // gg_a contributions (simplified formula: grad_a = -I * (psi_a - psi_ab))
  c10::complex<T> grad_a_over_g = -I_x * (psi_a - psi_ab);
  gg_out += gg_a * std::conj(grad_a_over_g);

  // d(grad_a)/dx = d(-I*(psi_a - psi_ab))/dx = -pdf * (psi_a - psi_ab)
  new_grad_x += gg_a * gradient * std::conj(-pdf * (psi_a - psi_ab));

  // d(grad_a)/da = d(-I*(psi_a - psi_ab))/da = -pdf*log(x)*(psi_a - psi_ab) - I*(psi1_a - psi1_ab)
  // Simplified: -I*(psi_a - psi_ab)^2 + I*(psi1_a - psi1_ab)
  c10::complex<T> d_grad_a_da = -I_x * (psi_a - psi_ab) * (psi_a - psi_ab) + I_x * (psi1_a - psi1_ab);
  new_grad_a += gg_a * gradient * std::conj(d_grad_a_da);

  // d(grad_a)/db = -I*(psi_a - psi_ab)*(psi_b - psi_ab) - I*psi1_ab
  c10::complex<T> d_grad_a_db = -I_x * (psi_a - psi_ab) * (psi_b - psi_ab) - I_x * psi1_ab;
  new_grad_b += gg_a * gradient * std::conj(d_grad_a_db);

  // gg_b contributions (simplified formula: grad_b = -I * (psi_b - psi_ab))
  c10::complex<T> grad_b_over_g = -I_x * (psi_b - psi_ab);
  gg_out += gg_b * std::conj(grad_b_over_g);

  // d(grad_b)/dx = -pdf * (psi_b - psi_ab)
  new_grad_x += gg_b * gradient * std::conj(-pdf * (psi_b - psi_ab));

  // d(grad_b)/da = -I*(psi_a - psi_ab)*(psi_b - psi_ab) - I*psi1_ab
  c10::complex<T> d_grad_b_da = -I_x * (psi_a - psi_ab) * (psi_b - psi_ab) - I_x * psi1_ab;
  new_grad_a += gg_b * gradient * std::conj(d_grad_b_da);

  // d(grad_b)/db = -I*(psi_b - psi_ab)^2 + I*(psi1_b - psi1_ab)
  c10::complex<T> d_grad_b_db = -I_x * (psi_b - psi_ab) * (psi_b - psi_ab) + I_x * (psi1_b - psi1_ab);
  new_grad_b += gg_b * gradient * std::conj(d_grad_b_db);

  return {gg_out, new_grad_x, new_grad_a, new_grad_b};
}

} // namespace torchscience::kernel::special_functions
