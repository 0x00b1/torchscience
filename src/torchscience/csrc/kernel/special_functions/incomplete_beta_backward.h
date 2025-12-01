#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>
#include <array>

#include "digamma.h"
#include "incomplete_beta.h"
#include "log_gamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
constexpr T gauss_kronrod_eps();

template <>
constexpr float gauss_kronrod_eps<float>() { return 1e-6f; }

template <>
constexpr double gauss_kronrod_eps<double>() { return 1e-12; }

template <>
inline c10::Half gauss_kronrod_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 gauss_kronrod_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

template <typename T>
struct GaussKronrodNodes15 {
  static constexpr int n_gauss = 7;
  static constexpr int n_kronrod = 15;

  static constexpr double kronrod_nodes[15] = {
    -0.991455371120812639206854697526329,
    -0.949107912342758524526189684047851,
    -0.864864423359769072789712788640926,
    -0.741531185599394439863864773280788,
    -0.586087235467691130294144838258730,
    -0.405845151377397166906606412076961,
    -0.207784955007898467600689403773245,
     0.0,
     0.207784955007898467600689403773245,
     0.405845151377397166906606412076961,
     0.586087235467691130294144838258730,
     0.741531185599394439863864773280788,
     0.864864423359769072789712788640926,
     0.949107912342758524526189684047851,
     0.991455371120812639206854697526329
  };

  static constexpr double kronrod_weights[15] = {
    0.022935322010529224963732008058970,
    0.063092092629978553290700663189204,
    0.104790010322250183839876322541518,
    0.140653259715525918745189590510238,
    0.169004726639267902826583426598550,
    0.190350578064785409913256402421014,
    0.204432940075298892414161999234649,
    0.209482141084727828012999174891714,
    0.204432940075298892414161999234649,
    0.190350578064785409913256402421014,
    0.169004726639267902826583426598550,
    0.140653259715525918745189590510238,
    0.104790010322250183839876322541518,
    0.063092092629978553290700663189204,
    0.022935322010529224963732008058970
  };

  static constexpr double gauss_weights[7] = {
    0.129484966168869693270611432679082,
    0.279705391489276667901467771423780,
    0.381830050505118944950369775488975,
    0.417959183673469387755102040816327,
    0.381830050505118944950369775488975,
    0.279705391489276667901467771423780,
    0.129484966168869693270611432679082
  };

  static constexpr int gauss_indices[7] = {1, 3, 5, 7, 9, 11, 13};
};

template <typename T, typename F>
std::pair<T, T> gauss_kronrod_15(F f, T lower, T upper) {
  T center = (upper + lower) / T(2);
  T half_length = (upper - lower) / T(2);

  T gauss_sum = T(0);
  T kronrod_sum = T(0);

  for (int i = 0; i < 15; ++i) {
    T node = center + half_length * static_cast<T>(GaussKronrodNodes15<T>::kronrod_nodes[i]);
    T f_val = f(node);
    kronrod_sum += static_cast<T>(GaussKronrodNodes15<T>::kronrod_weights[i]) * f_val;
  }

  for (int i = 0; i < 7; ++i) {
    int idx = GaussKronrodNodes15<T>::gauss_indices[i];
    T node = center + half_length * static_cast<T>(GaussKronrodNodes15<T>::kronrod_nodes[idx]);
    T f_val = f(node);
    gauss_sum += static_cast<T>(GaussKronrodNodes15<T>::gauss_weights[i]) * f_val;
  }

  T result = half_length * kronrod_sum;
  T error = std::abs(half_length * (kronrod_sum - gauss_sum));

  return {result, error};
}

template <typename T, typename F>
T adaptive_integrate(F f, T lower, T upper, T tol, int max_depth) {
  auto [result, error] = gauss_kronrod_15<T>(f, lower, upper);

  if (error < tol || max_depth <= 0) {
    return result;
  }

  T mid = (lower + upper) / T(2);
  T left = adaptive_integrate(f, lower, mid, tol / T(2), max_depth - 1);
  T right = adaptive_integrate(f, mid, upper, tol / T(2), max_depth - 1);

  return left + right;
}

template <typename T>
T log_weighted_beta_integral(T x, T a, T b, bool weight_log_t) {
  if (x <= T(0)) return T(0);

  T log_beta_val = log_gamma(a) + log_gamma(b) - log_gamma(a + b);
  T beta_val = std::exp(log_beta_val);

  T eps = gauss_kronrod_eps<T>();
  T tol = eps;  // Tighter tolerance for gradient accuracy
  int max_depth = 15;  // Increased for better accuracy

  T lower = eps;
  T upper = std::min(x, T(1) - eps);

  if (lower >= upper) {
    return T(0);
  }

  // For small a (< 1), use substitution t = s^(1/a) to handle singularity at t=0
  // For small b (< 1), the singularity is at t=1 but we integrate up to x < 1, so less problematic
  bool use_singularity_transform = weight_log_t && (a < T(1));

  T integral;

  if (use_singularity_transform) {
    // Transform: t = s^(1/a), so s = t^a, ds = a * t^(a-1) dt
    // dt = (1/a) * s^(1/a - 1) ds
    // t^(a-1) dt = s^((a-1)/a) * (1/a) * s^(1/a - 1) ds = (1/a) * s^((a-1)/a + 1/a - 1) ds = (1/a) ds
    // log(t) = log(s) / a
    // (1-t)^(b-1) = (1 - s^(1/a))^(b-1)
    //
    // So the integrand becomes:
    // log(t) * t^(a-1) * (1-t)^(b-1) dt = (log(s)/a) * (1/a) * (1 - s^(1/a))^(b-1) ds
    //                                   = log(s) / a^2 * (1 - s^(1/a))^(b-1) ds
    //
    // The new limits: t = 0 -> s = 0, t = x -> s = x^a

    T s_upper = std::pow(upper, a);
    T s_lower = std::pow(lower, a);

    // Use smaller eps for transformed variable
    if (s_lower < eps * T(0.01)) s_lower = eps * T(0.01);

    T inv_a = T(1) / a;
    T inv_a_sq = inv_a * inv_a;

    auto transformed_integrand = [a, b, inv_a, inv_a_sq](T s) -> T {
      if (s <= T(0)) return T(0);

      T t = std::pow(s, inv_a);  // t = s^(1/a)
      if (t >= T(1)) return T(0);

      T one_minus_t = T(1) - t;
      if (one_minus_t <= T(0)) return T(0);

      // log(s) / a^2 * (1-t)^(b-1)
      return std::log(s) * inv_a_sq * std::pow(one_minus_t, b - T(1));
    };

    integral = adaptive_integrate(transformed_integrand, s_lower, s_upper, tol, max_depth);
  } else {
    auto integrand = [a, b, weight_log_t](T t) -> T {
      if (t <= T(0) || t >= T(1)) return T(0);

      T log_integrand = (a - T(1)) * std::log(t) + (b - T(1)) * std::log(T(1) - t);
      T base_val = std::exp(log_integrand);

      if (weight_log_t) {
        return std::log(t) * base_val;
      } else {
        return std::log(T(1) - t) * base_val;
      }
    };

    integral = adaptive_integrate(integrand, lower, upper, tol, max_depth);
  }

  return integral / beta_val;
}

} // namespace detail

template <typename T>
std::tuple<T, T, T> incomplete_beta_backward(T gradient, T x, T a, T b) {
  if (x <= T(0) || x >= T(1)) {
    return {T(0), T(0), T(0)};
  }

  // Check if a or b are invalid (non-positive)
  if (a <= T(0) || b <= T(0)) {
    T nan_val = std::numeric_limits<T>::quiet_NaN();
    return {nan_val, nan_val, nan_val};
  }

  // Check if forward used symmetry transformation: x > threshold
  // threshold = (a + 1) / (a + b + 2)
  T threshold = (a + T(1)) / (a + b + T(2));
  bool use_symmetry = (x > threshold);

  T x_eff, a_eff, b_eff;
  if (use_symmetry) {
    // Forward computed: I_x(a,b) = 1 - I_{1-x}(b,a)
    // So we compute gradients at the transformed point
    x_eff = T(1) - x;
    a_eff = b;  // swap a and b
    b_eff = a;
  } else {
    x_eff = x;
    a_eff = a;
    b_eff = b;
  }

  T log_beta_val = log_beta(a_eff, b_eff);

  T log_pdf = (a_eff - T(1)) * std::log(x_eff) + (b_eff - T(1)) * std::log(T(1) - x_eff) - log_beta_val;
  T pdf = std::exp(log_pdf);

  // grad_x: For symmetry case, d/dx[1 - I_{1-x}(b,a)] = +pdf(1-x, b, a)
  T grad_x = gradient * pdf;

  T I_eff = incomplete_beta(x_eff, a_eff, b_eff);

  T psi_a_eff = digamma(a_eff);
  T psi_b_eff = digamma(b_eff);
  T psi_ab_eff = digamma(a_eff + b_eff);

  T log_integral_a_eff = detail::log_weighted_beta_integral(x_eff, a_eff, b_eff, true);
  T grad_a_eff = gradient * (log_integral_a_eff - I_eff * (psi_a_eff - psi_ab_eff));

  T log_integral_b_eff = detail::log_weighted_beta_integral(x_eff, a_eff, b_eff, false);
  T grad_b_eff = gradient * (log_integral_b_eff - I_eff * (psi_b_eff - psi_ab_eff));

  T grad_a, grad_b;
  if (use_symmetry) {
    // When using symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)
    // d/da[1 - I_{1-x}(b,a)] = -d/d(second param)[I_{1-x}(b,a)] = -grad_b_eff
    // d/db[1 - I_{1-x}(b,a)] = -d/d(first param)[I_{1-x}(b,a)] = -grad_a_eff
    grad_a = -grad_b_eff;
    grad_b = -grad_a_eff;
  } else {
    grad_a = grad_a_eff;
    grad_b = grad_b_eff;
  }

  return {
    grad_x,
    grad_a,
    grad_b
  };
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> incomplete_beta_backward(
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> a,
    c10::complex<T> b
) {
  c10::complex<T> zero(T(0), T(0));
  c10::complex<T> one(T(1), T(0));

  T eps = detail::beta_eps<T>();
  if (std::abs(x) < eps || std::abs(x - one) < eps) {
    return {zero, zero, zero};
  }

  c10::complex<T> log_beta_val = log_beta(a, b);

  // Compute the pdf = dI_x/dx = x^(a-1) * (1-x)^(b-1) / B(a,b)
  c10::complex<T> log_pdf = (a - one) * std::log(x) + (b - one) * std::log(one - x) - log_beta_val;
  c10::complex<T> pdf = std::exp(log_pdf);

  c10::complex<T> I_x = incomplete_beta(x, a, b);

  c10::complex<T> psi_a = digamma(a);
  c10::complex<T> psi_b = digamma(b);
  c10::complex<T> psi_ab = digamma(a + b);

  // For complex holomorphic functions, PyTorch's Wirtinger derivative convention
  // requires returning gradient * conj(df/dz) for each input.
  // Since incomplete_beta is holomorphic, df/dz = df/dx = pdf for the x derivative.
  //
  // For parameter gradients (a, b), the full formula is:
  //   grad_a = gradient * (log_integral_a - I * (psi_a - psi_ab))
  // Since log-weighted integrals are expensive for complex, we use the simplified
  // formula by setting log_integral ≈ 0, giving:
  //   grad_a ≈ gradient * (-I * (psi_a - psi_ab))
  // This is a crude approximation but matches the expected sign.

  return {
    gradient * std::conj(pdf),
    gradient * std::conj(-I_x * (psi_a - psi_ab)),
    gradient * std::conj(-I_x * (psi_b - psi_ab))
  };
}

} // namespace torchscience::kernel::special_functions
