#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::special_functions {

namespace detail {

// Constants for Lambert W computation
template <typename T>
constexpr T lambert_w_eps();

template <>
constexpr float lambert_w_eps<float>() { return 1e-6f; }

template <>
constexpr double lambert_w_eps<double>() { return 1e-14; }

template <>
inline c10::Half lambert_w_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 lambert_w_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

template <typename T>
constexpr int lambert_w_max_iter() { return 50; }

// Branch point constant: -1/e
template <typename T>
constexpr T minus_inv_e() {
  return static_cast<T>(-0.36787944117144232159552377016146086744581113103177);
}

// Euler's constant e
template <typename T>
constexpr T euler_e() {
  return static_cast<T>(2.71828182845904523536028747135266249775724709369995);
}

// Initial approximation for principal branch (k=0)
template <typename T>
T lambert_w_initial_guess_0(T z) {
  // Near z = 0: W(z) ~ z - z^2 + 3z^3/2
  if (std::abs(z) < T(0.1)) {
    T z2 = z * z;
    return z - z2 + T(1.5) * z2 * z;
  }

  // Near branch point z = -1/e: W(z) ~ -1 + sqrt(2(ez + 1))
  T branch_dist = z - minus_inv_e<T>();
  if (branch_dist >= T(0) && branch_dist < T(0.1)) {
    T p = std::sqrt(T(2) * euler_e<T>() * z + T(2));
    return T(-1) + p - p * p / T(3);
  }

  // For large positive z: W(z) ~ ln(z) - ln(ln(z))
  if (z > T(3)) {
    T lnz = std::log(z);
    T lnlnz = std::log(lnz);
    return lnz - lnlnz + lnlnz / lnz;
  }

  // For moderate z: start with log(1 + z) approximation
  if (z > T(0)) {
    return std::log(T(1) + z) * T(0.6);
  }

  // For negative z close to 0
  return z;
}

// Initial approximation for secondary branch (k=-1)
template <typename T>
T lambert_w_initial_guess_m1(T z) {
  // Branch -1 is only real for -1/e <= z < 0
  // Near branch point: W_{-1}(z) ~ -1 - sqrt(2(ez + 1))
  T branch_dist = z - minus_inv_e<T>();
  if (branch_dist >= T(0) && branch_dist < T(0.3)) {
    T p = std::sqrt(T(2) * euler_e<T>() * z + T(2));
    return T(-1) - p - p * p / T(3);
  }

  // For z closer to 0 from below: W_{-1}(z) ~ ln(-z) - ln(-ln(-z))
  if (z > T(-0.3) && z < T(0)) {
    T ln_mz = std::log(-z);
    T ln_mln_mz = std::log(-ln_mz);
    return ln_mz - ln_mln_mz;
  }

  // Default initial guess
  return T(-2);
}

// Halley's iteration for Lambert W
// w_{n+1} = w_n - (w_n * e^{w_n} - z) / (e^{w_n} * (w_n + 1) - (w_n + 2) * (w_n * e^{w_n} - z) / (2 * w_n + 2))
template <typename T>
T lambert_w_halley(T z, T w0) {
  const T eps = lambert_w_eps<T>();
  const int max_iter = lambert_w_max_iter<T>();

  T w = w0;

  for (int i = 0; i < max_iter; ++i) {
    T ew = std::exp(w);
    T wew = w * ew;
    T wewmz = wew - z;

    // Check convergence
    if (std::abs(wewmz) < eps * (std::abs(wew) + T(1))) {
      break;
    }

    T wp1 = w + T(1);

    // Avoid division by zero near w = -1
    if (std::abs(wp1) < eps) {
      // Use simple Newton step instead
      w = w - wewmz / (ew * wp1 + eps);
      continue;
    }

    // Halley's method
    T denom = ew * wp1 - (w + T(2)) * wewmz / (T(2) * wp1);

    if (std::abs(denom) < eps) {
      // Fall back to Newton
      w = w - wewmz / (ew * wp1);
    } else {
      w = w - wewmz / denom;
    }
  }

  return w;
}

// Complex Lambert W using Halley iteration
template <typename T>
c10::complex<T> lambert_w_halley_complex(c10::complex<T> z, c10::complex<T> w0) {
  const T eps = lambert_w_eps<T>();
  const int max_iter = lambert_w_max_iter<T>();

  c10::complex<T> w = w0;
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));

  for (int i = 0; i < max_iter; ++i) {
    c10::complex<T> ew = std::exp(w);
    c10::complex<T> wew = w * ew;
    c10::complex<T> wewmz = wew - z;

    // Check convergence
    if (std::abs(wewmz) < eps * (std::abs(wew) + T(1))) {
      break;
    }

    c10::complex<T> wp1 = w + one;

    // Halley's method
    c10::complex<T> denom = ew * wp1 - (w + two) * wewmz / (two * wp1);

    if (std::abs(denom) < eps) {
      // Fall back to Newton
      w = w - wewmz / (ew * wp1);
    } else {
      w = w - wewmz / denom;
    }
  }

  return w;
}

} // namespace detail

// Lambert W function principal branch (k=0)
template <typename T>
T lambert_w(T k, T z) {
  // Handle special cases
  if (std::isnan(z)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // W(0) = 0 for both branches
  if (z == T(0)) {
    return T(0);
  }

  // Branch point: W(-1/e) = -1
  const T branch_point = detail::minus_inv_e<T>();
  if (std::abs(z - branch_point) < detail::lambert_w_eps<T>()) {
    return T(-1);
  }

  // Determine branch from k (round to nearest integer)
  int branch = static_cast<int>(std::round(k));

  if (branch == 0) {
    // Principal branch: defined for z >= -1/e
    if (z < branch_point) {
      return std::numeric_limits<T>::quiet_NaN();
    }

    T w0 = detail::lambert_w_initial_guess_0(z);
    return detail::lambert_w_halley(z, w0);

  } else if (branch == -1) {
    // Branch -1: defined for -1/e <= z < 0
    if (z < branch_point || z >= T(0)) {
      return std::numeric_limits<T>::quiet_NaN();
    }

    T w0 = detail::lambert_w_initial_guess_m1(z);
    return detail::lambert_w_halley(z, w0);

  } else {
    // Other branches not supported for real numbers
    return std::numeric_limits<T>::quiet_NaN();
  }
}

// Complex Lambert W function
template <typename T>
c10::complex<T> lambert_w(c10::complex<T> k, c10::complex<T> z) {
  // Handle special case z = 0
  if (std::abs(z) < detail::lambert_w_eps<T>()) {
    return c10::complex<T>(T(0), T(0));
  }

  // Get branch index (use real part of k, rounded to integer)
  int branch = static_cast<int>(std::round(k.real()));

  // Initial guess for complex case
  // For principal branch, use log(z) as starting point
  // For branch k, use log(z) + 2*pi*i*k
  c10::complex<T> w0;

  if (branch == 0) {
    // Principal branch initial guess
    T abs_z = std::abs(z);
    if (abs_z < T(0.1)) {
      w0 = z;
    } else {
      w0 = std::log(z);
    }
  } else if (branch == -1) {
    // Branch -1 initial guess
    if (z.imag() >= T(0)) {
      w0 = std::log(z) - c10::complex<T>(T(0), T(2) * static_cast<T>(M_PI));
    } else {
      w0 = std::log(z) + c10::complex<T>(T(0), T(2) * static_cast<T>(M_PI));
    }
  } else {
    // General branch k
    T two_pi_k = T(2) * static_cast<T>(M_PI) * static_cast<T>(branch);
    w0 = std::log(z) + c10::complex<T>(T(0), two_pi_k);
  }

  return detail::lambert_w_halley_complex(z, w0);
}

} // namespace torchscience::kernel::special_functions
