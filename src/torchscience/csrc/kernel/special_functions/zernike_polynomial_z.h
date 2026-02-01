#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "zernike_polynomial_r.h"

namespace torchscience::kernel::special_functions {

// Full Zernike polynomial Z_n^m(rho, theta)
//
// The full Zernike polynomial combines the radial Zernike polynomial R_n^m(rho)
// with angular functions:
//
// Mathematical Definition:
//   Z_n^m(rho, theta) = R_n^|m|(rho) * cos(m * theta)   if m >= 0
//   Z_n^m(rho, theta) = R_n^|m|(rho) * sin(|m| * theta) if m < 0
//
// where R_n^m(rho) is the radial Zernike polynomial.
//
// Special Values:
// - Z_n^m(0, theta) = 0 for |m| > 0
// - Z_n^0(0, theta) = R_n^0(0) = (-1)^(n/2) for n even
// - Z_n^m(1, theta) = cos(m*theta) for m >= 0, sin(|m|*theta) for m < 0
//
// Constraints (from R_n^m):
// - n >= |m| >= 0
// - (n - |m|) must be even
//
// For invalid (n, m) combinations (n < |m| or (n - |m|) odd), returns 0.
//
// Applications:
// - Optical aberration analysis (wavefront decomposition)
// - Adaptive optics for telescope and microscopy
// - Corneal topography in ophthalmology
// - Image analysis and pattern recognition
//
// Standard Zernike modes (OSA/ANSI indexing, without normalization):
// - Z_0^0 = 1 (piston)
// - Z_1^{-1} = rho * sin(theta) (y-tilt)
// - Z_1^1 = rho * cos(theta) (x-tilt)
// - Z_2^{-2} = rho^2 * sin(2*theta) (astigmatism 45 degrees)
// - Z_2^0 = 2*rho^2 - 1 (defocus)
// - Z_2^2 = rho^2 * cos(2*theta) (astigmatism 0 degrees)
// - Z_3^{-1} = (3*rho^3 - 2*rho) * sin(theta) (y-coma)
// - Z_3^1 = (3*rho^3 - 2*rho) * cos(theta) (x-coma)

template <typename T>
T zernike_polynomial_z(T n, T m, T rho, T theta) {
  // Compute the radial part R_n^|m|(rho)
  T abs_m = std::abs(m);
  T radial = zernike_polynomial_r(n, abs_m, rho);

  // Combine with angular part
  if (m >= T(0)) {
    return radial * std::cos(m * theta);
  } else {
    return radial * std::sin(abs_m * theta);
  }
}

// Complex version
template <typename T>
c10::complex<T> zernike_polynomial_z(c10::complex<T> n, c10::complex<T> m, c10::complex<T> rho, c10::complex<T> theta) {
  c10::complex<T> zero(T(0), T(0));

  // Take absolute value of m (real part for the check)
  T m_real = m.real();
  T abs_m_real = std::abs(m_real);
  c10::complex<T> abs_m(abs_m_real, T(0));

  // Compute the radial part R_n^|m|(rho)
  c10::complex<T> radial = zernike_polynomial_r(n, abs_m, rho);

  // Combine with angular part
  if (m_real >= T(0)) {
    return radial * std::cos(m * theta);
  } else {
    return radial * std::sin(abs_m * theta);
  }
}

} // namespace torchscience::kernel::special_functions
