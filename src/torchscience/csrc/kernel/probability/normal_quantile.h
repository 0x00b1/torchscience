#pragma once

#include <cmath>
#include <limits>

namespace torchscience::kernel::probability {

// Acklam's inverse normal approximation
// Reference: https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
template <typename T>
T standard_normal_quantile(T p) {
  // Edge cases
  if (p <= T(0)) return -std::numeric_limits<T>::infinity();
  if (p >= T(1)) return std::numeric_limits<T>::infinity();
  if (p == T(0.5)) return T(0);

  // Coefficients for rational approximation
  const T a1 = T(-3.969683028665376e+01);
  const T a2 = T(2.209460984245205e+02);
  const T a3 = T(-2.759285104469687e+02);
  const T a4 = T(1.383577518672690e+02);
  const T a5 = T(-3.066479806614716e+01);
  const T a6 = T(2.506628277459239e+00);

  const T b1 = T(-5.447609879822406e+01);
  const T b2 = T(1.615858368580409e+02);
  const T b3 = T(-1.556989798598866e+02);
  const T b4 = T(6.680131188771972e+01);
  const T b5 = T(-1.328068155288572e+01);

  const T c1 = T(-7.784894002430293e-03);
  const T c2 = T(-3.223964580411365e-01);
  const T c3 = T(-2.400758277161838e+00);
  const T c4 = T(-2.549732539343734e+00);
  const T c5 = T(4.374664141464968e+00);
  const T c6 = T(2.938163982698783e+00);

  const T d1 = T(7.784695709041462e-03);
  const T d2 = T(3.224671290700398e-01);
  const T d3 = T(2.445134137142996e+00);
  const T d4 = T(3.754408661907416e+00);

  const T p_low = T(0.02425);
  const T p_high = T(1) - p_low;

  T q, r, x;

  if (p < p_low) {
    // Lower tail
    q = std::sqrt(-T(2) * std::log(p));
    x = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
        ((((d1*q + d2)*q + d3)*q + d4)*q + T(1));
  } else if (p <= p_high) {
    // Central region
    q = p - T(0.5);
    r = q * q;
    x = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q /
        (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + T(1));
  } else {
    // Upper tail
    q = std::sqrt(-T(2) * std::log(T(1) - p));
    x = -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
         ((((d1*q + d2)*q + d3)*q + d4)*q + T(1));
  }

  // Newton-Raphson refinement
  const T inv_sqrt_2pi = T(0.3989422804014327);
  T pdf = inv_sqrt_2pi * std::exp(T(-0.5) * x * x);
  T err = T(0.5) * (T(1) + std::erf(x / T(1.4142135623730951))) - p;
  x = x - err / pdf;

  return x;
}

// Normal quantile function (percent point function)
template <typename T>
T normal_quantile(T p, T loc, T scale) {
  return loc + scale * standard_normal_quantile(p);
}

}  // namespace torchscience::kernel::probability
