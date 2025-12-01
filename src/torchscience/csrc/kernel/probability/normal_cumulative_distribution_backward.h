#pragma once

#include <cmath>

namespace torchscience::kernel::probability {

// Helper to get inv_sqrt_2pi = 1/sqrt(2*pi) for any type
template <typename T>
inline T inv_sqrt_2pi() {
  return T(0.3989422804014327);
}

// Standard normal PDF: phi(z) = exp(-z^2/2) / sqrt(2*pi)
template <typename T>
T standard_normal_probability_density(T z) {
  return inv_sqrt_2pi<T>() * std::exp(T(-0.5) * z * z);
}

// Gradients of normal_cumulative_distribution:
// dF/dx = phi(z) / sigma
// dF/dloc = -phi(z) / sigma
// dF/dscale = -z * phi(z) / sigma
template <typename T>
void normal_cumulative_distribution_backward(
    T grad_output, T x, T loc, T scale,
    T& grad_x, T& grad_loc, T& grad_scale) {
  T z = (x - loc) / scale;
  T pdf = standard_normal_probability_density(z);
  T pdf_over_scale = pdf / scale;

  grad_x = grad_output * pdf_over_scale;
  grad_loc = grad_output * (-pdf_over_scale);
  grad_scale = grad_output * (-z * pdf_over_scale);
}

}  // namespace torchscience::kernel::probability
