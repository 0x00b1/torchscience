#pragma once

#include <cmath>

namespace torchscience::kernel::probability {

// Gradients of normal PDF
//
// PDF(x; loc, scale) = exp(-z^2/2) / (scale * sqrt(2*pi))
// where z = (x - loc) / scale
//
// dPDF/dx = -z * PDF / scale
// dPDF/dloc = z * PDF / scale (opposite sign to dx)
// dPDF/dscale = PDF * (z^2 - 1) / scale
template <typename T>
inline T inv_sqrt_2pi_bwd() {
  return T(0.3989422804014327);  // 1 / sqrt(2 * pi)
}

template <typename T>
void normal_probability_density_backward(
    T grad_output, T x, T loc, T scale,
    T& grad_x, T& grad_loc, T& grad_scale) {

  T z = (x - loc) / scale;

  // Compute PDF inline
  T pdf = inv_sqrt_2pi_bwd<T>() / scale * std::exp(T(-0.5) * z * z);

  T z_probability_density_over_scale = z * pdf / scale;

  grad_x = grad_output * (-z_probability_density_over_scale);
  grad_loc = grad_output * z_probability_density_over_scale;
  grad_scale = grad_output * pdf * (z * z - T(1)) / scale;
}

}  // namespace torchscience::kernel::probability
