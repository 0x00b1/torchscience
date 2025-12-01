#pragma once

#include <cmath>

namespace torchscience::kernel::probability {

// Gradients of normal survival function
// SF(x; loc, scale) = 0.5 * erfc(z / sqrt(2)) where z = (x - loc) / scale
//
// d(SF)/dx = -pdf(x; loc, scale)
// d(SF)/dloc = pdf(x; loc, scale)
// d(SF)/dscale = z * pdf(x; loc, scale)
template <typename T>
inline T inv_sqrt_2pi_survival() {
  return T(0.3989422804014327);  // 1 / sqrt(2 * pi)
}

template <typename T>
void normal_survival_backward(
    T grad_output, T x, T loc, T scale,
    T& grad_x, T& grad_loc, T& grad_scale) {

  T z = (x - loc) / scale;
  T pdf = inv_sqrt_2pi_survival<T>() / scale * std::exp(T(-0.5) * z * z);

  // dSF/dx = -pdf
  grad_x = grad_output * (-pdf);
  // dSF/dloc = pdf
  grad_loc = grad_output * pdf;
  // dSF/dscale = z * pdf
  grad_scale = grad_output * z * pdf;
}

}  // namespace torchscience::kernel::probability
