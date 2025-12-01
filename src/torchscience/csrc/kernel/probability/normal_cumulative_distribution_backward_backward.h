#pragma once

#include <cmath>
#include "normal_cumulative_distribution_backward.h"

namespace torchscience::kernel::probability {

// Second derivatives of normal_cumulative_distribution
//
// Let z = (x - loc) / scale, phi = exp(-z^2/2) / sqrt(2*pi)
// First derivatives:
//   grad_x = g * phi / scale
//   grad_loc = g * (-phi / scale)
//   grad_scale = g * (-z * phi / scale)
//
// Where g = grad_output (upstream gradient)
//
// For second-order gradients, we need derivatives of the backward outputs
// w.r.t. x, loc, scale, and grad_output.
//
// Key derivative: d(phi)/dz = -z * phi
// So d(phi/scale)/dx = d(phi/scale)/dz * dz/dx = (-z * phi / scale) * (1/scale) = -z * phi / scale^2
template <typename T>
void normal_cumulative_distribution_backward_backward(
    T gg_x, T gg_loc, T gg_scale,  // grad of grad w.r.t. x, loc, scale
    T grad, T x, T loc, T scale,   // saved inputs
    T& out_grad, T& out_x, T& out_loc, T& out_scale) {  // outputs

  T z = (x - loc) / scale;
  T phi = standard_normal_probability_density(z);
  T phi_over_scale = phi / scale;
  T phi_over_scale_sq = phi / (scale * scale);

  // d(phi/scale)/dz = -z * phi / scale^2
  // dz/dx = 1/scale, dz/dloc = -1/scale, dz/dscale = -z/scale

  // grad_x = grad * phi / scale
  // d(grad_x)/dx = grad * d(phi/scale)/dx = grad * (-z * phi / scale^2) * (1/scale) = -grad * z * phi / scale^3
  // d(grad_x)/dloc = grad * (-z * phi / scale^2) * (-1/scale) = grad * z * phi / scale^3
  // d(grad_x)/dscale = grad * (d(phi)/dscale / scale - phi / scale^2)
  //                  = grad * ((-z * phi / scale) * (-z/scale) / scale - phi / scale^2)
  //                  = grad * (z^2 * phi / scale^3 - phi / scale^2)
  //                  = grad * phi / scale^2 * (z^2 / scale - 1)
  // d(grad_x)/dgrad = phi / scale

  T neg_z_phi_over_scale_cubed = -z * phi / (scale * scale * scale);

  // Contributions to output gradients:
  // out_grad: sum of gg_* * d(grad_*)/dgrad
  out_grad = gg_x * phi_over_scale
           + gg_loc * (-phi_over_scale)
           + gg_scale * (-z * phi_over_scale);

  // out_x: sum of gg_* * d(grad_*)/dx
  // d(grad_x)/dx = -z * phi / scale^3
  // d(grad_loc)/dx = z * phi / scale^3 (same magnitude, opposite sign to grad_x)
  // d(grad_scale)/dx = d(-z * phi / scale)/dx
  //                  = -phi/scale * dz/dx + (-z) * d(phi/scale)/dx
  //                  = -phi/scale^2 + (-z) * (-z * phi / scale^3)
  //                  = -phi/scale^2 + z^2 * phi / scale^3
  //                  = phi/scale^2 * (z^2/scale - 1)
  T d_grad_x_dx = grad * neg_z_phi_over_scale_cubed;
  T d_grad_loc_dx = -d_grad_x_dx;  // opposite sign
  T d_grad_scale_dx = grad * phi_over_scale_sq * (z * z / scale - T(1));

  out_x = gg_x * d_grad_x_dx + gg_loc * d_grad_loc_dx + gg_scale * d_grad_scale_dx;

  // out_loc: sum of gg_* * d(grad_*)/dloc
  // By symmetry with x, dloc has opposite effect on z
  T d_grad_x_dloc = -d_grad_x_dx;  // dz/dloc = -1/scale, so opposite sign
  T d_grad_loc_dloc = d_grad_x_dx;  // = -d_grad_loc_dx
  T d_grad_scale_dloc = -d_grad_scale_dx;  // opposite effect on z

  out_loc = gg_x * d_grad_x_dloc + gg_loc * d_grad_loc_dloc + gg_scale * d_grad_scale_dloc;

  // out_scale: sum of gg_* * d(grad_*)/dscale
  // This is more complex. Let's compute carefully.
  // grad_x = g * phi / scale
  // d(grad_x)/dscale = g * (d(phi)/dscale / scale - phi / scale^2)
  //                  = g * ((-z * phi / scale) * (-z/scale) / scale - phi / scale^2)
  //                  = g * (z^2 * phi / scale^3 - phi / scale^2)
  //                  = g * phi / scale^2 * (z^2/scale - 1)
  T d_grad_x_dscale = grad * phi_over_scale_sq * (z * z / scale - T(1));

  // grad_loc = -g * phi / scale
  // d(grad_loc)/dscale = -d(grad_x)/dscale
  T d_grad_loc_dscale = -d_grad_x_dscale;

  // grad_scale = -g * z * phi / scale
  // d(grad_scale)/dscale = -g * (dz/dscale * phi / scale + z * d(phi/scale)/dscale)
  // dz/dscale = -z/scale
  // d(phi/scale)/dscale = d(phi)/dscale / scale - phi / scale^2
  //                     = (-z * phi / scale) * (-z/scale) / scale - phi / scale^2
  //                     = z^2 * phi / scale^3 - phi / scale^2
  // So:
  // d(grad_scale)/dscale = -g * ((-z/scale) * phi/scale + z * (z^2 * phi / scale^3 - phi / scale^2))
  //                      = -g * (-z * phi / scale^2 + z^3 * phi / scale^3 - z * phi / scale^2)
  //                      = -g * (z^3 * phi / scale^3 - 2 * z * phi / scale^2)
  //                      = g * phi / scale^2 * (2 * z - z^3 / scale)
  T d_grad_scale_dscale = grad * phi_over_scale_sq * (T(2) * z - z * z * z / scale);

  out_scale = gg_x * d_grad_x_dscale + gg_loc * d_grad_loc_dscale + gg_scale * d_grad_scale_dscale;
}

}  // namespace torchscience::kernel::probability
