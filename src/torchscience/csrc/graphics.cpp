// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// Shading - CPU
#include "cpu/graphics/shading/cook_torrance.h"
#include "cpu/graphics/shading/phong.h"
#include "cpu/graphics/shading/schlick_reflectance.h"

// Lighting - CPU
#include "cpu/graphics/lighting/spotlight.h"

// Tone mapping - CPU
#include "cpu/graphics/tone_mapping/reinhard.h"

// Texture mapping - CPU
#include "cpu/graphics/texture_mapping/cube_mapping.h"

// Projection - CPU
#include "cpu/graphics/projection/perspective_projection.h"

// Color - CPU
#include "cpu/graphics/color/srgb_to_hsv.h"
#include "cpu/graphics/color/hsv_to_srgb.h"
#include "cpu/graphics/color/srgb_to_hwb.h"
#include "cpu/graphics/color/hwb_to_srgb.h"
#include "cpu/graphics/color/srgb_to_hls.h"
#include "cpu/graphics/color/hls_to_srgb.h"
#include "cpu/graphics/color/srgb_to_srgb_linear.h"
#include "cpu/graphics/color/srgb_linear_to_srgb.h"
#include "cpu/graphics/color/srgb_to_xyz.h"
#include "cpu/graphics/color/xyz_to_srgb.h"
#include "cpu/graphics/color/srgb_to_lab.h"
#include "cpu/graphics/color/lab_to_srgb.h"
#include "cpu/graphics/color/srgb_to_luv.h"
#include "cpu/graphics/color/luv_to_srgb.h"
#include "cpu/graphics/color/srgb_to_lchab.h"
#include "cpu/graphics/color/lchab_to_srgb.h"
#include "cpu/graphics/color/srgb_to_lchuv.h"
#include "cpu/graphics/color/lchuv_to_srgb.h"
#include "cpu/graphics/color/srgb_to_oklab.h"
#include "cpu/graphics/color/oklab_to_srgb.h"
#include "cpu/graphics/color/srgb_to_oklch.h"
#include "cpu/graphics/color/oklch_to_srgb.h"
#include "cpu/graphics/color/srgb_to_ycbcr.h"
#include "cpu/graphics/color/ycbcr_to_srgb.h"
#include "cpu/graphics/color/srgb_to_ypbpr.h"
#include "cpu/graphics/color/ypbpr_to_srgb.h"
#include "cpu/graphics/color/srgb_to_yuv.h"
#include "cpu/graphics/color/yuv_to_srgb.h"

// Meta backend
#include "meta/graphics/shading/cook_torrance.h"
#include "meta/graphics/shading/phong.h"
#include "meta/graphics/shading/schlick_reflectance.h"
#include "meta/graphics/lighting/spotlight.h"
#include "meta/graphics/tone_mapping/reinhard.h"
#include "meta/graphics/texture_mapping/cube_mapping.h"
#include "meta/graphics/projection/perspective_projection.h"
#include "meta/graphics/color/srgb_to_hsv.h"
#include "meta/graphics/color/hsv_to_srgb.h"
#include "meta/graphics/color/srgb_to_hwb.h"
#include "meta/graphics/color/hwb_to_srgb.h"
#include "meta/graphics/color/srgb_to_hls.h"
#include "meta/graphics/color/hls_to_srgb.h"
#include "meta/graphics/color/srgb_to_srgb_linear.h"
#include "meta/graphics/color/srgb_linear_to_srgb.h"
#include "meta/graphics/color/srgb_to_xyz.h"
#include "meta/graphics/color/xyz_to_srgb.h"
#include "meta/graphics/color/srgb_to_lab.h"
#include "meta/graphics/color/lab_to_srgb.h"
#include "meta/graphics/color/srgb_to_luv.h"
#include "meta/graphics/color/luv_to_srgb.h"
#include "meta/graphics/color/srgb_to_lchab.h"
#include "meta/graphics/color/lchab_to_srgb.h"
#include "meta/graphics/color/srgb_to_lchuv.h"
#include "meta/graphics/color/lchuv_to_srgb.h"
#include "meta/graphics/color/srgb_to_oklab.h"
#include "meta/graphics/color/oklab_to_srgb.h"
#include "meta/graphics/color/srgb_to_oklch.h"
#include "meta/graphics/color/oklch_to_srgb.h"
#include "meta/graphics/color/srgb_to_ycbcr.h"
#include "meta/graphics/color/ycbcr_to_srgb.h"
#include "meta/graphics/color/srgb_to_ypbpr.h"
#include "meta/graphics/color/ypbpr_to_srgb.h"
#include "meta/graphics/color/srgb_to_yuv.h"
#include "meta/graphics/color/yuv_to_srgb.h"

// Autograd backend
#include "autograd/graphics/shading/cook_torrance.h"
#include "autograd/graphics/shading/phong.h"
#include "autograd/graphics/shading/schlick_reflectance.h"
#include "autograd/graphics/lighting/spotlight.h"
#include "autograd/graphics/tone_mapping/reinhard.h"
#include "autograd/graphics/projection/perspective_projection.h"
#include "autograd/graphics/color/srgb_to_hsv.h"
#include "autograd/graphics/color/hsv_to_srgb.h"
#include "autograd/graphics/color/srgb_to_hwb.h"
#include "autograd/graphics/color/hwb_to_srgb.h"
#include "autograd/graphics/color/srgb_to_hls.h"
#include "autograd/graphics/color/hls_to_srgb.h"
#include "autograd/graphics/color/srgb_to_srgb_linear.h"
#include "autograd/graphics/color/srgb_linear_to_srgb.h"
#include "autograd/graphics/color/srgb_to_xyz.h"
#include "autograd/graphics/color/xyz_to_srgb.h"
#include "autograd/graphics/color/srgb_to_lab.h"
#include "autograd/graphics/color/lab_to_srgb.h"
#include "autograd/graphics/color/srgb_to_luv.h"
#include "autograd/graphics/color/luv_to_srgb.h"
#include "autograd/graphics/color/srgb_to_lchab.h"
#include "autograd/graphics/color/lchab_to_srgb.h"
#include "autograd/graphics/color/srgb_to_lchuv.h"
#include "autograd/graphics/color/lchuv_to_srgb.h"
#include "autograd/graphics/color/srgb_to_oklab.h"
#include "autograd/graphics/color/oklab_to_srgb.h"
#include "autograd/graphics/color/srgb_to_oklch.h"
#include "autograd/graphics/color/oklch_to_srgb.h"
#include "autograd/graphics/color/srgb_to_ycbcr.h"
#include "autograd/graphics/color/ycbcr_to_srgb.h"
#include "autograd/graphics/color/srgb_to_ypbpr.h"
#include "autograd/graphics/color/ypbpr_to_srgb.h"
#include "autograd/graphics/color/srgb_to_yuv.h"
#include "autograd/graphics/color/yuv_to_srgb.h"

#ifdef TORCHSCIENCE_CUDA
#include "cuda/graphics/shading/cook_torrance.cu"
#endif

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Shading
  m.def("cook_torrance(Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> Tensor");
  m.def("cook_torrance_backward(Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("cook_torrance_backward_backward(Tensor gg_normal, Tensor gg_view, Tensor gg_light, Tensor gg_roughness, Tensor gg_f0, Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  m.def("phong(Tensor normal, Tensor view, Tensor light, Tensor shininess) -> Tensor");
  m.def("phong_backward(Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor shininess) -> (Tensor, Tensor, Tensor, Tensor)");

  m.def("schlick_reflectance(Tensor cosine, Tensor r0) -> Tensor");
  m.def("schlick_reflectance_backward(Tensor grad_output, Tensor cosine, Tensor r0) -> Tensor");

  // Lighting
  m.def("spotlight(Tensor light_pos, Tensor surface_pos, Tensor spot_direction, Tensor intensity, Tensor inner_angle, Tensor outer_angle) -> (Tensor, Tensor)");
  m.def("spotlight_backward(Tensor grad_irradiance, Tensor light_pos, Tensor surface_pos, Tensor spot_direction, Tensor intensity, Tensor inner_angle, Tensor outer_angle) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Tone mapping
  m.def("reinhard(Tensor input, Tensor? white_point) -> Tensor");
  m.def("reinhard_backward(Tensor grad_output, Tensor input, Tensor? white_point) -> (Tensor, Tensor)");

  // Texture mapping
  m.def("cube_mapping(Tensor direction) -> (Tensor, Tensor, Tensor)");

  // Projection
  m.def("perspective_projection(Tensor fov, Tensor aspect, Tensor near, Tensor far) -> Tensor");
  m.def("perspective_projection_backward(Tensor grad_output, Tensor fov, Tensor aspect, Tensor near, Tensor far) -> (Tensor, Tensor, Tensor, Tensor)");

  // Color conversions
  m.def("srgb_to_hsv(Tensor input) -> Tensor");
  m.def("srgb_to_hsv_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("hsv_to_srgb(Tensor input) -> Tensor");
  m.def("hsv_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("srgb_to_hwb(Tensor input) -> Tensor");
  m.def("srgb_to_hwb_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("hwb_to_srgb(Tensor input) -> Tensor");
  m.def("hwb_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("srgb_to_hls(Tensor input) -> Tensor");
  m.def("srgb_to_hls_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("hls_to_srgb(Tensor input) -> Tensor");
  m.def("hls_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("srgb_to_srgb_linear(Tensor input) -> Tensor");
  m.def("srgb_to_srgb_linear_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("srgb_linear_to_srgb(Tensor input) -> Tensor");
  m.def("srgb_linear_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("srgb_to_xyz(Tensor input) -> Tensor");
  m.def("srgb_to_xyz_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("xyz_to_srgb(Tensor input) -> Tensor");
  m.def("xyz_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("srgb_to_lab(Tensor input) -> Tensor");
  m.def("srgb_to_lab_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("lab_to_srgb(Tensor input) -> Tensor");
  m.def("lab_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("srgb_to_luv(Tensor input) -> Tensor");
  m.def("srgb_to_luv_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("luv_to_srgb(Tensor input) -> Tensor");
  m.def("luv_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("srgb_to_lchab(Tensor input) -> Tensor");
  m.def("srgb_to_lchab_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("lchab_to_srgb(Tensor input) -> Tensor");
  m.def("lchab_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("srgb_to_lchuv(Tensor input) -> Tensor");
  m.def("srgb_to_lchuv_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("lchuv_to_srgb(Tensor input) -> Tensor");
  m.def("lchuv_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("srgb_to_oklab(Tensor input) -> Tensor");
  m.def("srgb_to_oklab_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("oklab_to_srgb(Tensor input) -> Tensor");
  m.def("oklab_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("srgb_to_oklch(Tensor input) -> Tensor");
  m.def("srgb_to_oklch_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("oklch_to_srgb(Tensor input) -> Tensor");
  m.def("oklch_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("srgb_to_ycbcr(Tensor input) -> Tensor");
  m.def("srgb_to_ycbcr_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("ycbcr_to_srgb(Tensor input) -> Tensor");
  m.def("ycbcr_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("srgb_to_ypbpr(Tensor input) -> Tensor");
  m.def("srgb_to_ypbpr_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("ypbpr_to_srgb(Tensor input) -> Tensor");
  m.def("ypbpr_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("srgb_to_yuv(Tensor input) -> Tensor");
  m.def("srgb_to_yuv_backward(Tensor grad_output, Tensor input) -> Tensor");

  m.def("yuv_to_srgb(Tensor input) -> Tensor");
  m.def("yuv_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");
}
