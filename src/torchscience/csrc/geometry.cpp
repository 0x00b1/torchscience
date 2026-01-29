// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// Ray intersection - CPU
#include "cpu/geometry/ray_intersect.h"
#include "cpu/geometry/closest_point.h"
#include "cpu/geometry/ray_occluded.h"
#include "cpu/geometry/intersection/ray_plane.h"
#include "cpu/geometry/intersection/ray_sphere.h"
#include "cpu/geometry/intersection/ray_triangle.h"
#include "cpu/geometry/intersection/ray_aabb.h"

// Transforms - CPU
#include "cpu/geometry/transform/reflect.h"
#include "cpu/geometry/transform/refract.h"
#include "cpu/geometry/transform/quaternion_multiply.h"
#include "cpu/geometry/transform/quaternion_inverse.h"
#include "cpu/geometry/transform/quaternion_normalize.h"
#include "cpu/geometry/transform/quaternion_apply.h"
#include "cpu/geometry/transform/quaternion_to_matrix.h"
#include "cpu/geometry/transform/matrix_to_quaternion.h"
#include "cpu/geometry/transform/quaternion_slerp.h"

// Convex hull - CPU
#include "cpu/geometry/convex_hull.h"

// Meta backend
#include "meta/geometry/ray_intersect.h"
#include "meta/geometry/closest_point.h"
#include "meta/geometry/ray_occluded.h"
#include "meta/geometry/intersection/ray_plane.h"
#include "meta/geometry/intersection/ray_sphere.h"
#include "meta/geometry/intersection/ray_triangle.h"
#include "meta/geometry/intersection/ray_aabb.h"
#include "meta/geometry/transform/reflect.h"
#include "meta/geometry/transform/refract.h"
#include "meta/geometry/transform/quaternion_multiply.h"
#include "meta/geometry/transform/quaternion_inverse.h"
#include "meta/geometry/transform/quaternion_normalize.h"
#include "meta/geometry/transform/quaternion_apply.h"
#include "meta/geometry/transform/quaternion_to_matrix.h"
#include "meta/geometry/transform/matrix_to_quaternion.h"
#include "meta/geometry/transform/quaternion_slerp.h"
#include "meta/geometry/convex_hull.h"

// Autograd backend
#include "autograd/geometry/transform/reflect.h"
#include "autograd/geometry/transform/refract.h"
#include "autograd/geometry/transform/quaternion_multiply.h"
#include "autograd/geometry/transform/quaternion_inverse.h"
#include "autograd/geometry/transform/quaternion_normalize.h"
#include "autograd/geometry/transform/quaternion_apply.h"
#include "autograd/geometry/transform/quaternion_to_matrix.h"
#include "autograd/geometry/transform/matrix_to_quaternion.h"
#include "autograd/geometry/transform/quaternion_slerp.h"
#include "autograd/geometry/intersection/ray_plane.h"
#include "autograd/geometry/intersection/ray_sphere.h"
#include "autograd/geometry/intersection/ray_triangle.h"
#include "autograd/geometry/intersection/ray_aabb.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Ray-plane intersection
  m.def("ray_plane(Tensor origins, Tensor directions, Tensor plane_normals, Tensor plane_offsets) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("ray_plane_backward(Tensor grad_t, Tensor grad_hit_point, Tensor grad_normal, Tensor grad_uv, Tensor origins, Tensor directions, Tensor plane_normals, Tensor plane_offsets, Tensor t, Tensor hit) -> (Tensor, Tensor, Tensor, Tensor)");

  // Ray-sphere intersection
  m.def("ray_sphere(Tensor origins, Tensor directions, Tensor centers, Tensor radii) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("ray_sphere_backward(Tensor grad_t, Tensor grad_hit_point, Tensor grad_normal, Tensor grad_uv, Tensor origins, Tensor directions, Tensor centers, Tensor radii, Tensor t, Tensor hit) -> (Tensor, Tensor, Tensor, Tensor)");

  // Ray-triangle intersection (Möller-Trumbore)
  m.def("ray_triangle(Tensor origins, Tensor directions, Tensor v0, Tensor v1, Tensor v2) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("ray_triangle_backward(Tensor grad_t, Tensor grad_hit_point, Tensor grad_normal, Tensor grad_uv, Tensor origins, Tensor directions, Tensor v0, Tensor v1, Tensor v2, Tensor t, Tensor hit) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Ray-AABB intersection (slab method)
  m.def("ray_aabb(Tensor origins, Tensor directions, Tensor box_min, Tensor box_max) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("ray_aabb_backward(Tensor grad_t, Tensor grad_hit_point, Tensor grad_normal, Tensor grad_uv, Tensor origins, Tensor directions, Tensor box_min, Tensor box_max, Tensor t, Tensor hit) -> (Tensor, Tensor, Tensor, Tensor)");

  // Geometric transforms
  m.def("reflect(Tensor direction, Tensor normal) -> Tensor");
  m.def("reflect_backward(Tensor grad_output, Tensor direction, Tensor normal) -> (Tensor, Tensor)");

  m.def("refract(Tensor direction, Tensor normal, Tensor eta) -> Tensor");
  m.def("refract_backward(Tensor grad_output, Tensor direction, Tensor normal, Tensor eta) -> (Tensor, Tensor, Tensor)");

  // Quaternion operations
  m.def("quaternion_multiply(Tensor q1, Tensor q2) -> Tensor");
  m.def("quaternion_multiply_backward(Tensor grad_output, Tensor q1, Tensor q2) -> (Tensor, Tensor)");

  m.def("quaternion_inverse(Tensor q) -> Tensor");
  m.def("quaternion_inverse_backward(Tensor grad_output, Tensor q) -> Tensor");

  m.def("quaternion_normalize(Tensor q) -> Tensor");
  m.def("quaternion_normalize_backward(Tensor grad_output, Tensor q) -> Tensor");

  m.def("quaternion_apply(Tensor q, Tensor point) -> Tensor");
  m.def("quaternion_apply_backward(Tensor grad_output, Tensor q, Tensor point) -> (Tensor, Tensor)");

  m.def("quaternion_to_matrix(Tensor q) -> Tensor");
  m.def("quaternion_to_matrix_backward(Tensor grad_output, Tensor q) -> Tensor");

  m.def("matrix_to_quaternion(Tensor matrix) -> Tensor");
  m.def("matrix_to_quaternion_backward(Tensor grad_output, Tensor matrix) -> Tensor");

  m.def("quaternion_slerp(Tensor q1, Tensor q2, Tensor t) -> Tensor");
  m.def("quaternion_slerp_backward(Tensor grad_output, Tensor q1, Tensor q2, Tensor t) -> (Tensor, Tensor, Tensor)");

  // Convex hull
  m.def("convex_hull(Tensor points) -> "
        "(Tensor vertices, Tensor simplices, Tensor neighbors, "
        "Tensor equations, Tensor area, Tensor volume, "
        "Tensor n_vertices, Tensor n_facets)");
}
