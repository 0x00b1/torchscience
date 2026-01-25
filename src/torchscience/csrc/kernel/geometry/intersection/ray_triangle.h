#pragma once

#include <cmath>
#include <limits>

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace torchscience::kernel::geometry::intersection {

/**
 * Ray-triangle intersection kernel using the Moller-Trumbore algorithm.
 *
 * Given ray origin O, direction D, and triangle vertices V0, V1, V2,
 * computes the intersection using:
 *
 *   edge1 = V1 - V0
 *   edge2 = V2 - V0
 *   h = D x edge2
 *   a = edge1 . h
 *
 *   If |a| < epsilon: ray is parallel to triangle (miss)
 *
 *   f = 1/a
 *   s = O - V0
 *   u = f * (s . h)
 *
 *   If u < 0 or u > 1: miss
 *
 *   q = s x edge1
 *   v = f * (D . q)
 *
 *   If v < 0 or u + v > 1: miss
 *
 *   t = f * (edge2 . q)
 *
 *   If t < epsilon: intersection is behind ray (miss)
 *
 * Outputs:
 *   t: intersection parameter
 *   hit_point: O + t*D
 *   normal: normalize(edge1 x edge2) - geometric normal of the triangle face
 *   uv: (u, v) barycentric coordinates where P = (1-u-v)*V0 + u*V1 + v*V2
 *   hit: 1 if valid hit, 0 otherwise
 *
 * @param ray_origin_x Ray origin x coordinate
 * @param ray_origin_y Ray origin y coordinate
 * @param ray_origin_z Ray origin z coordinate
 * @param ray_dir_x Ray direction x coordinate
 * @param ray_dir_y Ray direction y coordinate
 * @param ray_dir_z Ray direction z coordinate
 * @param v0_x Triangle vertex V0 x coordinate
 * @param v0_y Triangle vertex V0 y coordinate
 * @param v0_z Triangle vertex V0 z coordinate
 * @param v1_x Triangle vertex V1 x coordinate
 * @param v1_y Triangle vertex V1 y coordinate
 * @param v1_z Triangle vertex V1 z coordinate
 * @param v2_x Triangle vertex V2 x coordinate
 * @param v2_y Triangle vertex V2 y coordinate
 * @param v2_z Triangle vertex V2 z coordinate
 * @param out_t Output intersection parameter t
 * @param out_hit_x Output hit point x coordinate
 * @param out_hit_y Output hit point y coordinate
 * @param out_hit_z Output hit point z coordinate
 * @param out_normal_x Output surface normal x coordinate
 * @param out_normal_y Output surface normal y coordinate
 * @param out_normal_z Output surface normal z coordinate
 * @param out_uv_u Output barycentric coordinate u
 * @param out_uv_v Output barycentric coordinate v
 * @param out_hit Output hit flag (1 if hit, 0 otherwise)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void ray_triangle_kernel(
    T ray_origin_x,
    T ray_origin_y,
    T ray_origin_z,
    T ray_dir_x,
    T ray_dir_y,
    T ray_dir_z,
    T v0_x,
    T v0_y,
    T v0_z,
    T v1_x,
    T v1_y,
    T v1_z,
    T v2_x,
    T v2_y,
    T v2_z,
    T& out_t,
    T& out_hit_x,
    T& out_hit_y,
    T& out_hit_z,
    T& out_normal_x,
    T& out_normal_y,
    T& out_normal_z,
    T& out_uv_u,
    T& out_uv_v,
    T& out_hit) {
  const T eps = T(1e-8);
  const T inf = std::numeric_limits<T>::infinity();

  // Edge vectors
  T edge1_x = v1_x - v0_x;
  T edge1_y = v1_y - v0_y;
  T edge1_z = v1_z - v0_z;

  T edge2_x = v2_x - v0_x;
  T edge2_y = v2_y - v0_y;
  T edge2_z = v2_z - v0_z;

  // h = D x edge2 (cross product)
  T h_x = ray_dir_y * edge2_z - ray_dir_z * edge2_y;
  T h_y = ray_dir_z * edge2_x - ray_dir_x * edge2_z;
  T h_z = ray_dir_x * edge2_y - ray_dir_y * edge2_x;

  // a = edge1 . h (determinant)
  T a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z;

  // If |a| < epsilon, ray is parallel to triangle (miss)
  if (a > -eps && a < eps) {
    out_t = inf;
    out_hit_x = T(0);
    out_hit_y = T(0);
    out_hit_z = T(0);
    out_normal_x = T(0);
    out_normal_y = T(0);
    out_normal_z = T(0);
    out_uv_u = T(0);
    out_uv_v = T(0);
    out_hit = T(0);
    return;
  }

  T f = T(1) / a;

  // s = O - V0
  T s_x = ray_origin_x - v0_x;
  T s_y = ray_origin_y - v0_y;
  T s_z = ray_origin_z - v0_z;

  // u = f * (s . h)
  T u = f * (s_x * h_x + s_y * h_y + s_z * h_z);

  // If u < 0 or u > 1: miss
  if (u < T(0) || u > T(1)) {
    out_t = inf;
    out_hit_x = T(0);
    out_hit_y = T(0);
    out_hit_z = T(0);
    out_normal_x = T(0);
    out_normal_y = T(0);
    out_normal_z = T(0);
    out_uv_u = T(0);
    out_uv_v = T(0);
    out_hit = T(0);
    return;
  }

  // q = s x edge1 (cross product)
  T q_x = s_y * edge1_z - s_z * edge1_y;
  T q_y = s_z * edge1_x - s_x * edge1_z;
  T q_z = s_x * edge1_y - s_y * edge1_x;

  // v = f * (D . q)
  T v = f * (ray_dir_x * q_x + ray_dir_y * q_y + ray_dir_z * q_z);

  // If v < 0 or u + v > 1: miss
  if (v < T(0) || u + v > T(1)) {
    out_t = inf;
    out_hit_x = T(0);
    out_hit_y = T(0);
    out_hit_z = T(0);
    out_normal_x = T(0);
    out_normal_y = T(0);
    out_normal_z = T(0);
    out_uv_u = T(0);
    out_uv_v = T(0);
    out_hit = T(0);
    return;
  }

  // t = f * (edge2 . q)
  T t = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z);

  // If t < epsilon: intersection is behind ray (miss)
  if (t < eps) {
    out_t = inf;
    out_hit_x = T(0);
    out_hit_y = T(0);
    out_hit_z = T(0);
    out_normal_x = T(0);
    out_normal_y = T(0);
    out_normal_z = T(0);
    out_uv_u = T(0);
    out_uv_v = T(0);
    out_hit = T(0);
    return;
  }

  // Hit point: O + t*D
  T hit_x = ray_origin_x + t * ray_dir_x;
  T hit_y = ray_origin_y + t * ray_dir_y;
  T hit_z = ray_origin_z + t * ray_dir_z;

  // Geometric normal: normalize(edge1 x edge2)
  T normal_x = edge1_y * edge2_z - edge1_z * edge2_y;
  T normal_y = edge1_z * edge2_x - edge1_x * edge2_z;
  T normal_z = edge1_x * edge2_y - edge1_y * edge2_x;

  T normal_len = std::sqrt(
      normal_x * normal_x + normal_y * normal_y + normal_z * normal_z);
  T inv_len = T(1) / (normal_len + eps);

  normal_x = normal_x * inv_len;
  normal_y = normal_y * inv_len;
  normal_z = normal_z * inv_len;

  out_t = t;
  out_hit_x = hit_x;
  out_hit_y = hit_y;
  out_hit_z = hit_z;
  out_normal_x = normal_x;
  out_normal_y = normal_y;
  out_normal_z = normal_z;
  out_uv_u = u;
  out_uv_v = v;
  out_hit = T(1);
}

}  // namespace torchscience::kernel::geometry::intersection
