#pragma once

#include <cmath>
#include <limits>

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace torchscience::kernel::geometry::intersection {

/**
 * Backward pass for ray-AABB intersection.
 *
 * The intersection parameter t is determined by one specific slab (axis):
 *   t = (box_face - origin[axis]) / direction[axis]
 * where box_face is either box_min[axis] or box_max[axis].
 *
 * Gradients for t (simple quotient rule):
 *   dt/d_origin[axis] = -1 / direction[axis]
 *   dt/d_direction[axis] = -t / direction[axis]
 *   dt/d_box_face = 1 / direction[axis]
 *
 * Only the determining axis contributes gradient through t; other axes have
 * zero gradient contribution.
 *
 * For hit_point = O + t*D:
 *   d_hit/d_O = I + D * dt/dO
 *   d_hit/d_D = t*I + D * dt/dD
 *   d_hit/d_box = D * dt/d_box
 *
 * Normal and UV gradients are not propagated (discrete/non-differentiable).
 *
 * @param grad_t Upstream gradient for intersection parameter t
 * @param grad_hit_x Upstream gradient for hit point x coordinate
 * @param grad_hit_y Upstream gradient for hit point y coordinate
 * @param grad_hit_z Upstream gradient for hit point z coordinate
 * @param grad_normal_x Upstream gradient for surface normal x coordinate
 * @param grad_normal_y Upstream gradient for surface normal y coordinate
 * @param grad_normal_z Upstream gradient for surface normal z coordinate
 * @param grad_uv_u Upstream gradient for UV coordinate u
 * @param grad_uv_v Upstream gradient for UV coordinate v
 * @param ray_origin_x Ray origin x coordinate (saved from forward)
 * @param ray_origin_y Ray origin y coordinate (saved from forward)
 * @param ray_origin_z Ray origin z coordinate (saved from forward)
 * @param ray_dir_x Ray direction x coordinate (saved from forward)
 * @param ray_dir_y Ray direction y coordinate (saved from forward)
 * @param ray_dir_z Ray direction z coordinate (saved from forward)
 * @param box_min_x AABB minimum corner x coordinate (saved from forward)
 * @param box_min_y AABB minimum corner y coordinate (saved from forward)
 * @param box_min_z AABB minimum corner z coordinate (saved from forward)
 * @param box_max_x AABB maximum corner x coordinate (saved from forward)
 * @param box_max_y AABB maximum corner y coordinate (saved from forward)
 * @param box_max_z AABB maximum corner z coordinate (saved from forward)
 * @param t Intersection parameter t (saved from forward)
 * @param hit Hit flag (saved from forward)
 * @param grad_origin_x Output gradient for ray origin x
 * @param grad_origin_y Output gradient for ray origin y
 * @param grad_origin_z Output gradient for ray origin z
 * @param grad_dir_x Output gradient for ray direction x
 * @param grad_dir_y Output gradient for ray direction y
 * @param grad_dir_z Output gradient for ray direction z
 * @param grad_box_min_x Output gradient for AABB minimum corner x
 * @param grad_box_min_y Output gradient for AABB minimum corner y
 * @param grad_box_min_z Output gradient for AABB minimum corner z
 * @param grad_box_max_x Output gradient for AABB maximum corner x
 * @param grad_box_max_y Output gradient for AABB maximum corner y
 * @param grad_box_max_z Output gradient for AABB maximum corner z
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void ray_aabb_backward_kernel(
    T grad_t,
    T grad_hit_x,
    T grad_hit_y,
    T grad_hit_z,
    T grad_normal_x,
    T grad_normal_y,
    T grad_normal_z,
    T grad_uv_u,
    T grad_uv_v,
    T ray_origin_x,
    T ray_origin_y,
    T ray_origin_z,
    T ray_dir_x,
    T ray_dir_y,
    T ray_dir_z,
    T box_min_x,
    T box_min_y,
    T box_min_z,
    T box_max_x,
    T box_max_y,
    T box_max_z,
    T t,
    T hit,
    T& grad_origin_x,
    T& grad_origin_y,
    T& grad_origin_z,
    T& grad_dir_x,
    T& grad_dir_y,
    T& grad_dir_z,
    T& grad_box_min_x,
    T& grad_box_min_y,
    T& grad_box_min_z,
    T& grad_box_max_x,
    T& grad_box_max_y,
    T& grad_box_max_z) {
  // If no hit, all gradients are zero
  if (hit == T(0)) {
    grad_origin_x = T(0);
    grad_origin_y = T(0);
    grad_origin_z = T(0);
    grad_dir_x = T(0);
    grad_dir_y = T(0);
    grad_dir_z = T(0);
    grad_box_min_x = T(0);
    grad_box_min_y = T(0);
    grad_box_min_z = T(0);
    grad_box_max_x = T(0);
    grad_box_max_y = T(0);
    grad_box_max_z = T(0);
    return;
  }

  const T inf = std::numeric_limits<T>::infinity();
  const T eps = T(1e-8);

  // Recompute slab intervals to determine which axis/face produced t
  T origin[3] = {ray_origin_x, ray_origin_y, ray_origin_z};
  T dir[3] = {ray_dir_x, ray_dir_y, ray_dir_z};
  T bmin[3] = {box_min_x, box_min_y, box_min_z};
  T bmax[3] = {box_max_x, box_max_y, box_max_z};

  T t_near = -inf;
  T t_far = inf;

  int near_axis = 0;
  int near_face = 0;  // 0 = min face, 1 = max face

  for (int i = 0; i < 3; ++i) {
    T inv_d = T(1) / (dir[i] + (dir[i] == T(0) ? eps : T(0)));

    T t1 = (bmin[i] - origin[i]) * inv_d;
    T t2 = (bmax[i] - origin[i]) * inv_d;

    int face = 0;
    if (inv_d < T(0)) {
      T tmp = t1;
      t1 = t2;
      t2 = tmp;
      face = 1;
    }

    if (t1 > t_near) {
      t_near = t1;
      near_axis = i;
      near_face = (inv_d < T(0)) ? 1 : 0;
    }

    if (t2 < t_far) {
      t_far = t2;
    }
  }

  // Determine the axis and face that produced the selected t
  int hit_axis;
  int hit_face;

  if (t_near >= T(0)) {
    hit_axis = near_axis;
    hit_face = near_face;
  } else {
    // Ray inside box: t = t_far. Find which axis produced t_far.
    hit_axis = 0;
    hit_face = 0;

    for (int i = 0; i < 3; ++i) {
      T inv_d = T(1) / (dir[i] + (dir[i] == T(0) ? eps : T(0)));

      T t1 = (bmin[i] - origin[i]) * inv_d;
      T t2 = (bmax[i] - origin[i]) * inv_d;

      if (inv_d < T(0)) {
        T tmp = t1;
        t1 = t2;
        t2 = tmp;
      }

      T diff = t2 - t_far;
      if (diff < T(0))
        diff = -diff;
      if (diff < eps) {
        hit_axis = i;
        hit_face = (inv_d < T(0)) ? 0 : 1;
        break;
      }
    }
  }

  // The determining slab equation:
  //   t = (box_face - origin[hit_axis]) / dir[hit_axis]
  // where box_face = bmin[hit_axis] if hit_face == 0
  //                  bmax[hit_axis] if hit_face == 1

  T d_axis = dir[hit_axis] + (dir[hit_axis] == T(0) ? eps : T(0));
  T inv_d_axis = T(1) / d_axis;

  // dt/d_origin[hit_axis] = -1 / dir[hit_axis]
  T dt_d_origin_axis = -inv_d_axis;

  // dt/d_direction[hit_axis] = -t / dir[hit_axis]
  T dt_d_dir_axis = -t * inv_d_axis;

  // dt/d_box_face = 1 / dir[hit_axis]
  T dt_d_box_face = inv_d_axis;

  // Total gradient flowing through t:
  // hit_point = O + t*D, so grad from hit_point contributes D . grad_hit
  T total_grad_t = grad_t;
  total_grad_t +=
      grad_hit_x * ray_dir_x + grad_hit_y * ray_dir_y + grad_hit_z * ray_dir_z;

  // Gradient w.r.t. ray origin (O)
  // From hit_point = O + t*D:
  //   d_hit/d_O_i = delta_i + D_i * dt/d_O_i
  // dt/d_O_i is nonzero only for i == hit_axis
  grad_origin_x = grad_hit_x;
  grad_origin_y = grad_hit_y;
  grad_origin_z = grad_hit_z;

  // Add contribution through t for the determining axis
  T dt_origin_contrib = total_grad_t * dt_d_origin_axis;
  if (hit_axis == 0) {
    grad_origin_x += dt_origin_contrib;
  } else if (hit_axis == 1) {
    grad_origin_y += dt_origin_contrib;
  } else {
    grad_origin_z += dt_origin_contrib;
  }

  // Gradient w.r.t. ray direction (D)
  // From hit_point = O + t*D:
  //   d_hit/d_D_i = t * delta_i + D_i * dt/d_D_i
  // dt/d_D_i is nonzero only for i == hit_axis
  grad_dir_x = t * grad_hit_x;
  grad_dir_y = t * grad_hit_y;
  grad_dir_z = t * grad_hit_z;

  T dt_dir_contrib = total_grad_t * dt_d_dir_axis;
  if (hit_axis == 0) {
    grad_dir_x += dt_dir_contrib;
  } else if (hit_axis == 1) {
    grad_dir_y += dt_dir_contrib;
  } else {
    grad_dir_z += dt_dir_contrib;
  }

  // Gradient w.r.t. box_min and box_max
  // Only the face that determined t gets a gradient
  grad_box_min_x = T(0);
  grad_box_min_y = T(0);
  grad_box_min_z = T(0);
  grad_box_max_x = T(0);
  grad_box_max_y = T(0);
  grad_box_max_z = T(0);

  // dt/d_box_face flows through hit_point as well:
  // d_hit/d_box_face = D * dt/d_box_face (all components of D)
  // But total_grad_t already includes the hit_point contribution via D.grad_hit,
  // so we just need total_grad_t * dt_d_box_face
  T box_grad = total_grad_t * dt_d_box_face;

  if (hit_face == 0) {
    // min face
    if (hit_axis == 0) {
      grad_box_min_x = box_grad;
    } else if (hit_axis == 1) {
      grad_box_min_y = box_grad;
    } else {
      grad_box_min_z = box_grad;
    }
  } else {
    // max face
    if (hit_axis == 0) {
      grad_box_max_x = box_grad;
    } else if (hit_axis == 1) {
      grad_box_max_y = box_grad;
    } else {
      grad_box_max_z = box_grad;
    }
  }
}

}  // namespace torchscience::kernel::geometry::intersection
