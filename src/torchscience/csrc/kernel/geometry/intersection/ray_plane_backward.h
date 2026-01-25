#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace torchscience::kernel::geometry::intersection {

/**
 * Backward pass for ray-plane intersection.
 *
 * Computes gradients of t, hit_point, normal, uv w.r.t. inputs.
 *
 * Key derivatives:
 *   t = (d - n . o) / (n . dir)
 *   dt/do = -n / (n . dir)
 *   dt/ddir = -t * n / (n . dir)
 *   dt/dn = (o - t * dir) / (n . dir) = (o - hit) / (n . dir)
 *   dt/dd = 1 / (n . dir)
 *
 *   hit = o + t * dir
 *   dhit/do = I + dir * dt/do
 *   dhit/ddir = t * I + dir * dt/ddir
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
 * @param plane_normal_x Plane normal x coordinate (saved from forward)
 * @param plane_normal_y Plane normal y coordinate (saved from forward)
 * @param plane_normal_z Plane normal z coordinate (saved from forward)
 * @param plane_offset Plane offset (saved from forward)
 * @param t Intersection parameter t (saved from forward)
 * @param hit Hit flag (saved from forward)
 * @param grad_origin_x Output gradient for ray origin x
 * @param grad_origin_y Output gradient for ray origin y
 * @param grad_origin_z Output gradient for ray origin z
 * @param grad_dir_x Output gradient for ray direction x
 * @param grad_dir_y Output gradient for ray direction y
 * @param grad_dir_z Output gradient for ray direction z
 * @param grad_plane_normal_x Output gradient for plane normal x
 * @param grad_plane_normal_y Output gradient for plane normal y
 * @param grad_plane_normal_z Output gradient for plane normal z
 * @param grad_plane_offset Output gradient for plane offset
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void ray_plane_backward_kernel(
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
    T plane_normal_x,
    T plane_normal_y,
    T plane_normal_z,
    T plane_offset,
    T t,
    T hit,
    T& grad_origin_x,
    T& grad_origin_y,
    T& grad_origin_z,
    T& grad_dir_x,
    T& grad_dir_y,
    T& grad_dir_z,
    T& grad_plane_normal_x,
    T& grad_plane_normal_y,
    T& grad_plane_normal_z,
    T& grad_plane_offset) {
  // If no hit, all gradients are zero
  if (hit == T(0)) {
    grad_origin_x = T(0);
    grad_origin_y = T(0);
    grad_origin_z = T(0);
    grad_dir_x = T(0);
    grad_dir_y = T(0);
    grad_dir_z = T(0);
    grad_plane_normal_x = T(0);
    grad_plane_normal_y = T(0);
    grad_plane_normal_z = T(0);
    grad_plane_offset = T(0);
    return;
  }

  const T eps = T(1e-8);

  // Compute n . dir
  T n_dot_dir = plane_normal_x * ray_dir_x + plane_normal_y * ray_dir_y +
                plane_normal_z * ray_dir_z;

  T inv_n_dot_dir = T(1) / (n_dot_dir + eps);

  // Gradient of t w.r.t. inputs:
  // dt/do = -n / (n . dir)
  T dt_do_x = -plane_normal_x * inv_n_dot_dir;
  T dt_do_y = -plane_normal_y * inv_n_dot_dir;
  T dt_do_z = -plane_normal_z * inv_n_dot_dir;

  // dt/ddir = -t * n / (n . dir)
  T dt_ddir_x = -t * plane_normal_x * inv_n_dot_dir;
  T dt_ddir_y = -t * plane_normal_y * inv_n_dot_dir;
  T dt_ddir_z = -t * plane_normal_z * inv_n_dot_dir;

  // dt/dd = 1 / (n . dir)
  T dt_dd = inv_n_dot_dir;

  // Compute hit point (needed for dt/dn)
  T hit_x = ray_origin_x + t * ray_dir_x;
  T hit_y = ray_origin_y + t * ray_dir_y;
  T hit_z = ray_origin_z + t * ray_dir_z;

  // dt/dn = (o - hit) / (n . dir) = -t * dir / (n . dir)
  T dt_dn_x = (ray_origin_x - hit_x) * inv_n_dot_dir;
  T dt_dn_y = (ray_origin_y - hit_y) * inv_n_dot_dir;
  T dt_dn_z = (ray_origin_z - hit_z) * inv_n_dot_dir;

  // Gradient from hit_point: hit = o + t * dir
  // dhit/do = I + dir * dt/do
  // dhit/ddir = t * I + dir * dt/ddir
  // dhit/dn = dir * dt/dn
  // dhit/dd = dir * dt/dd

  // UV coordinates are uv_u = hit_x, uv_v = hit_y
  // So grad from UV adds to hit gradient
  T total_grad_hit_x = grad_hit_x + grad_uv_u;
  T total_grad_hit_y = grad_hit_y + grad_uv_v;
  T total_grad_hit_z = grad_hit_z;

  // Total contribution to t gradient from hit point gradient
  // d(loss)/dt via hit = grad_hit . dir
  T grad_t_from_hit = total_grad_hit_x * ray_dir_x +
                      total_grad_hit_y * ray_dir_y +
                      total_grad_hit_z * ray_dir_z;

  T total_grad_t = grad_t + grad_t_from_hit;

  // Gradient for ray origin:
  // d(loss)/do = d(loss)/dhit * dhit/do + d(loss)/dt * dt/do
  //            = grad_hit * (I + dir * dt/do) + grad_t * dt/do
  //            = grad_hit + (grad_hit . dir + grad_t) * dt/do
  //            = grad_hit + total_grad_t * dt/do
  grad_origin_x = total_grad_hit_x + total_grad_t * dt_do_x;
  grad_origin_y = total_grad_hit_y + total_grad_t * dt_do_y;
  grad_origin_z = total_grad_hit_z + total_grad_t * dt_do_z;

  // Gradient for ray direction:
  // d(loss)/ddir = d(loss)/dhit * dhit/ddir + d(loss)/dt * dt/ddir
  //              = grad_hit * (t * I + dir * dt/ddir) + grad_t * dt/ddir
  //              = t * grad_hit + (grad_hit . dir + grad_t) * dt/ddir
  //              = t * grad_hit + total_grad_t * dt/ddir
  grad_dir_x = t * total_grad_hit_x + total_grad_t * dt_ddir_x;
  grad_dir_y = t * total_grad_hit_y + total_grad_t * dt_ddir_y;
  grad_dir_z = t * total_grad_hit_z + total_grad_t * dt_ddir_z;

  // Gradient for plane normal:
  // d(loss)/dn = d(loss)/dhit * dhit/dn + d(loss)/dt * dt/dn
  //            = grad_hit * (dir * dt/dn) + grad_t * dt/dn
  //            = (grad_hit . dir + grad_t) * dt/dn
  //            = total_grad_t * dt/dn
  // Note: We ignore gradient through normalized normal output for simplicity
  grad_plane_normal_x = total_grad_t * dt_dn_x;
  grad_plane_normal_y = total_grad_t * dt_dn_y;
  grad_plane_normal_z = total_grad_t * dt_dn_z;

  // Gradient for plane offset:
  // d(loss)/dd = d(loss)/dhit * dhit/dd + d(loss)/dt * dt/dd
  //            = grad_hit * (dir * dt/dd) + grad_t * dt/dd
  //            = (grad_hit . dir + grad_t) * dt/dd
  //            = total_grad_t * dt/dd
  grad_plane_offset = total_grad_t * dt_dd;
}

}  // namespace torchscience::kernel::geometry::intersection
