#pragma once

namespace torchscience::kernel::geometry::transform {

/**
 * Backward pass for quaternion_apply.
 * Computes gradients w.r.t. quaternion q and point.
 *
 * Forward: v' = v + w*t + (q_xyz x t)
 * where t = 2*(q_xyz x v), q_xyz = [x, y, z]
 *
 * Expanding fully:
 *   tx = 2*(y*pz - z*py)
 *   ty = 2*(z*px - x*pz)
 *   tz = 2*(x*py - y*px)
 *
 *   output_x = px + w*tx + y*tz - z*ty
 *   output_y = py + w*ty + z*tx - x*tz
 *   output_z = pz + w*tz + x*ty - y*tx
 */
template <typename T>
void quaternion_apply_backward_scalar(
    const T* grad_output,
    const T* q,
    const T* point,
    T* grad_q,
    T* grad_point
) {
  const T w = q[0], x = q[1], y = q[2], z = q[3];
  const T px = point[0], py = point[1], pz = point[2];
  const T gx = grad_output[0], gy = grad_output[1], gz = grad_output[2];

  // Forward intermediates
  const T tx = T(2) * (y * pz - z * py);
  const T ty = T(2) * (z * px - x * pz);
  const T tz = T(2) * (x * py - y * px);

  // Gradient w.r.t. w: d(output)/dw = [tx, ty, tz]
  grad_q[0] = gx * tx + gy * ty + gz * tz;

  // For gradients w.r.t. x, y, z, we need the Jacobian
  // output_x = px + w*tx + y*tz - z*ty
  //          = px + 2*w*(y*pz - z*py) + 2*y*(x*py - y*px) - 2*z*(z*px - x*pz)
  //          = px + 2*w*y*pz - 2*w*z*py + 2*x*y*py - 2*y^2*px - 2*z^2*px + 2*x*z*pz
  //
  // d(output_x)/dx = 2*y*py + 2*z*pz
  // d(output_x)/dy = 2*w*pz + 2*x*py - 4*y*px = 2*(w*pz + x*py - 2*y*px)
  // d(output_x)/dz = -2*w*py - 4*z*px + 2*x*pz = 2*(-w*py - 2*z*px + x*pz)
  //
  // output_y = py + w*ty + z*tx - x*tz
  //          = py + 2*w*(z*px - x*pz) + 2*z*(y*pz - z*py) - 2*x*(x*py - y*px)
  //          = py + 2*w*z*px - 2*w*x*pz + 2*y*z*pz - 2*z^2*py - 2*x^2*py + 2*x*y*px
  //
  // d(output_y)/dx = -2*w*pz - 4*x*py + 2*y*px = 2*(-w*pz - 2*x*py + y*px)
  // d(output_y)/dy = 2*z*pz + 2*x*px
  // d(output_y)/dz = 2*w*px + 2*y*pz - 4*z*py = 2*(w*px + y*pz - 2*z*py)
  //
  // output_z = pz + w*tz + x*ty - y*tx
  //          = pz + 2*w*(x*py - y*px) + 2*x*(z*px - x*pz) - 2*y*(y*pz - z*py)
  //          = pz + 2*w*x*py - 2*w*y*px + 2*x*z*px - 2*x^2*pz - 2*y^2*pz + 2*y*z*py
  //
  // d(output_z)/dx = 2*w*py + 2*z*px - 4*x*pz = 2*(w*py + z*px - 2*x*pz)
  // d(output_z)/dy = -2*w*px - 4*y*pz + 2*z*py = 2*(-w*px - 2*y*pz + z*py)
  // d(output_z)/dz = 2*x*px + 2*y*py

  // grad_x = sum_i g_i * d(output_i)/dx
  grad_q[1] = gx * T(2) * (y * py + z * pz) +
              gy * T(2) * (-w * pz - T(2) * x * py + y * px) +
              gz * T(2) * (w * py + z * px - T(2) * x * pz);

  // grad_y = sum_i g_i * d(output_i)/dy
  grad_q[2] = gx * T(2) * (w * pz + x * py - T(2) * y * px) +
              gy * T(2) * (z * pz + x * px) +
              gz * T(2) * (-w * px - T(2) * y * pz + z * py);

  // grad_z = sum_i g_i * d(output_i)/dz
  grad_q[3] = gx * T(2) * (-w * py - T(2) * z * px + x * pz) +
              gy * T(2) * (w * px + y * pz - T(2) * z * py) +
              gz * T(2) * (x * px + y * py);

  // Gradient w.r.t. point
  // The Jacobian d(output)/d(point) is the rotation matrix R(q).
  // For backward pass, we need R^T @ grad_output.
  // For unit quaternions, R^T = R^{-1} = R(q^{-1}) = R(q*) where q* is conjugate.
  // So grad_point = quaternion_apply(q*, grad_output)
  // where q* = [w, -x, -y, -z]
  //
  // Using the same formula with conjugate:
  // t' = 2 * ((-q_xyz) x g) = -2 * (q_xyz x g)
  // grad_point = g + w*t' + ((-q_xyz) x t')
  //            = g - w*2*(q_xyz x g) + (q_xyz x 2*(q_xyz x g))
  //
  // Let's compute directly:
  const T gx2 = T(2) * (y * gz - z * gy);  // 2*(q_xyz x g)
  const T gy2 = T(2) * (z * gx - x * gz);
  const T gz2 = T(2) * (x * gy - y * gx);

  // For conjugate: t' = -[gx2, gy2, gz2], so w*t' = -w*[gx2, gy2, gz2]
  // And (-q_xyz) x t' = (-q_xyz) x (-[gx2,gy2,gz2]) = q_xyz x [gx2,gy2,gz2]
  // So: grad_point = g - w*[gx2,gy2,gz2] + (q_xyz x [gx2,gy2,gz2])
  //                = g - w*[gx2,gy2,gz2] + [y*gz2 - z*gy2, z*gx2 - x*gz2, x*gy2 - y*gx2]
  //
  // Wait, let's re-derive. The formula with q = [w,x,y,z]:
  //   output = p + w*t + (q_xyz x t) where t = 2*(q_xyz x p)
  //
  // With conjugate q* = [w, -x, -y, -z]:
  //   t' = 2*((-q_xyz) x g) = -2*(q_xyz x g) = -[gx2, gy2, gz2]
  //   grad_point = g + w*t' + ((-q_xyz) x t')
  //              = g + w*(-[gx2,gy2,gz2]) + ((-q_xyz) x (-[gx2,gy2,gz2]))
  //              = g - w*[gx2,gy2,gz2] + (q_xyz x [gx2,gy2,gz2])
  //
  // Actually easier: just use the formula with -x, -y, -z:
  const T nx = -x, ny = -y, nz = -z;
  const T tx2 = T(2) * (ny * gz - nz * gy);  // = 2*(-y*gz + z*gy) = -gx2
  const T ty2 = T(2) * (nz * gx - nx * gz);  // = 2*(-z*gx + x*gz) = -gy2
  const T tz2 = T(2) * (nx * gy - ny * gx);  // = 2*(-x*gy + y*gx) = -gz2

  grad_point[0] = gx + w * tx2 + (ny * tz2 - nz * ty2);
  grad_point[1] = gy + w * ty2 + (nz * tx2 - nx * tz2);
  grad_point[2] = gz + w * tz2 + (nx * ty2 - ny * tx2);
}

}  // namespace torchscience::kernel::geometry::transform
