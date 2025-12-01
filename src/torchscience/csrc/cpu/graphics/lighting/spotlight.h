#pragma once

#include <cmath>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/macros/Macros.h>
#include <torch/library.h>

namespace torchscience::cpu::graphics::lighting {

namespace {

// Smoothstep function: smooth Hermite interpolation
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T smoothstep(T edge0, T edge1, T x) {
    // Clamp x to [0, 1] range after normalizing
    T t = (x - edge0) / (edge1 - edge0);
    t = t < T(0) ? T(0) : (t > T(1) ? T(1) : t);
    // Smooth Hermite interpolation
    return t * t * (T(3) - T(2) * t);
}

// Derivative of smoothstep with respect to t
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T smoothstep_derivative(T edge0, T edge1, T x) {
    T t = (x - edge0) / (edge1 - edge0);
    if (t <= T(0) || t >= T(1)) {
        return T(0);  // Zero derivative at clamped regions
    }
    // d(t^2 * (3 - 2t))/dt = 6t(1-t)
    // dt/dx = 1/(edge1 - edge0)
    return T(6) * t * (T(1) - t) / (edge1 - edge0);
}

// Forward kernel
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void spotlight_kernel(
    const T* light_pos,
    const T* surface_pos,
    const T* spot_direction,
    T intensity,
    T inner_angle,
    T outer_angle,
    T* irradiance,
    T* light_dir
) {
    // Compute direction from surface to light
    T to_light[3] = {
        light_pos[0] - surface_pos[0],
        light_pos[1] - surface_pos[1],
        light_pos[2] - surface_pos[2]
    };

    // Distance squared
    T dist_sq = to_light[0] * to_light[0] + to_light[1] * to_light[1] + to_light[2] * to_light[2];
    T dist = std::sqrt(dist_sq);

    // Normalize light direction (from surface to light)
    if (dist > T(0)) {
        light_dir[0] = to_light[0] / dist;
        light_dir[1] = to_light[1] / dist;
        light_dir[2] = to_light[2] / dist;
    } else {
        light_dir[0] = T(0);
        light_dir[1] = T(0);
        light_dir[2] = T(0);
        *irradiance = T(0);
        return;
    }

    // Direction from light to surface (negative of light_dir)
    T neg_light_dir[3] = {-light_dir[0], -light_dir[1], -light_dir[2]};

    // Dot product: cos(theta) = dot(-light_to_surface, spot_direction)
    T cos_theta = neg_light_dir[0] * spot_direction[0] +
                  neg_light_dir[1] * spot_direction[1] +
                  neg_light_dir[2] * spot_direction[2];

    // Compute angular falloff using smoothstep
    T cos_outer = std::cos(outer_angle);
    T cos_inner = std::cos(inner_angle);

    // If outside outer cone, no light
    if (cos_theta <= cos_outer) {
        *irradiance = T(0);
        return;
    }

    // Compute falloff: 1.0 inside inner cone, 0.0 outside outer cone, smooth in between
    T falloff = smoothstep(cos_outer, cos_inner, cos_theta);

    // Irradiance = intensity * falloff / distance^2
    *irradiance = intensity * falloff / dist_sq;
}

// Backward kernel
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void spotlight_backward_kernel(
    T grad_irradiance,
    const T* light_pos,
    const T* surface_pos,
    const T* spot_direction,
    T intensity,
    T inner_angle,
    T outer_angle,
    T* grad_light_pos,
    T* grad_surface_pos,
    T* grad_spot_direction,
    T* grad_intensity,
    T* grad_inner_angle,
    T* grad_outer_angle
) {
    // Initialize gradients
    for (int i = 0; i < 3; ++i) {
        grad_light_pos[i] = T(0);
        grad_surface_pos[i] = T(0);
        grad_spot_direction[i] = T(0);
    }
    *grad_intensity = T(0);
    *grad_inner_angle = T(0);
    *grad_outer_angle = T(0);

    // Recompute forward pass values
    T to_light[3] = {
        light_pos[0] - surface_pos[0],
        light_pos[1] - surface_pos[1],
        light_pos[2] - surface_pos[2]
    };

    T dist_sq = to_light[0] * to_light[0] + to_light[1] * to_light[1] + to_light[2] * to_light[2];
    if (dist_sq < T(1e-12)) return;

    T dist = std::sqrt(dist_sq);
    T inv_dist = T(1) / dist;

    T light_dir[3] = {to_light[0] * inv_dist, to_light[1] * inv_dist, to_light[2] * inv_dist};
    T neg_light_dir[3] = {-light_dir[0], -light_dir[1], -light_dir[2]};

    T cos_theta = neg_light_dir[0] * spot_direction[0] +
                  neg_light_dir[1] * spot_direction[1] +
                  neg_light_dir[2] * spot_direction[2];

    T cos_outer = std::cos(outer_angle);
    T cos_inner = std::cos(inner_angle);

    if (cos_theta <= cos_outer) return;  // No gradient if outside cone

    T falloff = smoothstep(cos_outer, cos_inner, cos_theta);
    T d_falloff_d_cos_theta = smoothstep_derivative(cos_outer, cos_inner, cos_theta);

    // irradiance = intensity * falloff / dist_sq
    T inv_dist_sq = T(1) / dist_sq;

    // d_irradiance/d_intensity = falloff / dist_sq
    *grad_intensity = grad_irradiance * falloff * inv_dist_sq;

    // d_irradiance/d_falloff = intensity / dist_sq
    T grad_falloff = grad_irradiance * intensity * inv_dist_sq;

    // d_irradiance/d_dist_sq = -intensity * falloff / dist_sq^2
    T grad_dist_sq = grad_irradiance * (-intensity * falloff * inv_dist_sq * inv_dist_sq);

    // d_falloff/d_cos_theta
    T grad_cos_theta = grad_falloff * d_falloff_d_cos_theta;

    // d_falloff/d_cos_outer and d_falloff/d_cos_inner
    // smoothstep(edge0, edge1, x) where t = (x - edge0)/(edge1 - edge0)
    // dt/d_edge0 = -(1 + (x - edge0)/(edge1 - edge0)) / (edge1 - edge0) = -1/(edge1-edge0) - t/(edge1-edge0)
    // dt/d_edge1 = (x - edge0) / (edge1 - edge0)^2 = t/(edge1 - edge0)
    T t = (cos_theta - cos_outer) / (cos_inner - cos_outer);
    t = t < T(0) ? T(0) : (t > T(1) ? T(1) : t);
    T d_smoothstep_dt = T(6) * t * (T(1) - t);
    T range = cos_inner - cos_outer;
    T range_sq = range * range;

    if (std::abs(range) > T(1e-10) && t > T(0) && t < T(1)) {
        T dt_d_cos_outer = (t - T(1)) / range;
        T dt_d_cos_inner = -t / range;

        T grad_cos_outer = grad_falloff * d_smoothstep_dt * dt_d_cos_outer;
        T grad_cos_inner = grad_falloff * d_smoothstep_dt * dt_d_cos_inner;

        // d_cos_outer/d_outer_angle = -sin(outer_angle)
        *grad_outer_angle = grad_cos_outer * (-std::sin(outer_angle));
        // d_cos_inner/d_inner_angle = -sin(inner_angle)
        *grad_inner_angle = grad_cos_inner * (-std::sin(inner_angle));
    }

    // cos_theta = dot(-light_dir, spot_direction) = dot(-(to_light/dist), spot_direction)
    // = -dot(to_light, spot_direction) / dist
    T dot_to_light_spot = to_light[0] * spot_direction[0] +
                          to_light[1] * spot_direction[1] +
                          to_light[2] * spot_direction[2];

    // d_cos_theta/d_spot_direction = -to_light / dist = -light_dir
    for (int i = 0; i < 3; ++i) {
        grad_spot_direction[i] = grad_cos_theta * (-light_dir[i]);
    }

    // d_cos_theta/d_to_light = -spot_direction/dist + dot_to_light_spot * to_light / dist^3
    // = (-spot_direction + cos_theta * light_dir) / dist
    T d_cos_theta_d_to_light[3];
    for (int i = 0; i < 3; ++i) {
        d_cos_theta_d_to_light[i] = (-spot_direction[i] + cos_theta * light_dir[i]) * inv_dist;
    }

    // d_dist_sq/d_to_light = 2 * to_light
    for (int i = 0; i < 3; ++i) {
        T grad_to_light_i = grad_cos_theta * d_cos_theta_d_to_light[i] + grad_dist_sq * T(2) * to_light[i];
        grad_light_pos[i] = grad_to_light_i;
        grad_surface_pos[i] = -grad_to_light_i;
    }
}

}  // namespace

inline std::tuple<at::Tensor, at::Tensor> spotlight(
    const at::Tensor& light_pos,
    const at::Tensor& surface_pos,
    const at::Tensor& spot_direction,
    const at::Tensor& intensity,
    const at::Tensor& inner_angle,
    const at::Tensor& outer_angle
) {
    TORCH_CHECK(light_pos.size(-1) == 3, "spotlight: light_pos must have last dimension 3");
    TORCH_CHECK(surface_pos.size(-1) == 3, "spotlight: surface_pos must have last dimension 3");
    TORCH_CHECK(spot_direction.size(-1) == 3, "spotlight: spot_direction must have last dimension 3");

    auto light_pos_contig = light_pos.contiguous();
    auto surface_pos_contig = surface_pos.contiguous();
    auto spot_direction_contig = spot_direction.contiguous();
    auto intensity_contig = intensity.contiguous();
    auto inner_angle_contig = inner_angle.contiguous();
    auto outer_angle_contig = outer_angle.contiguous();

    // Compute output shape (batch dimensions)
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < light_pos.dim() - 1; ++i) {
        batch_shape.push_back(light_pos.size(i));
    }

    auto irradiance = at::empty(batch_shape, light_pos.options());
    auto light_dir = at::empty_like(light_pos);
    int64_t num_elements = irradiance.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        light_pos.scalar_type(), "spotlight_cpu", [&] {
            const scalar_t* light_pos_ptr = light_pos_contig.data_ptr<scalar_t>();
            const scalar_t* surface_pos_ptr = surface_pos_contig.data_ptr<scalar_t>();
            const scalar_t* spot_direction_ptr = spot_direction_contig.data_ptr<scalar_t>();
            const scalar_t* intensity_ptr = intensity_contig.data_ptr<scalar_t>();
            const scalar_t* inner_angle_ptr = inner_angle_contig.data_ptr<scalar_t>();
            const scalar_t* outer_angle_ptr = outer_angle_contig.data_ptr<scalar_t>();
            scalar_t* irradiance_ptr = irradiance.data_ptr<scalar_t>();
            scalar_t* light_dir_ptr = light_dir.data_ptr<scalar_t>();

            at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    spotlight_kernel(
                        light_pos_ptr + i * 3,
                        surface_pos_ptr + i * 3,
                        spot_direction_ptr + i * 3,
                        intensity_ptr[i],
                        inner_angle_ptr[i],
                        outer_angle_ptr[i],
                        irradiance_ptr + i,
                        light_dir_ptr + i * 3
                    );
                }
            });
        }
    );

    return std::make_tuple(irradiance, light_dir);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> spotlight_backward(
    const at::Tensor& grad_irradiance,
    const at::Tensor& light_pos,
    const at::Tensor& surface_pos,
    const at::Tensor& spot_direction,
    const at::Tensor& intensity,
    const at::Tensor& inner_angle,
    const at::Tensor& outer_angle
) {
    auto grad_irradiance_contig = grad_irradiance.contiguous();
    auto light_pos_contig = light_pos.contiguous();
    auto surface_pos_contig = surface_pos.contiguous();
    auto spot_direction_contig = spot_direction.contiguous();
    auto intensity_contig = intensity.contiguous();
    auto inner_angle_contig = inner_angle.contiguous();
    auto outer_angle_contig = outer_angle.contiguous();

    auto grad_light_pos = at::empty_like(light_pos);
    auto grad_surface_pos = at::empty_like(surface_pos);
    auto grad_spot_direction = at::empty_like(spot_direction);
    auto grad_intensity = at::empty_like(intensity);
    auto grad_inner_angle = at::empty_like(inner_angle);
    auto grad_outer_angle = at::empty_like(outer_angle);

    int64_t num_elements = grad_irradiance.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        light_pos.scalar_type(), "spotlight_backward_cpu", [&] {
            const scalar_t* grad_irradiance_ptr = grad_irradiance_contig.data_ptr<scalar_t>();
            const scalar_t* light_pos_ptr = light_pos_contig.data_ptr<scalar_t>();
            const scalar_t* surface_pos_ptr = surface_pos_contig.data_ptr<scalar_t>();
            const scalar_t* spot_direction_ptr = spot_direction_contig.data_ptr<scalar_t>();
            const scalar_t* intensity_ptr = intensity_contig.data_ptr<scalar_t>();
            const scalar_t* inner_angle_ptr = inner_angle_contig.data_ptr<scalar_t>();
            const scalar_t* outer_angle_ptr = outer_angle_contig.data_ptr<scalar_t>();
            scalar_t* grad_light_pos_ptr = grad_light_pos.data_ptr<scalar_t>();
            scalar_t* grad_surface_pos_ptr = grad_surface_pos.data_ptr<scalar_t>();
            scalar_t* grad_spot_direction_ptr = grad_spot_direction.data_ptr<scalar_t>();
            scalar_t* grad_intensity_ptr = grad_intensity.data_ptr<scalar_t>();
            scalar_t* grad_inner_angle_ptr = grad_inner_angle.data_ptr<scalar_t>();
            scalar_t* grad_outer_angle_ptr = grad_outer_angle.data_ptr<scalar_t>();

            at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    spotlight_backward_kernel(
                        grad_irradiance_ptr[i],
                        light_pos_ptr + i * 3,
                        surface_pos_ptr + i * 3,
                        spot_direction_ptr + i * 3,
                        intensity_ptr[i],
                        inner_angle_ptr[i],
                        outer_angle_ptr[i],
                        grad_light_pos_ptr + i * 3,
                        grad_surface_pos_ptr + i * 3,
                        grad_spot_direction_ptr + i * 3,
                        grad_intensity_ptr + i,
                        grad_inner_angle_ptr + i,
                        grad_outer_angle_ptr + i
                    );
                }
            });
        }
    );

    return std::make_tuple(grad_light_pos, grad_surface_pos, grad_spot_direction,
                           grad_intensity, grad_inner_angle, grad_outer_angle);
}

}  // namespace torchscience::cpu::graphics::lighting

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("spotlight", &torchscience::cpu::graphics::lighting::spotlight);
    m.impl("spotlight_backward", &torchscience::cpu::graphics::lighting::spotlight_backward);
}
