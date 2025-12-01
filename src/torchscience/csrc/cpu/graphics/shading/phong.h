#pragma once

#include <cmath>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/macros/Macros.h>
#include <torch/library.h>

namespace torchscience::cpu::graphics::shading {

namespace {

// Forward kernel
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T phong_kernel(
    const T* normal,
    const T* view,
    const T* light,
    T shininess
) {
    // n.l
    T n_dot_l = normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2];

    // Back-facing check
    if (n_dot_l <= T(0)) {
        return T(0);
    }

    // Compute reflection vector: R = 2(n.l)n - l
    T two_n_dot_l = T(2) * n_dot_l;
    T reflect[3] = {
        two_n_dot_l * normal[0] - light[0],
        two_n_dot_l * normal[1] - light[1],
        two_n_dot_l * normal[2] - light[2]
    };

    // R.v
    T r_dot_v = reflect[0] * view[0] + reflect[1] * view[1] + reflect[2] * view[2];

    if (r_dot_v <= T(0)) {
        return T(0);
    }

    // Specular = (R.v)^shininess
    return std::pow(r_dot_v, shininess);
}

// Backward kernel
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void phong_backward_kernel(
    T grad_output,
    const T* normal,
    const T* view,
    const T* light,
    T shininess,
    T* grad_normal,
    T* grad_view,
    T* grad_light,
    T* grad_shininess
) {
    // Initialize gradients to zero
    for (int i = 0; i < 3; ++i) {
        grad_normal[i] = T(0);
        grad_view[i] = T(0);
        grad_light[i] = T(0);
    }
    *grad_shininess = T(0);

    T n_dot_l = normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2];

    if (n_dot_l <= T(0)) {
        return;
    }

    // Reflection vector
    T two_n_dot_l = T(2) * n_dot_l;
    T reflect[3] = {
        two_n_dot_l * normal[0] - light[0],
        two_n_dot_l * normal[1] - light[1],
        two_n_dot_l * normal[2] - light[2]
    };

    T r_dot_v = reflect[0] * view[0] + reflect[1] * view[1] + reflect[2] * view[2];

    if (r_dot_v <= T(0)) {
        return;
    }

    // Forward value: f = (R.v)^shininess
    T f = std::pow(r_dot_v, shininess);

    // df/d(shininess) = f * log(R.v)
    *grad_shininess = grad_output * f * std::log(r_dot_v);

    // df/d(R.v) = shininess * (R.v)^(shininess-1)
    T df_drdotv = grad_output * shininess * std::pow(r_dot_v, shininess - T(1));

    // d(R.v)/dv = R
    for (int i = 0; i < 3; ++i) {
        grad_view[i] = df_drdotv * reflect[i];
    }

    // d(R.v)/dR = v, dR/d(n.l) = 2n, dR/dn = 2(n.l)*I, dR/dl = 2n*n^T - I
    // d(R.v)/d(n.l) = 2 * (n.v)
    T n_dot_v = normal[0] * view[0] + normal[1] * view[1] + normal[2] * view[2];

    // d(n.l)/dn = l, d(n.l)/dl = n
    // Total grad_normal = df/d(n.l) * l + df/dR * dR/dn
    // R_i = 2(n.l)*n_i - l_i
    // dR_i/dn_j = 2*l_j*n_i + 2*(n.l)*delta_ij
    // d(R.v)/dn_j = sum_i v_i * dR_i/dn_j = 2*l_j*(n.v) + 2*(n.l)*v_j
    for (int i = 0; i < 3; ++i) {
        grad_normal[i] = df_drdotv * (T(2) * light[i] * n_dot_v + T(2) * n_dot_l * view[i]);
        // dR_i/dl_j = 2*n_j*n_i - delta_ij
        // d(R.v)/dl_j = sum_i v_i * dR_i/dl_j = 2*n_j*(n.v) - v_j
        grad_light[i] = df_drdotv * (T(2) * normal[i] * n_dot_v - view[i]);
    }
}

}  // namespace

inline at::Tensor phong(
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& shininess
) {
    TORCH_CHECK(normal.size(-1) == 3, "phong: normal must have last dimension 3");
    TORCH_CHECK(view.size(-1) == 3, "phong: view must have last dimension 3");
    TORCH_CHECK(light.size(-1) == 3, "phong: light must have last dimension 3");

    auto normal_contig = normal.contiguous();
    auto view_contig = view.contiguous();
    auto light_contig = light.contiguous();
    auto shininess_contig = shininess.contiguous();

    // Compute output shape (batch dimensions)
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < normal.dim() - 1; ++i) {
        output_shape.push_back(normal.size(i));
    }

    auto output = at::empty(output_shape, normal.options());
    int64_t num_elements = output.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        normal.scalar_type(), "phong_cpu", [&] {
            const scalar_t* normal_ptr = normal_contig.data_ptr<scalar_t>();
            const scalar_t* view_ptr = view_contig.data_ptr<scalar_t>();
            const scalar_t* light_ptr = light_contig.data_ptr<scalar_t>();
            const scalar_t* shininess_ptr = shininess_contig.data_ptr<scalar_t>();
            scalar_t* output_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    output_ptr[i] = phong_kernel(
                        normal_ptr + i * 3,
                        view_ptr + i * 3,
                        light_ptr + i * 3,
                        shininess_ptr[i]
                    );
                }
            });
        }
    );

    return output;
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> phong_backward(
    const at::Tensor& grad_output,
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& shininess
) {
    auto grad_output_contig = grad_output.contiguous();
    auto normal_contig = normal.contiguous();
    auto view_contig = view.contiguous();
    auto light_contig = light.contiguous();
    auto shininess_contig = shininess.contiguous();

    auto grad_normal = at::empty_like(normal);
    auto grad_view = at::empty_like(view);
    auto grad_light = at::empty_like(light);
    auto grad_shininess = at::empty_like(shininess);

    int64_t num_elements = grad_output.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        normal.scalar_type(), "phong_backward_cpu", [&] {
            const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
            const scalar_t* normal_ptr = normal_contig.data_ptr<scalar_t>();
            const scalar_t* view_ptr = view_contig.data_ptr<scalar_t>();
            const scalar_t* light_ptr = light_contig.data_ptr<scalar_t>();
            const scalar_t* shininess_ptr = shininess_contig.data_ptr<scalar_t>();
            scalar_t* grad_normal_ptr = grad_normal.data_ptr<scalar_t>();
            scalar_t* grad_view_ptr = grad_view.data_ptr<scalar_t>();
            scalar_t* grad_light_ptr = grad_light.data_ptr<scalar_t>();
            scalar_t* grad_shininess_ptr = grad_shininess.data_ptr<scalar_t>();

            at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    phong_backward_kernel(
                        grad_output_ptr[i],
                        normal_ptr + i * 3,
                        view_ptr + i * 3,
                        light_ptr + i * 3,
                        shininess_ptr[i],
                        grad_normal_ptr + i * 3,
                        grad_view_ptr + i * 3,
                        grad_light_ptr + i * 3,
                        grad_shininess_ptr + i
                    );
                }
            });
        }
    );

    return std::make_tuple(grad_normal, grad_view, grad_light, grad_shininess);
}

}  // namespace torchscience::cpu::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("phong", &torchscience::cpu::graphics::shading::phong);
    m.impl("phong_backward", &torchscience::cpu::graphics::shading::phong_backward);
}
