#pragma once

#include <cmath>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/macros/Macros.h>
#include <torch/library.h>

namespace torchscience::cpu::graphics::projection {

namespace {

// Perspective projection forward kernel
// Creates a 4x4 perspective projection matrix from fov, aspect, near, far
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void perspective_projection_forward_kernel(
    T fov,
    T aspect,
    T near,
    T far,
    T* output
) {
    // f = 1 / tan(fov / 2)
    T f = T(1) / std::tan(fov * T(0.5));

    T n_minus_f = near - far;

    // Row 0: [f/aspect, 0, 0, 0]
    output[0] = f / aspect;
    output[1] = T(0);
    output[2] = T(0);
    output[3] = T(0);

    // Row 1: [0, f, 0, 0]
    output[4] = T(0);
    output[5] = f;
    output[6] = T(0);
    output[7] = T(0);

    // Row 2: [0, 0, (far+near)/(near-far), 2*far*near/(near-far)]
    output[8] = T(0);
    output[9] = T(0);
    output[10] = (far + near) / n_minus_f;
    output[11] = T(2) * far * near / n_minus_f;

    // Row 3: [0, 0, -1, 0]
    output[12] = T(0);
    output[13] = T(0);
    output[14] = T(-1);
    output[15] = T(0);
}

// Backward kernel for perspective projection
// Computes gradients with respect to fov, aspect, near, far
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void perspective_projection_backward_kernel(
    const T* grad_output,  // 4x4 = 16 elements
    T fov,
    T aspect,
    T near,
    T far,
    T* grad_fov,
    T* grad_aspect,
    T* grad_near,
    T* grad_far
) {
    // f = 1 / tan(fov / 2)
    T half_fov = fov * T(0.5);
    T tan_half = std::tan(half_fov);
    T f = T(1) / tan_half;

    // df/dfov = -1/(2 * tan^2(fov/2)) * (1 + tan^2(fov/2))
    //         = -1/(2 * sin^2(fov/2)) * cos^2(fov/2) / cos^2(fov/2)
    //         = -1 / (2 * sin(fov/2) * cos(fov/2))
    //         = -1 / sin(fov)
    // Actually: df/dfov = d/dfov [1/tan(fov/2)]
    //                   = -sec^2(fov/2) * 0.5 / tan^2(fov/2)
    //                   = -0.5 * (1 + tan^2(fov/2)) / tan^2(fov/2)
    T sec_half_sq = T(1) + tan_half * tan_half;
    T df_dfov = -T(0.5) * sec_half_sq / (tan_half * tan_half);

    T n_minus_f = near - far;
    T n_minus_f_sq = n_minus_f * n_minus_f;

    // Gradients from each matrix element
    // M[0,0] = f / aspect
    // dL/dfov += grad[0,0] * (df/dfov / aspect)
    // dL/daspect += grad[0,0] * (-f / aspect^2)
    *grad_fov = grad_output[0] * df_dfov / aspect;
    *grad_aspect = grad_output[0] * (-f / (aspect * aspect));

    // M[1,1] = f
    // dL/dfov += grad[1,1] * df/dfov
    *grad_fov += grad_output[5] * df_dfov;

    // M[2,2] = (far + near) / (near - far)
    // d/dnear = [(near-far) - (far+near)] / (near-far)^2 = -2*far / (near-far)^2
    // d/dfar  = [(near-far) + (far+near)] / (near-far)^2 = 2*near / (near-far)^2
    *grad_near = grad_output[10] * (-T(2) * far / n_minus_f_sq);
    *grad_far = grad_output[10] * (T(2) * near / n_minus_f_sq);

    // M[2,3] = 2 * far * near / (near - far)
    // d/dnear = [2*far*(near-far) - 2*far*near] / (near-far)^2 = -2*far^2 / (near-far)^2
    // d/dfar  = [2*near*(near-far) - 2*far*near] / (near-far)^2 = 2*near^2 / (near-far)^2 - 2*far*near / (near-far)^2
    //         = 2*near*(near - far - far) / (near-far)^2 = -2*near^2 / (near-far)^2
    // Wait, let me recalculate:
    // M[2,3] = 2*f*n / (n-f)
    // d/dn = [2*f*(n-f) - 2*f*n*1] / (n-f)^2 = -2*f^2 / (n-f)^2
    // d/df = [2*n*(n-f) - 2*f*n*(-1)] / (n-f)^2 = [2*n*(n-f) + 2*f*n] / (n-f)^2 = 2*n^2 / (n-f)^2
    *grad_near += grad_output[11] * (-T(2) * far * far / n_minus_f_sq);
    *grad_far += grad_output[11] * (T(2) * near * near / n_minus_f_sq);

    // M[3,2] = -1 (constant, no gradient)
}

}  // namespace

inline at::Tensor perspective_projection(
    const at::Tensor& fov,
    const at::Tensor& aspect,
    const at::Tensor& near,
    const at::Tensor& far
) {
    auto fov_contig = fov.contiguous();
    auto aspect_contig = aspect.contiguous();
    auto near_contig = near.contiguous();
    auto far_contig = far.contiguous();

    // Determine batch shape (broadcast all inputs)
    auto batch_sizes = at::infer_size(fov.sizes(), aspect.sizes());
    batch_sizes = at::infer_size(batch_sizes, near.sizes());
    batch_sizes = at::infer_size(batch_sizes, far.sizes());

    // Output shape: batch_shape + (4, 4)
    std::vector<int64_t> output_shape(batch_sizes.begin(), batch_sizes.end());
    output_shape.push_back(4);
    output_shape.push_back(4);

    auto options = fov.options();
    auto output = at::empty(output_shape, options);

    // Expand inputs to batch shape
    auto fov_expanded = fov_contig.expand(batch_sizes).contiguous();
    auto aspect_expanded = aspect_contig.expand(batch_sizes).contiguous();
    auto near_expanded = near_contig.expand(batch_sizes).contiguous();
    auto far_expanded = far_contig.expand(batch_sizes).contiguous();

    int64_t num_elements = output.numel() / 16;  // Number of matrices

    AT_DISPATCH_FLOATING_TYPES(
        fov.scalar_type(), "perspective_projection_cpu", [&] {
            const scalar_t* fov_ptr = fov_expanded.data_ptr<scalar_t>();
            const scalar_t* aspect_ptr = aspect_expanded.data_ptr<scalar_t>();
            const scalar_t* near_ptr = near_expanded.data_ptr<scalar_t>();
            const scalar_t* far_ptr = far_expanded.data_ptr<scalar_t>();
            scalar_t* output_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    perspective_projection_forward_kernel(
                        fov_ptr[i],
                        aspect_ptr[i],
                        near_ptr[i],
                        far_ptr[i],
                        output_ptr + i * 16
                    );
                }
            });
        }
    );

    return output;
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
perspective_projection_backward(
    const at::Tensor& grad_output,
    const at::Tensor& fov,
    const at::Tensor& aspect,
    const at::Tensor& near,
    const at::Tensor& far
) {
    auto grad_output_contig = grad_output.contiguous();
    auto fov_contig = fov.contiguous();
    auto aspect_contig = aspect.contiguous();
    auto near_contig = near.contiguous();
    auto far_contig = far.contiguous();

    // Batch shape is grad_output shape minus last two dims
    auto grad_sizes = grad_output.sizes();
    std::vector<int64_t> batch_shape(grad_sizes.begin(), grad_sizes.end() - 2);

    auto options = grad_output.options();
    auto grad_fov = at::empty(batch_shape, options);
    auto grad_aspect = at::empty(batch_shape, options);
    auto grad_near = at::empty(batch_shape, options);
    auto grad_far = at::empty(batch_shape, options);

    // Expand inputs to batch shape
    auto fov_expanded = fov_contig.expand(batch_shape).contiguous();
    auto aspect_expanded = aspect_contig.expand(batch_shape).contiguous();
    auto near_expanded = near_contig.expand(batch_shape).contiguous();
    auto far_expanded = far_contig.expand(batch_shape).contiguous();

    int64_t num_elements = grad_fov.numel();

    AT_DISPATCH_FLOATING_TYPES(
        grad_output.scalar_type(), "perspective_projection_backward_cpu", [&] {
            const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
            const scalar_t* fov_ptr = fov_expanded.data_ptr<scalar_t>();
            const scalar_t* aspect_ptr = aspect_expanded.data_ptr<scalar_t>();
            const scalar_t* near_ptr = near_expanded.data_ptr<scalar_t>();
            const scalar_t* far_ptr = far_expanded.data_ptr<scalar_t>();
            scalar_t* grad_fov_ptr = grad_fov.data_ptr<scalar_t>();
            scalar_t* grad_aspect_ptr = grad_aspect.data_ptr<scalar_t>();
            scalar_t* grad_near_ptr = grad_near.data_ptr<scalar_t>();
            scalar_t* grad_far_ptr = grad_far.data_ptr<scalar_t>();

            at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    perspective_projection_backward_kernel(
                        grad_output_ptr + i * 16,
                        fov_ptr[i],
                        aspect_ptr[i],
                        near_ptr[i],
                        far_ptr[i],
                        grad_fov_ptr + i,
                        grad_aspect_ptr + i,
                        grad_near_ptr + i,
                        grad_far_ptr + i
                    );
                }
            });
        }
    );

    return std::make_tuple(grad_fov, grad_aspect, grad_near, grad_far);
}

}  // namespace torchscience::cpu::graphics::projection

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("perspective_projection", &torchscience::cpu::graphics::projection::perspective_projection);
    m.impl("perspective_projection_backward", &torchscience::cpu::graphics::projection::perspective_projection_backward);
}
