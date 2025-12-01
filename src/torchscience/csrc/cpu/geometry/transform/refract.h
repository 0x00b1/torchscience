#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/geometry/transform/refract.h"
#include "../../../kernel/geometry/transform/refract_backward.h"

namespace torchscience::cpu::geometry::transform {

/**
 * CPU implementation for vector refraction using Snell's law.
 *
 * Supports both scalar eta (broadcast to all rays) and batched eta (one per ray).
 */
inline at::Tensor refract(
    const at::Tensor& direction,
    const at::Tensor& normal,
    const at::Tensor& eta
) {
  TORCH_CHECK(direction.size(-1) == 3,
              "refract: direction must have last dimension 3, got ", direction.size(-1));
  TORCH_CHECK(normal.size(-1) == 3,
              "refract: normal must have last dimension 3, got ", normal.size(-1));
  TORCH_CHECK(direction.scalar_type() == normal.scalar_type(),
              "refract: direction and normal must have the same dtype");
  TORCH_CHECK(direction.scalar_type() == eta.scalar_type(),
              "refract: eta must have the same dtype as direction and normal");
  TORCH_CHECK(direction.sizes().slice(0, direction.dim() - 1) ==
              normal.sizes().slice(0, normal.dim() - 1),
              "refract: direction and normal must have matching batch dimensions");

  const int64_t num_vectors = direction.numel() / 3;

  // Check eta shape: either scalar, 1-element, or matching batch size
  const bool scalar_eta = (eta.numel() == 1);
  if (!scalar_eta) {
    TORCH_CHECK(eta.numel() == num_vectors,
                "refract: eta must be scalar or broadcast to batch size, got ",
                eta.numel(), " elements but expected 1 or ", num_vectors);
  }

  auto output = at::empty_like(direction);

  auto direction_contig = direction.contiguous();
  auto normal_contig = normal.contiguous();
  auto eta_contig = eta.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    direction.scalar_type(),
    "refract_cpu",
    [&] {
      const scalar_t* direction_ptr = direction_contig.data_ptr<scalar_t>();
      const scalar_t* normal_ptr = normal_contig.data_ptr<scalar_t>();
      const scalar_t* eta_ptr = eta_contig.data_ptr<scalar_t>();
      scalar_t* output_ptr = output.data_ptr<scalar_t>();

      if (scalar_eta) {
        const scalar_t eta_val = eta_ptr[0];
        at::parallel_for(0, num_vectors, 0, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            kernel::geometry::transform::refract_scalar(
              direction_ptr + i * 3,
              normal_ptr + i * 3,
              eta_val,
              output_ptr + i * 3
            );
          }
        });
      } else {
        at::parallel_for(0, num_vectors, 0, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            kernel::geometry::transform::refract_scalar(
              direction_ptr + i * 3,
              normal_ptr + i * 3,
              eta_ptr[i],
              output_ptr + i * 3
            );
          }
        });
      }
    }
  );

  return output;
}

/**
 * CPU implementation for vector refraction backward pass.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> refract_backward(
    const at::Tensor& grad_output,
    const at::Tensor& direction,
    const at::Tensor& normal,
    const at::Tensor& eta
) {
  TORCH_CHECK(grad_output.size(-1) == 3,
              "refract_backward: grad_output must have last dimension 3");
  TORCH_CHECK(direction.size(-1) == 3,
              "refract_backward: direction must have last dimension 3");
  TORCH_CHECK(normal.size(-1) == 3,
              "refract_backward: normal must have last dimension 3");

  const int64_t num_vectors = direction.numel() / 3;
  const bool scalar_eta = (eta.numel() == 1);

  auto grad_direction = at::empty_like(direction);
  auto grad_normal = at::empty_like(normal);
  auto grad_eta = scalar_eta ? at::zeros({1}, eta.options()) : at::empty_like(eta);

  auto grad_output_contig = grad_output.contiguous();
  auto direction_contig = direction.contiguous();
  auto normal_contig = normal.contiguous();
  auto eta_contig = eta.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    direction.scalar_type(),
    "refract_backward_cpu",
    [&] {
      const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
      const scalar_t* direction_ptr = direction_contig.data_ptr<scalar_t>();
      const scalar_t* normal_ptr = normal_contig.data_ptr<scalar_t>();
      const scalar_t* eta_ptr = eta_contig.data_ptr<scalar_t>();
      scalar_t* grad_direction_ptr = grad_direction.data_ptr<scalar_t>();
      scalar_t* grad_normal_ptr = grad_normal.data_ptr<scalar_t>();
      scalar_t* grad_eta_ptr = grad_eta.data_ptr<scalar_t>();

      if (scalar_eta) {
        const scalar_t eta_val = eta_ptr[0];
        // For scalar eta, we need to accumulate gradients
        // Use thread-local accumulation then reduce
        std::atomic<double> total_grad_eta{0.0};

        at::parallel_for(0, num_vectors, 0, [&](int64_t begin, int64_t end) {
          double local_grad_eta = 0.0;
          for (int64_t i = begin; i < end; ++i) {
            scalar_t grad_eta_i;
            kernel::geometry::transform::refract_backward_scalar(
              grad_output_ptr + i * 3,
              direction_ptr + i * 3,
              normal_ptr + i * 3,
              eta_val,
              grad_direction_ptr + i * 3,
              grad_normal_ptr + i * 3,
              &grad_eta_i
            );
            local_grad_eta += static_cast<double>(grad_eta_i);
          }
          // Atomic add for thread safety
          double expected = total_grad_eta.load();
          while (!total_grad_eta.compare_exchange_weak(expected, expected + local_grad_eta)) {}
        });

        grad_eta_ptr[0] = static_cast<scalar_t>(total_grad_eta.load());
      } else {
        at::parallel_for(0, num_vectors, 0, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            kernel::geometry::transform::refract_backward_scalar(
              grad_output_ptr + i * 3,
              direction_ptr + i * 3,
              normal_ptr + i * 3,
              eta_ptr[i],
              grad_direction_ptr + i * 3,
              grad_normal_ptr + i * 3,
              grad_eta_ptr + i
            );
          }
        });
      }
    }
  );

  return std::make_tuple(grad_direction, grad_normal, grad_eta);
}

}  // namespace torchscience::cpu::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("refract", &torchscience::cpu::geometry::transform::refract);
  m.impl("refract_backward", &torchscience::cpu::geometry::transform::refract_backward);
}
