#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/probability/normal_cumulative_distribution.h"
#include "../../kernel/probability/normal_cumulative_distribution_backward.h"
#include "../../kernel/probability/normal_cumulative_distribution_backward_backward.h"
#include "../../kernel/probability/normal_probability_density.h"
#include "../../kernel/probability/normal_probability_density_backward.h"
#include "../../kernel/probability/normal_quantile.h"
#include "../../kernel/probability/normal_quantile_backward.h"
#include "../../kernel/probability/normal_survival.h"
#include "../../kernel/probability/normal_survival_backward.h"
#include "../../kernel/probability/normal_log_probability_density.h"
#include "../../kernel/probability/normal_log_probability_density_backward.h"

namespace torchscience::cpu::probability {

// Helper to reduce gradient to match original input shape
inline at::Tensor reduce_grad(const at::Tensor& grad, const at::Tensor& input) {
  if (grad.sizes() == input.sizes()) {
    return grad;
  }
  // Sum along dimensions that were broadcast
  auto grad_reduced = grad;
  // Handle extra leading dimensions
  while (grad_reduced.dim() > input.dim()) {
    grad_reduced = grad_reduced.sum(0);
  }
  // Handle size-1 dimensions that were broadcast
  for (int64_t i = 0; i < input.dim(); ++i) {
    if (input.size(i) == 1 && grad_reduced.size(i) > 1) {
      grad_reduced = grad_reduced.sum(i, /*keepdim=*/true);
    }
  }
  return grad_reduced;
}

at::Tensor normal_cumulative_distribution(
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  // Broadcast all inputs together
  auto tensors = at::broadcast_tensors({x, loc, scale});
  auto x_b = tensors[0].contiguous();
  auto loc_b = tensors[1].contiguous();
  auto scale_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "normal_cumulative_distribution_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto loc_data = loc_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::normal_cumulative_distribution<scalar_t>(
                x_data[i], loc_data[i], scale_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_cumulative_distribution_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  // Broadcast all inputs
  auto tensors = at::broadcast_tensors({grad, x, loc, scale});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto loc_b = tensors[2].contiguous();
  auto scale_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_loc = at::empty_like(loc_b);
  auto grad_scale = at::empty_like(scale_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "normal_cumulative_distribution_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto loc_data = loc_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_loc_data = grad_loc.data_ptr<scalar_t>();
        auto grad_scale_data = grad_scale.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            kernel::probability::normal_cumulative_distribution_backward<scalar_t>(
                grad_data[i], x_data[i], loc_data[i], scale_data[i],
                grad_x_data[i], grad_loc_data[i], grad_scale_data[i]);
          }
        });
      });

  // Reduce gradients if inputs were broadcast
  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_loc, loc),
      reduce_grad(grad_scale, scale));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> normal_cumulative_distribution_backward_backward(
    const at::Tensor& gg_x,
    const at::Tensor& gg_loc,
    const at::Tensor& gg_scale,
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  // Broadcast all inputs
  auto tensors = at::broadcast_tensors({gg_x, gg_loc, gg_scale, grad, x, loc, scale});
  auto gg_x_b = tensors[0].contiguous();
  auto gg_loc_b = tensors[1].contiguous();
  auto gg_scale_b = tensors[2].contiguous();
  auto grad_b = tensors[3].contiguous();
  auto x_b = tensors[4].contiguous();
  auto loc_b = tensors[5].contiguous();
  auto scale_b = tensors[6].contiguous();

  auto out_grad = at::empty_like(grad_b);
  auto out_x = at::empty_like(x_b);
  auto out_loc = at::empty_like(loc_b);
  auto out_scale = at::empty_like(scale_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "normal_cumulative_distribution_backward_backward_cpu", [&] {
        auto gg_x_data = gg_x_b.data_ptr<scalar_t>();
        auto gg_loc_data = gg_loc_b.data_ptr<scalar_t>();
        auto gg_scale_data = gg_scale_b.data_ptr<scalar_t>();
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto loc_data = loc_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto out_grad_data = out_grad.data_ptr<scalar_t>();
        auto out_x_data = out_x.data_ptr<scalar_t>();
        auto out_loc_data = out_loc.data_ptr<scalar_t>();
        auto out_scale_data = out_scale.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            kernel::probability::normal_cumulative_distribution_backward_backward<scalar_t>(
                gg_x_data[i], gg_loc_data[i], gg_scale_data[i],
                grad_data[i], x_data[i], loc_data[i], scale_data[i],
                out_grad_data[i], out_x_data[i], out_loc_data[i], out_scale_data[i]);
          }
        });
      });

  // Reduce gradients if inputs were broadcast
  return std::make_tuple(
      reduce_grad(out_grad, grad),
      reduce_grad(out_x, x),
      reduce_grad(out_loc, loc),
      reduce_grad(out_scale, scale));
}

at::Tensor normal_probability_density(
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  // Broadcast all inputs together
  auto tensors = at::broadcast_tensors({x, loc, scale});
  auto x_b = tensors[0].contiguous();
  auto loc_b = tensors[1].contiguous();
  auto scale_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "normal_probability_density_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto loc_data = loc_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::normal_probability_density<scalar_t>(
                x_data[i], loc_data[i], scale_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  // Broadcast all inputs
  auto tensors = at::broadcast_tensors({grad, x, loc, scale});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto loc_b = tensors[2].contiguous();
  auto scale_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_loc = at::empty_like(loc_b);
  auto grad_scale = at::empty_like(scale_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "normal_probability_density_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto loc_data = loc_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_loc_data = grad_loc.data_ptr<scalar_t>();
        auto grad_scale_data = grad_scale.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            kernel::probability::normal_probability_density_backward<scalar_t>(
                grad_data[i], x_data[i], loc_data[i], scale_data[i],
                grad_x_data[i], grad_loc_data[i], grad_scale_data[i]);
          }
        });
      });

  // Reduce gradients if inputs were broadcast
  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_loc, loc),
      reduce_grad(grad_scale, scale));
}

at::Tensor normal_quantile(
    const at::Tensor& p,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  // Broadcast all inputs together
  auto tensors = at::broadcast_tensors({p, loc, scale});
  auto p_b = tensors[0].contiguous();
  auto loc_b = tensors[1].contiguous();
  auto scale_b = tensors[2].contiguous();

  auto output = at::empty_like(p_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, p.scalar_type(), "normal_quantile_cpu", [&] {
        auto p_data = p_b.data_ptr<scalar_t>();
        auto loc_data = loc_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::normal_quantile<scalar_t>(
                p_data[i], loc_data[i], scale_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_quantile_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  // Broadcast all inputs
  auto tensors = at::broadcast_tensors({grad, p, loc, scale});
  auto grad_b = tensors[0].contiguous();
  auto p_b = tensors[1].contiguous();
  auto loc_b = tensors[2].contiguous();
  auto scale_b = tensors[3].contiguous();

  auto grad_p = at::empty_like(p_b);
  auto grad_loc = at::empty_like(loc_b);
  auto grad_scale = at::empty_like(scale_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, p.scalar_type(), "normal_quantile_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto p_data = p_b.data_ptr<scalar_t>();
        auto loc_data = loc_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto grad_p_data = grad_p.data_ptr<scalar_t>();
        auto grad_loc_data = grad_loc.data_ptr<scalar_t>();
        auto grad_scale_data = grad_scale.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            kernel::probability::normal_quantile_backward<scalar_t>(
                grad_data[i], p_data[i], loc_data[i], scale_data[i],
                grad_p_data[i], grad_loc_data[i], grad_scale_data[i]);
          }
        });
      });

  // Reduce gradients if inputs were broadcast
  return std::make_tuple(
      reduce_grad(grad_p, p),
      reduce_grad(grad_loc, loc),
      reduce_grad(grad_scale, scale));
}

at::Tensor normal_survival(
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  // Broadcast all inputs together
  auto tensors = at::broadcast_tensors({x, loc, scale});
  auto x_b = tensors[0].contiguous();
  auto loc_b = tensors[1].contiguous();
  auto scale_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "normal_survival_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto loc_data = loc_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::normal_survival<scalar_t>(
                x_data[i], loc_data[i], scale_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_survival_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  // Broadcast all inputs
  auto tensors = at::broadcast_tensors({grad, x, loc, scale});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto loc_b = tensors[2].contiguous();
  auto scale_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_loc = at::empty_like(loc_b);
  auto grad_scale = at::empty_like(scale_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "normal_survival_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto loc_data = loc_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_loc_data = grad_loc.data_ptr<scalar_t>();
        auto grad_scale_data = grad_scale.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            kernel::probability::normal_survival_backward<scalar_t>(
                grad_data[i], x_data[i], loc_data[i], scale_data[i],
                grad_x_data[i], grad_loc_data[i], grad_scale_data[i]);
          }
        });
      });

  // Reduce gradients if inputs were broadcast
  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_loc, loc),
      reduce_grad(grad_scale, scale));
}

at::Tensor normal_log_probability_density(
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  // Broadcast all inputs together
  auto tensors = at::broadcast_tensors({x, loc, scale});
  auto x_b = tensors[0].contiguous();
  auto loc_b = tensors[1].contiguous();
  auto scale_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "normal_log_probability_density_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto loc_data = loc_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::normal_log_probability_density<scalar_t>(
                x_data[i], loc_data[i], scale_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_log_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  // Broadcast all inputs
  auto tensors = at::broadcast_tensors({grad, x, loc, scale});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto loc_b = tensors[2].contiguous();
  auto scale_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_loc = at::empty_like(loc_b);
  auto grad_scale = at::empty_like(scale_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "normal_log_probability_density_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto loc_data = loc_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_loc_data = grad_loc.data_ptr<scalar_t>();
        auto grad_scale_data = grad_scale.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            kernel::probability::normal_log_probability_density_backward<scalar_t>(
                grad_data[i], x_data[i], loc_data[i], scale_data[i],
                grad_x_data[i], grad_loc_data[i], grad_scale_data[i]);
          }
        });
      });

  // Reduce gradients if inputs were broadcast
  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_loc, loc),
      reduce_grad(grad_scale, scale));
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("normal_cumulative_distribution", &normal_cumulative_distribution);
  m.impl("normal_cumulative_distribution_backward", &normal_cumulative_distribution_backward);
  m.impl("normal_cumulative_distribution_backward_backward", &normal_cumulative_distribution_backward_backward);
  m.impl("normal_probability_density", &normal_probability_density);
  m.impl("normal_probability_density_backward", &normal_probability_density_backward);
  m.impl("normal_quantile", &normal_quantile);
  m.impl("normal_quantile_backward", &normal_quantile_backward);
  m.impl("normal_survival", &normal_survival);
  m.impl("normal_survival_backward", &normal_survival_backward);
  m.impl("normal_log_probability_density", &normal_log_probability_density);
  m.impl("normal_log_probability_density_backward", &normal_log_probability_density_backward);
}

}  // namespace torchscience::cpu::probability
