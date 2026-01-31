#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/probability/f_cumulative_distribution.h"
#include "../../kernel/probability/f_cumulative_distribution_backward.h"
#include "../../kernel/probability/f_probability_density.h"
#include "../../kernel/probability/f_probability_density_backward.h"
#include "../../kernel/probability/f_quantile.h"
#include "../../kernel/probability/f_quantile_backward.h"
#include "../../kernel/probability/f_survival.h"
#include "../../kernel/probability/f_survival_backward.h"
#include "../../kernel/probability/f_log_probability_density.h"
#include "../../kernel/probability/f_log_probability_density_backward.h"

namespace torchscience::cpu::probability {

// reduce_grad is defined in normal.h and already available in this namespace

at::Tensor f_cumulative_distribution(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({x, dfn, dfd});
  auto x_b = tensors[0].contiguous();
  auto dfn_b = tensors[1].contiguous();
  auto dfd_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "f_cumulative_distribution_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::f_cumulative_distribution<scalar_t>(
                x_data[i], dfn_data[i], dfd_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_cumulative_distribution_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({grad, x, dfn, dfd});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto dfn_b = tensors[2].contiguous();
  auto dfd_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_dfn = at::empty_like(dfn_b);
  auto grad_dfd = at::empty_like(dfd_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "f_cumulative_distribution_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_dfn_data = grad_dfn.data_ptr<scalar_t>();
        auto grad_dfd_data = grad_dfd.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, gdfn, gdfd] = kernel::probability::f_cumulative_distribution_backward<scalar_t>(
                grad_data[i], x_data[i], dfn_data[i], dfd_data[i]);
            grad_x_data[i] = gx;
            grad_dfn_data[i] = gdfn;
            grad_dfd_data[i] = gdfd;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_dfn, dfn),
      reduce_grad(grad_dfd, dfd));
}

at::Tensor f_probability_density(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({x, dfn, dfd});
  auto x_b = tensors[0].contiguous();
  auto dfn_b = tensors[1].contiguous();
  auto dfd_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "f_probability_density_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::f_probability_density<scalar_t>(
                x_data[i], dfn_data[i], dfd_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({grad, x, dfn, dfd});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto dfn_b = tensors[2].contiguous();
  auto dfd_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_dfn = at::empty_like(dfn_b);
  auto grad_dfd = at::empty_like(dfd_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "f_probability_density_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_dfn_data = grad_dfn.data_ptr<scalar_t>();
        auto grad_dfd_data = grad_dfd.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, gdfn, gdfd] = kernel::probability::f_probability_density_backward<scalar_t>(
                grad_data[i], x_data[i], dfn_data[i], dfd_data[i]);
            grad_x_data[i] = gx;
            grad_dfn_data[i] = gdfn;
            grad_dfd_data[i] = gdfd;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_dfn, dfn),
      reduce_grad(grad_dfd, dfd));
}

at::Tensor f_quantile(
    const at::Tensor& p,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({p, dfn, dfd});
  auto p_b = tensors[0].contiguous();
  auto dfn_b = tensors[1].contiguous();
  auto dfd_b = tensors[2].contiguous();

  auto output = at::empty_like(p_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, p.scalar_type(), "f_quantile_cpu", [&] {
        auto p_data = p_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::f_quantile<scalar_t>(
                p_data[i], dfn_data[i], dfd_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_quantile_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({grad, p, dfn, dfd});
  auto grad_b = tensors[0].contiguous();
  auto p_b = tensors[1].contiguous();
  auto dfn_b = tensors[2].contiguous();
  auto dfd_b = tensors[3].contiguous();

  auto grad_p = at::empty_like(p_b);
  auto grad_dfn = at::empty_like(dfn_b);
  auto grad_dfd = at::empty_like(dfd_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, p.scalar_type(), "f_quantile_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto p_data = p_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto grad_p_data = grad_p.data_ptr<scalar_t>();
        auto grad_dfn_data = grad_dfn.data_ptr<scalar_t>();
        auto grad_dfd_data = grad_dfd.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gp, gdfn, gdfd] = kernel::probability::f_quantile_backward<scalar_t>(
                grad_data[i], p_data[i], dfn_data[i], dfd_data[i]);
            grad_p_data[i] = gp;
            grad_dfn_data[i] = gdfn;
            grad_dfd_data[i] = gdfd;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_p, p),
      reduce_grad(grad_dfn, dfn),
      reduce_grad(grad_dfd, dfd));
}

at::Tensor f_survival(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({x, dfn, dfd});
  auto x_b = tensors[0].contiguous();
  auto dfn_b = tensors[1].contiguous();
  auto dfd_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "f_survival_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::f_survival<scalar_t>(
                x_data[i], dfn_data[i], dfd_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_survival_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({grad, x, dfn, dfd});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto dfn_b = tensors[2].contiguous();
  auto dfd_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_dfn = at::empty_like(dfn_b);
  auto grad_dfd = at::empty_like(dfd_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "f_survival_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_dfn_data = grad_dfn.data_ptr<scalar_t>();
        auto grad_dfd_data = grad_dfd.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, gdfn, gdfd] = kernel::probability::f_survival_backward<scalar_t>(
                grad_data[i], x_data[i], dfn_data[i], dfd_data[i]);
            grad_x_data[i] = gx;
            grad_dfn_data[i] = gdfn;
            grad_dfd_data[i] = gdfd;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_dfn, dfn),
      reduce_grad(grad_dfd, dfd));
}

at::Tensor f_log_probability_density(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({x, dfn, dfd});
  auto x_b = tensors[0].contiguous();
  auto dfn_b = tensors[1].contiguous();
  auto dfd_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "f_log_probability_density_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::f_log_probability_density<scalar_t>(
                x_data[i], dfn_data[i], dfd_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_log_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({grad, x, dfn, dfd});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto dfn_b = tensors[2].contiguous();
  auto dfd_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_dfn = at::empty_like(dfn_b);
  auto grad_dfd = at::empty_like(dfd_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "f_log_probability_density_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_dfn_data = grad_dfn.data_ptr<scalar_t>();
        auto grad_dfd_data = grad_dfd.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, gdfn, gdfd] = kernel::probability::f_log_probability_density_backward<scalar_t>(
                grad_data[i], x_data[i], dfn_data[i], dfd_data[i]);
            grad_x_data[i] = gx;
            grad_dfn_data[i] = gdfn;
            grad_dfd_data[i] = gdfd;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_dfn, dfn),
      reduce_grad(grad_dfd, dfd));
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("f_cumulative_distribution", &f_cumulative_distribution);
  m.impl("f_cumulative_distribution_backward", &f_cumulative_distribution_backward);
  m.impl("f_probability_density", &f_probability_density);
  m.impl("f_probability_density_backward", &f_probability_density_backward);
  m.impl("f_quantile", &f_quantile);
  m.impl("f_quantile_backward", &f_quantile_backward);
  m.impl("f_survival", &f_survival);
  m.impl("f_survival_backward", &f_survival_backward);
  m.impl("f_log_probability_density", &f_log_probability_density);
  m.impl("f_log_probability_density_backward", &f_log_probability_density_backward);
}

}  // namespace torchscience::cpu::probability
