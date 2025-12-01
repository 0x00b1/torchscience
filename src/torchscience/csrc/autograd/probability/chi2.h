#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::probability {

/**
 * Autograd function for chi2 CDF.
 */
class Chi2CdfFunction : public torch::autograd::Function<Chi2CdfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x,
      const at::Tensor& df
  ) {
    ctx->save_for_backward({x, df});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::chi2_cumulative_distribution", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(x, df);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor x = saved[0];
    at::Tensor df = saved[1];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::chi2_cumulative_distribution_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, x, df);

    return {std::get<0>(result), std::get<1>(result)};
  }
};

inline at::Tensor chi2_cumulative_distribution(
    const at::Tensor& x,
    const at::Tensor& df
) {
  return Chi2CdfFunction::apply(x, df);
}

/**
 * Autograd function for chi2 PDF.
 */
class Chi2PdfFunction : public torch::autograd::Function<Chi2PdfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x,
      const at::Tensor& df
  ) {
    ctx->save_for_backward({x, df});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::chi2_probability_density", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(x, df);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor x = saved[0];
    at::Tensor df = saved[1];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::chi2_probability_density_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, x, df);

    return {std::get<0>(result), std::get<1>(result)};
  }
};

inline at::Tensor chi2_probability_density(
    const at::Tensor& x,
    const at::Tensor& df
) {
  return Chi2PdfFunction::apply(x, df);
}

/**
 * Autograd function for chi2 PPF.
 */
class Chi2PpfFunction : public torch::autograd::Function<Chi2PpfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& p,
      const at::Tensor& df
  ) {
    ctx->save_for_backward({p, df});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::chi2_quantile", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(p, df);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor p = saved[0];
    at::Tensor df = saved[1];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::chi2_quantile_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, p, df);

    return {std::get<0>(result), std::get<1>(result)};
  }
};

inline at::Tensor chi2_quantile(
    const at::Tensor& p,
    const at::Tensor& df
) {
  return Chi2PpfFunction::apply(p, df);
}

/**
 * Autograd function for chi2 SF (survival function).
 */
class Chi2SfFunction : public torch::autograd::Function<Chi2SfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x,
      const at::Tensor& df
  ) {
    ctx->save_for_backward({x, df});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::chi2_survival", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(x, df);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor x = saved[0];
    at::Tensor df = saved[1];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::chi2_survival_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, x, df);

    return {std::get<0>(result), std::get<1>(result)};
  }
};

inline at::Tensor chi2_survival(
    const at::Tensor& x,
    const at::Tensor& df
) {
  return Chi2SfFunction::apply(x, df);
}

}  // namespace torchscience::autograd::probability

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("chi2_cumulative_distribution", &torchscience::autograd::probability::chi2_cumulative_distribution);
  m.impl("chi2_probability_density", &torchscience::autograd::probability::chi2_probability_density);
  m.impl("chi2_quantile", &torchscience::autograd::probability::chi2_quantile);
  m.impl("chi2_survival", &torchscience::autograd::probability::chi2_survival);
}
