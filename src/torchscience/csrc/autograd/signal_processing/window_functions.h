#pragma once

#include <ATen/ATen.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::window_function {

// =============================================================================
// Gaussian Window Autograd
// =============================================================================

class GaussianWindowBackward : public torch::autograd::Function<GaussianWindowBackward> {
public:
  static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t n,
    const at::Tensor& std_input,
    bool periodic,
    bool std_requires_grad
  ) {
    ctx->save_for_backward({grad_output, output, std_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["std_requires_grad"] = std_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_gaussian_window_backward"
      : "torchscience::gaussian_window_backward";

    auto grad_std = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
      .call(grad_output, output, n, std_input);

    return {grad_std};
  }

  static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grad_outputs
  ) {
    // Second-order gradients not implemented for window functions
    (void)ctx;
    (void)grad_outputs;
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

class GaussianWindow : public torch::autograd::Function<GaussianWindow> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    int64_t n,
    const at::Tensor& std_input,
    bool periodic,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device
  ) {
    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_gaussian_window"
      : "torchscience::gaussian_window";

    auto output = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>)>()
      .call(n, std_input, dtype, layout, device);

    ctx->save_for_backward({output, std_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["std_requires_grad"] = std_input.requires_grad() && at::isFloatingType(std_input.scalar_type());

    return output;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& grad_outputs
  ) {
    auto saved = ctx->get_saved_variables();
    int64_t n = ctx->saved_data["n"].toInt();
    bool periodic = ctx->saved_data["periodic"].toBool();
    bool std_requires_grad = ctx->saved_data["std_requires_grad"].toBool();

    at::Tensor grad_std;
    if (std_requires_grad && grad_outputs[0].defined()) {
      // Call the backward operator directly instead of through GaussianWindowBackward
      at::AutoDispatchBelowAutograd guard;

      std::string op_name = periodic
        ? "torchscience::periodic_gaussian_window_backward"
        : "torchscience::gaussian_window_backward";

      grad_std = c10::Dispatcher::singleton()
        .findSchemaOrThrow(op_name.c_str(), "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
        .call(grad_outputs[0], saved[0], n, saved[1]);
    }

    return {at::Tensor(), grad_std, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

inline at::Tensor gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return GaussianWindow::apply(n, std_input, false, dtype, layout, device);
}

inline at::Tensor periodic_gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return GaussianWindow::apply(n, std_input, true, dtype, layout, device);
}

// =============================================================================
// General Hamming Window Autograd
// =============================================================================

class GeneralHammingWindowBackward : public torch::autograd::Function<GeneralHammingWindowBackward> {
public:
  static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t n,
    const at::Tensor& alpha_input,
    bool periodic,
    bool alpha_requires_grad
  ) {
    ctx->save_for_backward({grad_output, output, alpha_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["alpha_requires_grad"] = alpha_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_general_hamming_window_backward"
      : "torchscience::general_hamming_window_backward";

    auto grad_alpha = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
      .call(grad_output, output, n, alpha_input);

    return {grad_alpha};
  }

  static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grad_outputs
  ) {
    // Second-order gradients not implemented for window functions
    (void)ctx;
    (void)grad_outputs;
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

class GeneralHammingWindow : public torch::autograd::Function<GeneralHammingWindow> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    int64_t n,
    const at::Tensor& alpha_input,
    bool periodic,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device
  ) {
    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_general_hamming_window"
      : "torchscience::general_hamming_window";

    auto output = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>)>()
      .call(n, alpha_input, dtype, layout, device);

    ctx->save_for_backward({output, alpha_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["alpha_requires_grad"] = alpha_input.requires_grad() && at::isFloatingType(alpha_input.scalar_type());

    return output;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& grad_outputs
  ) {
    auto saved = ctx->get_saved_variables();
    int64_t n = ctx->saved_data["n"].toInt();
    bool periodic = ctx->saved_data["periodic"].toBool();
    bool alpha_requires_grad = ctx->saved_data["alpha_requires_grad"].toBool();

    at::Tensor grad_alpha;
    if (alpha_requires_grad && grad_outputs[0].defined()) {
      // Call the backward operator directly instead of through GeneralHammingWindowBackward
      at::AutoDispatchBelowAutograd guard;

      std::string op_name = periodic
        ? "torchscience::periodic_general_hamming_window_backward"
        : "torchscience::general_hamming_window_backward";

      grad_alpha = c10::Dispatcher::singleton()
        .findSchemaOrThrow(op_name.c_str(), "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
        .call(grad_outputs[0], saved[0], n, saved[1]);
    }

    return {at::Tensor(), grad_alpha, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

inline at::Tensor general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return GeneralHammingWindow::apply(n, alpha_input, false, dtype, layout, device);
}

inline at::Tensor periodic_general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return GeneralHammingWindow::apply(n, alpha_input, true, dtype, layout, device);
}

// =============================================================================
// General Cosine Window Autograd
// =============================================================================

class GeneralCosineWindowBackward : public torch::autograd::Function<GeneralCosineWindowBackward> {
public:
  static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t n,
    const at::Tensor& coeffs_input,
    bool periodic,
    bool coeffs_requires_grad
  ) {
    ctx->save_for_backward({grad_output, output, coeffs_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_general_cosine_window_backward"
      : "torchscience::general_cosine_window_backward";

    auto grad_coeffs = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
      .call(grad_output, output, n, coeffs_input);

    return {grad_coeffs};
  }

  static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grad_outputs
  ) {
    // Second-order gradients not implemented for window functions
    (void)ctx;
    (void)grad_outputs;
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

class GeneralCosineWindow : public torch::autograd::Function<GeneralCosineWindow> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    int64_t n,
    const at::Tensor& coeffs_input,
    bool periodic,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device
  ) {
    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_general_cosine_window"
      : "torchscience::general_cosine_window";

    auto output = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>)>()
      .call(n, coeffs_input, dtype, layout, device);

    ctx->save_for_backward({output, coeffs_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["coeffs_requires_grad"] = coeffs_input.requires_grad() && at::isFloatingType(coeffs_input.scalar_type());

    return output;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& grad_outputs
  ) {
    auto saved = ctx->get_saved_variables();
    int64_t n = ctx->saved_data["n"].toInt();
    bool periodic = ctx->saved_data["periodic"].toBool();
    bool coeffs_requires_grad = ctx->saved_data["coeffs_requires_grad"].toBool();

    at::Tensor grad_coeffs;
    if (coeffs_requires_grad && grad_outputs[0].defined()) {
      // Call the backward operator directly instead of through GeneralCosineWindowBackward
      at::AutoDispatchBelowAutograd guard;

      std::string op_name = periodic
        ? "torchscience::periodic_general_cosine_window_backward"
        : "torchscience::general_cosine_window_backward";

      grad_coeffs = c10::Dispatcher::singleton()
        .findSchemaOrThrow(op_name.c_str(), "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
        .call(grad_outputs[0], saved[0], n, saved[1]);
    }

    return {at::Tensor(), grad_coeffs, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

inline at::Tensor general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return GeneralCosineWindow::apply(n, coeffs_input, false, dtype, layout, device);
}

inline at::Tensor periodic_general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return GeneralCosineWindow::apply(n, coeffs_input, true, dtype, layout, device);
}

// =============================================================================
// Tukey Window Autograd
// =============================================================================

class TukeyWindowBackward : public torch::autograd::Function<TukeyWindowBackward> {
public:
  static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t n,
    const at::Tensor& alpha_input,
    bool periodic,
    bool alpha_requires_grad
  ) {
    ctx->save_for_backward({grad_output, output, alpha_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["alpha_requires_grad"] = alpha_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_tukey_window_backward"
      : "torchscience::tukey_window_backward";

    auto grad_alpha = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
      .call(grad_output, output, n, alpha_input);

    return {grad_alpha};
  }

  static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grad_outputs
  ) {
    // Second-order gradients not implemented for window functions
    (void)ctx;
    (void)grad_outputs;
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

class TukeyWindow : public torch::autograd::Function<TukeyWindow> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    int64_t n,
    const at::Tensor& alpha_input,
    bool periodic,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device
  ) {
    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_tukey_window"
      : "torchscience::tukey_window";

    auto output = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>)>()
      .call(n, alpha_input, dtype, layout, device);

    ctx->save_for_backward({output, alpha_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["alpha_requires_grad"] = alpha_input.requires_grad() && at::isFloatingType(alpha_input.scalar_type());

    return output;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& grad_outputs
  ) {
    auto saved = ctx->get_saved_variables();
    int64_t n = ctx->saved_data["n"].toInt();
    bool periodic = ctx->saved_data["periodic"].toBool();
    bool alpha_requires_grad = ctx->saved_data["alpha_requires_grad"].toBool();

    at::Tensor grad_alpha;
    if (alpha_requires_grad && grad_outputs[0].defined()) {
      // Call the backward operator directly instead of through TukeyWindowBackward
      at::AutoDispatchBelowAutograd guard;

      std::string op_name = periodic
        ? "torchscience::periodic_tukey_window_backward"
        : "torchscience::tukey_window_backward";

      grad_alpha = c10::Dispatcher::singleton()
        .findSchemaOrThrow(op_name.c_str(), "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
        .call(grad_outputs[0], saved[0], n, saved[1]);
    }

    return {at::Tensor(), grad_alpha, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

inline at::Tensor tukey_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return TukeyWindow::apply(n, alpha_input, false, dtype, layout, device);
}

inline at::Tensor periodic_tukey_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return TukeyWindow::apply(n, alpha_input, true, dtype, layout, device);
}

// =============================================================================
// Exponential Window Autograd
// =============================================================================

class ExponentialWindowBackward : public torch::autograd::Function<ExponentialWindowBackward> {
public:
  static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t n,
    const at::Tensor& tau_input,
    bool periodic,
    bool tau_requires_grad
  ) {
    ctx->save_for_backward({grad_output, output, tau_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["tau_requires_grad"] = tau_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_exponential_window_backward"
      : "torchscience::exponential_window_backward";

    auto grad_tau = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
      .call(grad_output, output, n, tau_input);

    return {grad_tau};
  }

  static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grad_outputs
  ) {
    // Second-order gradients not implemented for window functions
    (void)ctx;
    (void)grad_outputs;
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

class ExponentialWindow : public torch::autograd::Function<ExponentialWindow> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    int64_t n,
    const at::Tensor& tau_input,
    bool periodic,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device
  ) {
    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_exponential_window"
      : "torchscience::exponential_window";

    auto output = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>)>()
      .call(n, tau_input, dtype, layout, device);

    ctx->save_for_backward({output, tau_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["tau_requires_grad"] = tau_input.requires_grad() && at::isFloatingType(tau_input.scalar_type());

    return output;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& grad_outputs
  ) {
    auto saved = ctx->get_saved_variables();
    int64_t n = ctx->saved_data["n"].toInt();
    bool periodic = ctx->saved_data["periodic"].toBool();
    bool tau_requires_grad = ctx->saved_data["tau_requires_grad"].toBool();

    at::Tensor grad_tau;
    if (tau_requires_grad && grad_outputs[0].defined()) {
      // Call the backward operator directly instead of through ExponentialWindowBackward
      at::AutoDispatchBelowAutograd guard;

      std::string op_name = periodic
        ? "torchscience::periodic_exponential_window_backward"
        : "torchscience::exponential_window_backward";

      grad_tau = c10::Dispatcher::singleton()
        .findSchemaOrThrow(op_name.c_str(), "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
        .call(grad_outputs[0], saved[0], n, saved[1]);
    }

    return {at::Tensor(), grad_tau, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

inline at::Tensor exponential_window(
  int64_t n,
  const at::Tensor& tau_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return ExponentialWindow::apply(n, tau_input, false, dtype, layout, device);
}

inline at::Tensor periodic_exponential_window(
  int64_t n,
  const at::Tensor& tau_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return ExponentialWindow::apply(n, tau_input, true, dtype, layout, device);
}

// =============================================================================
// Hann-Poisson Window Autograd
// =============================================================================

class HannPoissonWindowBackward : public torch::autograd::Function<HannPoissonWindowBackward> {
public:
  static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t n,
    const at::Tensor& alpha_input,
    bool periodic,
    bool alpha_requires_grad
  ) {
    ctx->save_for_backward({grad_output, output, alpha_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["alpha_requires_grad"] = alpha_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_hann_poisson_window_backward"
      : "torchscience::hann_poisson_window_backward";

    auto grad_alpha = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
      .call(grad_output, output, n, alpha_input);

    return {grad_alpha};
  }

  static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grad_outputs
  ) {
    // Second-order gradients not implemented for window functions
    (void)ctx;
    (void)grad_outputs;
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

class HannPoissonWindow : public torch::autograd::Function<HannPoissonWindow> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    int64_t n,
    const at::Tensor& alpha_input,
    bool periodic,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device
  ) {
    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_hann_poisson_window"
      : "torchscience::hann_poisson_window";

    auto output = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>)>()
      .call(n, alpha_input, dtype, layout, device);

    ctx->save_for_backward({output, alpha_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["alpha_requires_grad"] = alpha_input.requires_grad() && at::isFloatingType(alpha_input.scalar_type());

    return output;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& grad_outputs
  ) {
    auto saved = ctx->get_saved_variables();
    int64_t n = ctx->saved_data["n"].toInt();
    bool periodic = ctx->saved_data["periodic"].toBool();
    bool alpha_requires_grad = ctx->saved_data["alpha_requires_grad"].toBool();

    at::Tensor grad_alpha;
    if (alpha_requires_grad && grad_outputs[0].defined()) {
      at::AutoDispatchBelowAutograd guard;

      std::string op_name = periodic
        ? "torchscience::periodic_hann_poisson_window_backward"
        : "torchscience::hann_poisson_window_backward";

      grad_alpha = c10::Dispatcher::singleton()
        .findSchemaOrThrow(op_name.c_str(), "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
        .call(grad_outputs[0], saved[0], n, saved[1]);
    }

    return {at::Tensor(), grad_alpha, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

inline at::Tensor hann_poisson_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return HannPoissonWindow::apply(n, alpha_input, false, dtype, layout, device);
}

inline at::Tensor periodic_hann_poisson_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return HannPoissonWindow::apply(n, alpha_input, true, dtype, layout, device);
}

// =============================================================================
// Generalized Normal Window Autograd (two parameters: p and sigma)
// =============================================================================

class GeneralizedNormalWindow : public torch::autograd::Function<GeneralizedNormalWindow> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    int64_t n,
    const at::Tensor& p_input,
    const at::Tensor& sigma_input,
    bool periodic,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device
  ) {
    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_generalized_normal_window"
      : "torchscience::generalized_normal_window";

    auto output = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(int64_t, const at::Tensor&, const at::Tensor&, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>)>()
      .call(n, p_input, sigma_input, dtype, layout, device);

    ctx->save_for_backward({output, p_input, sigma_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["p_requires_grad"] = p_input.requires_grad() && at::isFloatingType(p_input.scalar_type());
    ctx->saved_data["sigma_requires_grad"] = sigma_input.requires_grad() && at::isFloatingType(sigma_input.scalar_type());

    return output;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& grad_outputs
  ) {
    auto saved = ctx->get_saved_variables();
    int64_t n = ctx->saved_data["n"].toInt();
    bool periodic = ctx->saved_data["periodic"].toBool();
    bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();
    bool sigma_requires_grad = ctx->saved_data["sigma_requires_grad"].toBool();

    at::Tensor grad_p;
    at::Tensor grad_sigma;

    if ((p_requires_grad || sigma_requires_grad) && grad_outputs[0].defined()) {
      at::AutoDispatchBelowAutograd guard;

      std::string op_name = periodic
        ? "torchscience::periodic_generalized_normal_window_backward"
        : "torchscience::generalized_normal_window_backward";

      auto grads = c10::Dispatcher::singleton()
        .findSchemaOrThrow(op_name.c_str(), "")
        .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&)>()
        .call(grad_outputs[0], saved[0], n, saved[1], saved[2]);

      if (p_requires_grad) {
        grad_p = std::get<0>(grads);
      }
      if (sigma_requires_grad) {
        grad_sigma = std::get<1>(grads);
      }
    }

    // Returns: n (no grad), p, sigma, periodic (no grad), dtype (no grad), layout (no grad), device (no grad)
    return {at::Tensor(), grad_p, grad_sigma, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

inline at::Tensor generalized_normal_window(
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return GeneralizedNormalWindow::apply(n, p_input, sigma_input, false, dtype, layout, device);
}

inline at::Tensor periodic_generalized_normal_window(
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return GeneralizedNormalWindow::apply(n, p_input, sigma_input, true, dtype, layout, device);
}

// =============================================================================
// Kaiser Window Autograd
// =============================================================================

class KaiserWindowBackward : public torch::autograd::Function<KaiserWindowBackward> {
public:
  static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t n,
    const at::Tensor& beta_input,
    bool periodic,
    bool beta_requires_grad
  ) {
    ctx->save_for_backward({grad_output, output, beta_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["beta_requires_grad"] = beta_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_kaiser_window_backward"
      : "torchscience::kaiser_window_backward";

    auto grad_beta = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
      .call(grad_output, output, n, beta_input);

    return {grad_beta};
  }

  static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grad_outputs
  ) {
    // Second-order gradients not implemented for window functions
    (void)ctx;
    (void)grad_outputs;
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

class KaiserWindow : public torch::autograd::Function<KaiserWindow> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    int64_t n,
    const at::Tensor& beta_input,
    bool periodic,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device
  ) {
    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_kaiser_window"
      : "torchscience::kaiser_window";

    auto output = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>)>()
      .call(n, beta_input, dtype, layout, device);

    ctx->save_for_backward({output, beta_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["beta_requires_grad"] = beta_input.requires_grad() && at::isFloatingType(beta_input.scalar_type());

    return output;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& grad_outputs
  ) {
    auto saved = ctx->get_saved_variables();
    int64_t n = ctx->saved_data["n"].toInt();
    bool periodic = ctx->saved_data["periodic"].toBool();
    bool beta_requires_grad = ctx->saved_data["beta_requires_grad"].toBool();

    at::Tensor grad_beta;
    if (beta_requires_grad && grad_outputs[0].defined()) {
      at::AutoDispatchBelowAutograd guard;

      std::string op_name = periodic
        ? "torchscience::periodic_kaiser_window_backward"
        : "torchscience::kaiser_window_backward";

      grad_beta = c10::Dispatcher::singleton()
        .findSchemaOrThrow(op_name.c_str(), "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
        .call(grad_outputs[0], saved[0], n, saved[1]);
    }

    return {at::Tensor(), grad_beta, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

inline at::Tensor kaiser_window(
  int64_t n,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return KaiserWindow::apply(n, beta_input, false, dtype, layout, device);
}

inline at::Tensor periodic_kaiser_window(
  int64_t n,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return KaiserWindow::apply(n, beta_input, true, dtype, layout, device);
}

// =============================================================================
// Planck-taper Window Autograd
// =============================================================================

class PlanckTaperWindowBackward : public torch::autograd::Function<PlanckTaperWindowBackward> {
public:
  static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t n,
    const at::Tensor& epsilon_input,
    bool periodic,
    bool epsilon_requires_grad
  ) {
    ctx->save_for_backward({grad_output, output, epsilon_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["epsilon_requires_grad"] = epsilon_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_planck_taper_window_backward"
      : "torchscience::planck_taper_window_backward";

    auto grad_epsilon = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
      .call(grad_output, output, n, epsilon_input);

    return {grad_epsilon};
  }

  static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grad_outputs
  ) {
    (void)ctx;
    (void)grad_outputs;
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

class PlanckTaperWindow : public torch::autograd::Function<PlanckTaperWindow> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    int64_t n,
    const at::Tensor& epsilon_input,
    bool periodic,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device
  ) {
    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_planck_taper_window"
      : "torchscience::planck_taper_window";

    auto output = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>)>()
      .call(n, epsilon_input, dtype, layout, device);

    ctx->save_for_backward({output, epsilon_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["epsilon_requires_grad"] = epsilon_input.requires_grad() && at::isFloatingType(epsilon_input.scalar_type());

    return output;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& grad_outputs
  ) {
    auto saved = ctx->get_saved_variables();
    int64_t n = ctx->saved_data["n"].toInt();
    bool periodic = ctx->saved_data["periodic"].toBool();
    bool epsilon_requires_grad = ctx->saved_data["epsilon_requires_grad"].toBool();

    at::Tensor grad_epsilon;
    if (epsilon_requires_grad && grad_outputs[0].defined()) {
      at::AutoDispatchBelowAutograd guard;

      std::string op_name = periodic
        ? "torchscience::periodic_planck_taper_window_backward"
        : "torchscience::planck_taper_window_backward";

      grad_epsilon = c10::Dispatcher::singleton()
        .findSchemaOrThrow(op_name.c_str(), "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
        .call(grad_outputs[0], saved[0], n, saved[1]);
    }

    return {at::Tensor(), grad_epsilon, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

inline at::Tensor planck_taper_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return PlanckTaperWindow::apply(n, epsilon_input, false, dtype, layout, device);
}

inline at::Tensor periodic_planck_taper_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return PlanckTaperWindow::apply(n, epsilon_input, true, dtype, layout, device);
}

// =============================================================================
// Planck-Bessel Window Autograd (two parameters: epsilon and beta)
// =============================================================================

class PlanckBesselWindow : public torch::autograd::Function<PlanckBesselWindow> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    int64_t n,
    const at::Tensor& epsilon_input,
    const at::Tensor& beta_input,
    bool periodic,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device
  ) {
    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_planck_bessel_window"
      : "torchscience::planck_bessel_window";

    auto output = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name.c_str(), "")
      .typed<at::Tensor(int64_t, const at::Tensor&, const at::Tensor&, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>)>()
      .call(n, epsilon_input, beta_input, dtype, layout, device);

    ctx->save_for_backward({output, epsilon_input, beta_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["epsilon_requires_grad"] = epsilon_input.requires_grad() && at::isFloatingType(epsilon_input.scalar_type());
    ctx->saved_data["beta_requires_grad"] = beta_input.requires_grad() && at::isFloatingType(beta_input.scalar_type());

    return output;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& grad_outputs
  ) {
    auto saved = ctx->get_saved_variables();
    int64_t n = ctx->saved_data["n"].toInt();
    bool periodic = ctx->saved_data["periodic"].toBool();
    bool epsilon_requires_grad = ctx->saved_data["epsilon_requires_grad"].toBool();
    bool beta_requires_grad = ctx->saved_data["beta_requires_grad"].toBool();

    at::Tensor grad_epsilon;
    at::Tensor grad_beta;

    if ((epsilon_requires_grad || beta_requires_grad) && grad_outputs[0].defined()) {
      at::AutoDispatchBelowAutograd guard;

      std::string op_name = periodic
        ? "torchscience::periodic_planck_bessel_window_backward"
        : "torchscience::planck_bessel_window_backward";

      auto grads = c10::Dispatcher::singleton()
        .findSchemaOrThrow(op_name.c_str(), "")
        .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&)>()
        .call(grad_outputs[0], saved[0], n, saved[1], saved[2]);

      if (epsilon_requires_grad) {
        grad_epsilon = std::get<0>(grads);
      }
      if (beta_requires_grad) {
        grad_beta = std::get<1>(grads);
      }
    }

    // Returns: n (no grad), epsilon, beta, periodic (no grad), dtype (no grad), layout (no grad), device (no grad)
    return {at::Tensor(), grad_epsilon, grad_beta, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

inline at::Tensor planck_bessel_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return PlanckBesselWindow::apply(n, epsilon_input, beta_input, false, dtype, layout, device);
}

inline at::Tensor periodic_planck_bessel_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return PlanckBesselWindow::apply(n, epsilon_input, beta_input, true, dtype, layout, device);
}

}  // namespace torchscience::autograd::window_function

// =============================================================================
// TORCH_LIBRARY_IMPL registrations
// =============================================================================

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("gaussian_window", torchscience::autograd::window_function::gaussian_window);
  m.impl("periodic_gaussian_window", torchscience::autograd::window_function::periodic_gaussian_window);

  m.impl("general_hamming_window", torchscience::autograd::window_function::general_hamming_window);
  m.impl("periodic_general_hamming_window", torchscience::autograd::window_function::periodic_general_hamming_window);

  m.impl("general_cosine_window", torchscience::autograd::window_function::general_cosine_window);
  m.impl("periodic_general_cosine_window", torchscience::autograd::window_function::periodic_general_cosine_window);

  m.impl("tukey_window", torchscience::autograd::window_function::tukey_window);
  m.impl("periodic_tukey_window", torchscience::autograd::window_function::periodic_tukey_window);

  m.impl("exponential_window", torchscience::autograd::window_function::exponential_window);
  m.impl("periodic_exponential_window", torchscience::autograd::window_function::periodic_exponential_window);

  m.impl("hann_poisson_window", torchscience::autograd::window_function::hann_poisson_window);
  m.impl("periodic_hann_poisson_window", torchscience::autograd::window_function::periodic_hann_poisson_window);

  m.impl("generalized_normal_window", torchscience::autograd::window_function::generalized_normal_window);
  m.impl("periodic_generalized_normal_window", torchscience::autograd::window_function::periodic_generalized_normal_window);

  m.impl("kaiser_window", torchscience::autograd::window_function::kaiser_window);
  m.impl("periodic_kaiser_window", torchscience::autograd::window_function::periodic_kaiser_window);

  m.impl("planck_taper_window", torchscience::autograd::window_function::planck_taper_window);
  m.impl("periodic_planck_taper_window", torchscience::autograd::window_function::periodic_planck_taper_window);

  m.impl("planck_bessel_window", torchscience::autograd::window_function::planck_bessel_window);
  m.impl("periodic_planck_bessel_window", torchscience::autograd::window_function::periodic_planck_bessel_window);
}
