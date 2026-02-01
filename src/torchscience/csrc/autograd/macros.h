#pragma once

#include <torch/extension.h>

#define TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(name, Name, arg)                  \
namespace torchscience::autograd::special_functions {                          \
                                                                               \
class Name##Backward : public torch::autograd::Function<Name##Backward> {      \
public:                                                                        \
  static std::vector<at::Tensor> forward(                                      \
    torch::autograd::AutogradContext *context,                                 \
    const at::Tensor &gradient_output,                                         \
    const at::Tensor &input,                                                   \
    bool input_requires_gradient                                               \
  ) {                                                                          \
    context->save_for_backward(                                                \
      {                                                                        \
        gradient_output,                                                       \
        input                                                                  \
      }                                                                        \
    );                                                                         \
                                                                               \
    context->saved_data["input_requires_grad"] = input_requires_gradient;      \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    static auto op = c10::Dispatcher::singleton()                              \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name "_backward",                                    \
        ""                                                                     \
      )                                                                        \
      .typed<at::Tensor(                                                       \
        const at::Tensor &,                                                    \
        const at::Tensor &                                                     \
      )>();                                                                    \
                                                                               \
    return {                                                                   \
      op.call(                                                                 \
        gradient_output,                                                       \
        input                                                                  \
      )                                                                        \
    };                                                                         \
  }                                                                            \
                                                                               \
  static std::vector<at::Tensor> backward(                                     \
    torch::autograd::AutogradContext *context,                                 \
    const std::vector<at::Tensor>                                              \
    &gradient_outputs                                                          \
  ) {                                                                          \
    torch::autograd::variable_list saved_variables = context->get_saved_variables(); \
                                                                               \
    bool input_requires_grad = context->saved_data["input_requires_grad"].toBool(); \
                                                                               \
    if (!gradient_outputs[0].defined() || !input_requires_grad) {              \
      return {                                                                 \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor()                                                           \
      };                                                                       \
    }                                                                          \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    auto [                                                                     \
      gradient_gradient_output,                                                \
      gradient_output                                                          \
    ] = c10::Dispatcher::singleton()                                           \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name "_backward_backward",                           \
        ""                                                                     \
      )                                                                        \
    .typed<std::tuple<                                                         \
      at::Tensor,                                                              \
      at::Tensor                                                               \
    >(                                                                         \
      const at::Tensor &,                                                      \
      const at::Tensor &,                                                      \
      const at::Tensor &                                                       \
    )>()                                                                       \
    .call(                                                                     \
      gradient_outputs[0],                                                     \
      saved_variables[0],                                                      \
      saved_variables[1]                                                       \
    );                                                                         \
                                                                               \
    return {                                                                   \
      gradient_gradient_output,                                                \
      gradient_output,                                                         \
      at::Tensor()                                                             \
    };                                                                         \
  }                                                                            \
};                                                                             \
                                                                               \
class Name : public torch::autograd::Function<Name> {                          \
public:                                                                        \
  static at::Tensor forward(                                                   \
    torch::autograd::AutogradContext *context,                                 \
    const at::Tensor &input                                                    \
  ) {                                                                          \
    context->save_for_backward(                                                \
      {                                                                        \
        input                                                                  \
      }                                                                        \
    );                                                                         \
                                                                               \
    context->saved_data["input_requires_grad"] = input.requires_grad() && (isFloatingType(input.scalar_type()) || isComplexType(input.scalar_type())); \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    return c10::Dispatcher::singleton()                                        \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name,                                                \
        ""                                                                     \
      )                                                                        \
      .typed<at::Tensor(                                                       \
        const at::Tensor &                                                     \
      )>()                                                                     \
      .call(                                                                   \
        input                                                                  \
      );                                                                       \
  }                                                                            \
                                                                               \
  static torch::autograd::variable_list backward(                              \
    torch::autograd::AutogradContext *context,                                 \
    const torch::autograd::variable_list &gradient_outputs                     \
  ) {                                                                          \
    torch::autograd::variable_list variables = context->get_saved_variables(); \
                                                                               \
    bool input_requires_grad = context->saved_data["input_requires_grad"].toBool(); \
                                                                               \
    auto gradients = Name##Backward::apply(                                    \
      gradient_outputs[0],                                                     \
      variables[0],                                                            \
      input_requires_grad                                                      \
    );                                                                         \
                                                                               \
    at::Tensor output;                                                         \
                                                                               \
    if (input_requires_grad) {                                                 \
      output = gradients[0];                                                   \
    } else {                                                                   \
      output = at::Tensor();                                                   \
    }                                                                          \
                                                                               \
    return {                                                                   \
      output                                                                   \
    };                                                                         \
  }                                                                            \
};                                                                             \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &arg                                                        \
) {                                                                            \
  return Name::apply(arg);                                                     \
}                                                                              \
                                                                               \
} /* namespace torchscience::autograd::special_functions */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {                           \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autograd::special_functions::name                            \
  );                                                                           \
}

#define TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(name, Name, arg1, arg2)          \
namespace torchscience::autograd::special_functions {                          \
                                                                               \
class Name##Backward : public torch::autograd::Function<Name##Backward> {      \
public:                                                                        \
  static std::vector<at::Tensor> forward(                                      \
    torch::autograd::AutogradContext *context,                                 \
    const at::Tensor &gradient_output,                                         \
    const at::Tensor &arg1##_input,                                            \
    const at::Tensor &arg2##_input,                                            \
    bool arg1##_requires_gradient,                                             \
    bool arg2##_requires_gradient                                              \
  ) {                                                                          \
    context->save_for_backward(                                                \
      {                                                                        \
        gradient_output,                                                       \
        arg1##_input,                                                          \
        arg2##_input                                                           \
      }                                                                        \
    );                                                                         \
                                                                               \
    context->saved_data[#arg1 "_requires_grad"] = arg1##_requires_gradient;    \
    context->saved_data[#arg2 "_requires_grad"] = arg2##_requires_gradient;    \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    static auto op = c10::Dispatcher::singleton()                              \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name "_backward",                                    \
        ""                                                                     \
      )                                                                        \
      .typed<std::tuple<at::Tensor, at::Tensor>(                               \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &                                                     \
      )>();                                                                    \
                                                                               \
    auto [arg1##_gradient, arg2##_gradient] = op.call(                         \
      gradient_output,                                                         \
      arg1##_input,                                                            \
      arg2##_input                                                             \
    );                                                                         \
                                                                               \
    return {                                                                   \
      arg1##_gradient,                                                         \
      arg2##_gradient                                                          \
    };                                                                         \
  }                                                                            \
                                                                               \
  static std::vector<at::Tensor> backward(                                     \
    torch::autograd::AutogradContext *context,                                 \
    const std::vector<at::Tensor> &gradient_outputs                            \
  ) {                                                                          \
    torch::autograd::variable_list saved_variables = context->get_saved_variables(); \
                                                                               \
    bool arg1##_requires_grad = context->saved_data[#arg1 "_requires_grad"].toBool(); \
    bool arg2##_requires_grad = context->saved_data[#arg2 "_requires_grad"].toBool(); \
                                                                               \
    if ((!gradient_outputs[0].defined() && !gradient_outputs[1].defined()) ||  \
        (!arg1##_requires_grad && !arg2##_requires_grad)) {                    \
      return {                                                                 \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor()                                                           \
      };                                                                       \
    }                                                                          \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    auto arg1##_gg = gradient_outputs[0].defined()                             \
      ? gradient_outputs[0]                                                    \
      : at::zeros_like(saved_variables[1]);                                    \
    auto arg2##_gg = gradient_outputs[1].defined()                             \
      ? gradient_outputs[1]                                                    \
      : at::zeros_like(saved_variables[2]);                                    \
                                                                               \
    auto [                                                                     \
      gradient_gradient_output,                                                \
      arg1##_gradient_output,                                                  \
      arg2##_gradient_output                                                   \
    ] = c10::Dispatcher::singleton()                                           \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name "_backward_backward",                           \
        ""                                                                     \
      )                                                                        \
      .typed<std::tuple<                                                       \
        at::Tensor,                                                            \
        at::Tensor,                                                            \
        at::Tensor                                                             \
      >(                                                                       \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &                                                     \
      )>()                                                                     \
      .call(                                                                   \
        arg1##_gg,                                                             \
        arg2##_gg,                                                             \
        saved_variables[0],                                                    \
        saved_variables[1],                                                    \
        saved_variables[2]                                                     \
      );                                                                       \
                                                                               \
    return {                                                                   \
      gradient_gradient_output,                                                \
      arg1##_gradient_output,                                                  \
      arg2##_gradient_output,                                                  \
      at::Tensor(),                                                            \
      at::Tensor()                                                             \
    };                                                                         \
  }                                                                            \
};                                                                             \
                                                                               \
class Name : public torch::autograd::Function<Name> {                          \
public:                                                                        \
  static at::Tensor forward(                                                   \
    torch::autograd::AutogradContext *context,                                 \
    const at::Tensor &arg1##_input,                                            \
    const at::Tensor &arg2##_input                                             \
  ) {                                                                          \
    context->save_for_backward(                                                \
      {                                                                        \
        arg1##_input,                                                          \
        arg2##_input                                                           \
      }                                                                        \
    );                                                                         \
                                                                               \
    context->saved_data[#arg1 "_requires_grad"] = arg1##_input.requires_grad() && (at::isFloatingType(arg1##_input.scalar_type()) || at::isComplexType(arg1##_input.scalar_type())); \
    context->saved_data[#arg2 "_requires_grad"] = arg2##_input.requires_grad() && (at::isFloatingType(arg2##_input.scalar_type()) || at::isComplexType(arg2##_input.scalar_type())); \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    return c10::Dispatcher::singleton()                                        \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name,                                                \
        ""                                                                     \
      )                                                                        \
      .typed<at::Tensor(                                                       \
        const at::Tensor &,                                                    \
        const at::Tensor &                                                     \
      )>()                                                                     \
      .call(                                                                   \
        arg1##_input,                                                          \
        arg2##_input                                                           \
      );                                                                       \
  }                                                                            \
                                                                               \
  static torch::autograd::variable_list backward(                              \
    torch::autograd::AutogradContext *context,                                 \
    const torch::autograd::variable_list &gradient_outputs                     \
  ) {                                                                          \
    torch::autograd::variable_list variables = context->get_saved_variables(); \
                                                                               \
    bool arg1##_requires_grad = context->saved_data[#arg1 "_requires_grad"].toBool(); \
    bool arg2##_requires_grad = context->saved_data[#arg2 "_requires_grad"].toBool(); \
                                                                               \
    auto gradients = Name##Backward::apply(                                    \
      gradient_outputs[0],                                                     \
      variables[0],                                                            \
      variables[1],                                                            \
      arg1##_requires_grad,                                                    \
      arg2##_requires_grad                                                     \
    );                                                                         \
                                                                               \
    return {                                                                   \
      arg1##_requires_grad ? gradients[0] : at::Tensor(),                      \
      arg2##_requires_grad ? gradients[1] : at::Tensor()                       \
    };                                                                         \
  }                                                                            \
};                                                                             \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &arg1,                                                      \
  const at::Tensor &arg2                                                       \
) {                                                                            \
  return Name::apply(arg1, arg2);                                              \
}                                                                              \
                                                                               \
} /* namespace torchscience::autograd::special_functions */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {                           \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autograd::special_functions::name                            \
  );                                                                           \
}

#define TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(name, Name, arg1, arg2, arg3)   \
namespace torchscience::autograd::special_functions {                          \
                                                                               \
class Name##Backward : public torch::autograd::Function<Name##Backward> {      \
public:                                                                        \
  static std::vector<at::Tensor> forward(                                      \
    torch::autograd::AutogradContext *context,                                 \
    const at::Tensor &gradient_output,                                         \
    const at::Tensor &arg1##_input,                                            \
    const at::Tensor &arg2##_input,                                            \
    const at::Tensor &arg3##_input,                                            \
    bool arg1##_requires_gradient,                                             \
    bool arg2##_requires_gradient,                                             \
    bool arg3##_requires_gradient                                              \
  ) {                                                                          \
    context->save_for_backward(                                                \
      {                                                                        \
        gradient_output,                                                       \
        arg1##_input,                                                          \
        arg2##_input,                                                          \
        arg3##_input                                                           \
      }                                                                        \
    );                                                                         \
                                                                               \
    context->saved_data[#arg1 "_requires_grad"] = arg1##_requires_gradient;    \
    context->saved_data[#arg2 "_requires_grad"] = arg2##_requires_gradient;    \
    context->saved_data[#arg3 "_requires_grad"] = arg3##_requires_gradient;    \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    static auto op = c10::Dispatcher::singleton()                              \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name "_backward",                                    \
        ""                                                                     \
      )                                                                        \
      .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(                   \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &                                                     \
      )>();                                                                    \
                                                                               \
    auto [arg1##_gradient, arg2##_gradient, arg3##_gradient] = op.call(        \
      gradient_output,                                                         \
      arg1##_input,                                                            \
      arg2##_input,                                                            \
      arg3##_input                                                             \
    );                                                                         \
                                                                               \
    return {                                                                   \
      arg1##_gradient,                                                         \
      arg2##_gradient,                                                         \
      arg3##_gradient                                                          \
    };                                                                         \
  }                                                                            \
                                                                               \
  static std::vector<at::Tensor> backward(                                     \
    torch::autograd::AutogradContext *context,                                 \
    const std::vector<at::Tensor> &gradient_outputs                            \
  ) {                                                                          \
    torch::autograd::variable_list saved_variables = context->get_saved_variables(); \
                                                                               \
    bool arg1##_requires_grad = context->saved_data[#arg1 "_requires_grad"].toBool(); \
    bool arg2##_requires_grad = context->saved_data[#arg2 "_requires_grad"].toBool(); \
    bool arg3##_requires_grad = context->saved_data[#arg3 "_requires_grad"].toBool(); \
                                                                               \
    if ((!gradient_outputs[0].defined() &&                                     \
         !gradient_outputs[1].defined() &&                                     \
         !gradient_outputs[2].defined()) ||                                    \
        (!arg1##_requires_grad &&                                              \
         !arg2##_requires_grad &&                                              \
         !arg3##_requires_grad)) {                                             \
      return {                                                                 \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor()                                                           \
      };                                                                       \
    }                                                                          \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    auto arg1##_gg = gradient_outputs[0].defined()                             \
      ? gradient_outputs[0]                                                    \
      : at::zeros_like(saved_variables[1]);                                    \
    auto arg2##_gg = gradient_outputs[1].defined()                             \
      ? gradient_outputs[1]                                                    \
      : at::zeros_like(saved_variables[2]);                                    \
    auto arg3##_gg = gradient_outputs[2].defined()                             \
      ? gradient_outputs[2]                                                    \
      : at::zeros_like(saved_variables[3]);                                    \
                                                                               \
    auto [                                                                     \
      gradient_gradient_output,                                                \
      arg1##_gradient_output,                                                  \
      arg2##_gradient_output,                                                  \
      arg3##_gradient_output                                                   \
    ] = c10::Dispatcher::singleton()                                           \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name "_backward_backward",                           \
        ""                                                                     \
      )                                                                        \
      .typed<std::tuple<                                                       \
        at::Tensor,                                                            \
        at::Tensor,                                                            \
        at::Tensor,                                                            \
        at::Tensor                                                             \
      >(                                                                       \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &                                                     \
      )>()                                                                     \
      .call(                                                                   \
        arg1##_gg,                                                             \
        arg2##_gg,                                                             \
        arg3##_gg,                                                             \
        saved_variables[0],                                                    \
        saved_variables[1],                                                    \
        saved_variables[2],                                                    \
        saved_variables[3]                                                     \
      );                                                                       \
                                                                               \
    return {                                                                   \
      gradient_gradient_output,                                                \
      arg1##_gradient_output,                                                  \
      arg2##_gradient_output,                                                  \
      arg3##_gradient_output,                                                  \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor()                                                             \
    };                                                                         \
  }                                                                            \
};                                                                             \
                                                                               \
class Name : public torch::autograd::Function<Name> {                          \
public:                                                                        \
  static at::Tensor forward(                                                   \
    torch::autograd::AutogradContext *context,                                 \
    const at::Tensor &arg1##_input,                                            \
    const at::Tensor &arg2##_input,                                            \
    const at::Tensor &arg3##_input                                             \
  ) {                                                                          \
    context->save_for_backward(                                                \
      {                                                                        \
        arg1##_input,                                                          \
        arg2##_input,                                                          \
        arg3##_input                                                           \
      }                                                                        \
    );                                                                         \
                                                                               \
    context->saved_data[#arg1 "_requires_grad"] = arg1##_input.requires_grad() && (at::isFloatingType(arg1##_input.scalar_type()) || at::isComplexType(arg1##_input.scalar_type())); \
    context->saved_data[#arg2 "_requires_grad"] = arg2##_input.requires_grad() && (at::isFloatingType(arg2##_input.scalar_type()) || at::isComplexType(arg2##_input.scalar_type())); \
    context->saved_data[#arg3 "_requires_grad"] = arg3##_input.requires_grad() && (at::isFloatingType(arg3##_input.scalar_type()) || at::isComplexType(arg3##_input.scalar_type())); \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    return c10::Dispatcher::singleton()                                        \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name,                                                \
        ""                                                                     \
      )                                                                        \
      .typed<at::Tensor(                                                       \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &                                                     \
      )>()                                                                     \
      .call(                                                                   \
        arg1##_input,                                                          \
        arg2##_input,                                                          \
        arg3##_input                                                           \
      );                                                                       \
  }                                                                            \
                                                                               \
  static torch::autograd::variable_list backward(                              \
    torch::autograd::AutogradContext *context,                                 \
    const torch::autograd::variable_list &gradient_outputs                     \
  ) {                                                                          \
    torch::autograd::variable_list variables = context->get_saved_variables(); \
                                                                               \
    bool arg1##_requires_grad = context->saved_data[#arg1 "_requires_grad"].toBool(); \
    bool arg2##_requires_grad = context->saved_data[#arg2 "_requires_grad"].toBool(); \
    bool arg3##_requires_grad = context->saved_data[#arg3 "_requires_grad"].toBool(); \
                                                                               \
    auto gradients = Name##Backward::apply(                                    \
      gradient_outputs[0],                                                     \
      variables[0],                                                            \
      variables[1],                                                            \
      variables[2],                                                            \
      arg1##_requires_grad,                                                    \
      arg2##_requires_grad,                                                    \
      arg3##_requires_grad                                                     \
    );                                                                         \
                                                                               \
    return {                                                                   \
      arg1##_requires_grad ? gradients[0] : at::Tensor(),                      \
      arg2##_requires_grad ? gradients[1] : at::Tensor(),                      \
      arg3##_requires_grad ? gradients[2] : at::Tensor()                       \
    };                                                                         \
  }                                                                            \
};                                                                             \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &arg1,                                                      \
  const at::Tensor &arg2,                                                      \
  const at::Tensor &arg3                                                       \
) {                                                                            \
  return Name::apply(arg1, arg2, arg3);                                        \
}                                                                              \
                                                                               \
} /* namespace torchscience::autograd::special_functions */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {                           \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autograd::special_functions::name                            \
  );                                                                           \
}

#define TORCHSCIENCE_AUTOGRAD_POINTWISE_QUATERNARY_OPERATOR(name, Name, arg1, arg2, arg3, arg4) \
namespace torchscience::autograd::special_functions {                          \
                                                                               \
class Name##Backward : public torch::autograd::Function<Name##Backward> {      \
public:                                                                        \
  static std::vector<at::Tensor> forward(                                      \
    torch::autograd::AutogradContext *context,                                 \
    const at::Tensor &gradient_output,                                         \
    const at::Tensor &arg1##_input,                                            \
    const at::Tensor &arg2##_input,                                            \
    const at::Tensor &arg3##_input,                                            \
    const at::Tensor &arg4##_input,                                            \
    bool arg1##_requires_gradient,                                             \
    bool arg2##_requires_gradient,                                             \
    bool arg3##_requires_gradient,                                             \
    bool arg4##_requires_gradient                                              \
  ) {                                                                          \
    context->save_for_backward(                                                \
      {                                                                        \
        gradient_output,                                                       \
        arg1##_input,                                                          \
        arg2##_input,                                                          \
        arg3##_input,                                                          \
        arg4##_input                                                           \
      }                                                                        \
    );                                                                         \
                                                                               \
    context->saved_data[#arg1 "_requires_grad"] = arg1##_requires_gradient;    \
    context->saved_data[#arg2 "_requires_grad"] = arg2##_requires_gradient;    \
    context->saved_data[#arg3 "_requires_grad"] = arg3##_requires_gradient;    \
    context->saved_data[#arg4 "_requires_grad"] = arg4##_requires_gradient;    \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    static auto op = c10::Dispatcher::singleton()                              \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name "_backward",                                    \
        ""                                                                     \
      )                                                                        \
      .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(       \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &                                                     \
      )>();                                                                    \
                                                                               \
    auto [arg1##_gradient, arg2##_gradient, arg3##_gradient, arg4##_gradient] = op.call( \
      gradient_output,                                                         \
      arg1##_input,                                                            \
      arg2##_input,                                                            \
      arg3##_input,                                                            \
      arg4##_input                                                             \
    );                                                                         \
                                                                               \
    return {                                                                   \
      arg1##_gradient,                                                         \
      arg2##_gradient,                                                         \
      arg3##_gradient,                                                         \
      arg4##_gradient                                                          \
    };                                                                         \
  }                                                                            \
                                                                               \
  static std::vector<at::Tensor> backward(                                     \
    torch::autograd::AutogradContext *context,                                 \
    const std::vector<at::Tensor> &gradient_outputs                            \
  ) {                                                                          \
    torch::autograd::variable_list saved_variables = context->get_saved_variables(); \
                                                                               \
    bool arg1##_requires_grad = context->saved_data[#arg1 "_requires_grad"].toBool(); \
    bool arg2##_requires_grad = context->saved_data[#arg2 "_requires_grad"].toBool(); \
    bool arg3##_requires_grad = context->saved_data[#arg3 "_requires_grad"].toBool(); \
    bool arg4##_requires_grad = context->saved_data[#arg4 "_requires_grad"].toBool(); \
                                                                               \
    if ((!gradient_outputs[0].defined() &&                                     \
         !gradient_outputs[1].defined() &&                                     \
         !gradient_outputs[2].defined() &&                                     \
         !gradient_outputs[3].defined()) ||                                    \
        (!arg1##_requires_grad &&                                              \
         !arg2##_requires_grad &&                                              \
         !arg3##_requires_grad &&                                              \
         !arg4##_requires_grad)) {                                             \
      return {                                                                 \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor()                                                           \
      };                                                                       \
    }                                                                          \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    auto arg1##_gg = gradient_outputs[0].defined()                             \
      ? gradient_outputs[0]                                                    \
      : at::zeros_like(saved_variables[1]);                                    \
    auto arg2##_gg = gradient_outputs[1].defined()                             \
      ? gradient_outputs[1]                                                    \
      : at::zeros_like(saved_variables[2]);                                    \
    auto arg3##_gg = gradient_outputs[2].defined()                             \
      ? gradient_outputs[2]                                                    \
      : at::zeros_like(saved_variables[3]);                                    \
    auto arg4##_gg = gradient_outputs[3].defined()                             \
      ? gradient_outputs[3]                                                    \
      : at::zeros_like(saved_variables[4]);                                    \
                                                                               \
    auto [                                                                     \
      gradient_gradient_output,                                                \
      arg1##_gradient_output,                                                  \
      arg2##_gradient_output,                                                  \
      arg3##_gradient_output,                                                  \
      arg4##_gradient_output                                                   \
    ] = c10::Dispatcher::singleton()                                           \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name "_backward_backward",                           \
        ""                                                                     \
      )                                                                        \
      .typed<std::tuple<                                                       \
        at::Tensor,                                                            \
        at::Tensor,                                                            \
        at::Tensor,                                                            \
        at::Tensor,                                                            \
        at::Tensor                                                             \
      >(                                                                       \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &                                                     \
      )>()                                                                     \
      .call(                                                                   \
        arg1##_gg,                                                             \
        arg2##_gg,                                                             \
        arg3##_gg,                                                             \
        arg4##_gg,                                                             \
        saved_variables[0],                                                    \
        saved_variables[1],                                                    \
        saved_variables[2],                                                    \
        saved_variables[3],                                                    \
        saved_variables[4]                                                     \
      );                                                                       \
                                                                               \
    return {                                                                   \
      gradient_gradient_output,                                                \
      arg1##_gradient_output,                                                  \
      arg2##_gradient_output,                                                  \
      arg3##_gradient_output,                                                  \
      arg4##_gradient_output,                                                  \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor()                                                             \
    };                                                                         \
  }                                                                            \
};                                                                             \
                                                                               \
class Name : public torch::autograd::Function<Name> {                          \
public:                                                                        \
  static at::Tensor forward(                                                   \
    torch::autograd::AutogradContext *context,                                 \
    const at::Tensor &arg1##_input,                                            \
    const at::Tensor &arg2##_input,                                            \
    const at::Tensor &arg3##_input,                                            \
    const at::Tensor &arg4##_input                                             \
  ) {                                                                          \
    context->save_for_backward(                                                \
      {                                                                        \
        arg1##_input,                                                          \
        arg2##_input,                                                          \
        arg3##_input,                                                          \
        arg4##_input                                                           \
      }                                                                        \
    );                                                                         \
                                                                               \
    context->saved_data[#arg1 "_requires_grad"] = arg1##_input.requires_grad() && (at::isFloatingType(arg1##_input.scalar_type()) || at::isComplexType(arg1##_input.scalar_type())); \
    context->saved_data[#arg2 "_requires_grad"] = arg2##_input.requires_grad() && (at::isFloatingType(arg2##_input.scalar_type()) || at::isComplexType(arg2##_input.scalar_type())); \
    context->saved_data[#arg3 "_requires_grad"] = arg3##_input.requires_grad() && (at::isFloatingType(arg3##_input.scalar_type()) || at::isComplexType(arg3##_input.scalar_type())); \
    context->saved_data[#arg4 "_requires_grad"] = arg4##_input.requires_grad() && (at::isFloatingType(arg4##_input.scalar_type()) || at::isComplexType(arg4##_input.scalar_type())); \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    return c10::Dispatcher::singleton()                                        \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name,                                                \
        ""                                                                     \
      )                                                                        \
      .typed<at::Tensor(                                                       \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &                                                     \
      )>()                                                                     \
      .call(                                                                   \
        arg1##_input,                                                          \
        arg2##_input,                                                          \
        arg3##_input,                                                          \
        arg4##_input                                                           \
      );                                                                       \
  }                                                                            \
                                                                               \
  static torch::autograd::variable_list backward(                              \
    torch::autograd::AutogradContext *context,                                 \
    const torch::autograd::variable_list &gradient_outputs                     \
  ) {                                                                          \
    torch::autograd::variable_list variables = context->get_saved_variables(); \
                                                                               \
    bool arg1##_requires_grad = context->saved_data[#arg1 "_requires_grad"].toBool(); \
    bool arg2##_requires_grad = context->saved_data[#arg2 "_requires_grad"].toBool(); \
    bool arg3##_requires_grad = context->saved_data[#arg3 "_requires_grad"].toBool(); \
    bool arg4##_requires_grad = context->saved_data[#arg4 "_requires_grad"].toBool(); \
                                                                               \
    auto gradients = Name##Backward::apply(                                    \
      gradient_outputs[0],                                                     \
      variables[0],                                                            \
      variables[1],                                                            \
      variables[2],                                                            \
      variables[3],                                                            \
      arg1##_requires_grad,                                                    \
      arg2##_requires_grad,                                                    \
      arg3##_requires_grad,                                                    \
      arg4##_requires_grad                                                     \
    );                                                                         \
                                                                               \
    return {                                                                   \
      arg1##_requires_grad ? gradients[0] : at::Tensor(),                      \
      arg2##_requires_grad ? gradients[1] : at::Tensor(),                      \
      arg3##_requires_grad ? gradients[2] : at::Tensor(),                      \
      arg4##_requires_grad ? gradients[3] : at::Tensor()                       \
    };                                                                         \
  }                                                                            \
};                                                                             \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &(arg1),                                                    \
  const at::Tensor &(arg2),                                                    \
  const at::Tensor &(arg3),                                                    \
  const at::Tensor &(arg4)                                                     \
) {                                                                            \
  return Name::apply(arg1, arg2, arg3, arg4);                                  \
}                                                                              \
                                                                               \
} /* namespace torchscience::autograd::special_functions */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {                           \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autograd::special_functions::name                            \
  );                                                                           \
}

#define TORCHSCIENCE_AUTOGRAD_POINTWISE_QUINARY_OPERATOR(name, Name, arg1, arg2, arg3, arg4, arg5) \
namespace torchscience::autograd::special_functions {                          \
                                                                               \
class Name##Backward : public torch::autograd::Function<Name##Backward> {      \
public:                                                                        \
  static std::vector<at::Tensor> forward(                                      \
    torch::autograd::AutogradContext *context,                                 \
    const at::Tensor &gradient_output,                                         \
    const at::Tensor &arg1##_input,                                            \
    const at::Tensor &arg2##_input,                                            \
    const at::Tensor &arg3##_input,                                            \
    const at::Tensor &arg4##_input,                                            \
    const at::Tensor &arg5##_input,                                            \
    bool arg1##_requires_gradient,                                             \
    bool arg2##_requires_gradient,                                             \
    bool arg3##_requires_gradient,                                             \
    bool arg4##_requires_gradient,                                             \
    bool arg5##_requires_gradient                                              \
  ) {                                                                          \
    context->save_for_backward(                                                \
      {                                                                        \
        gradient_output,                                                       \
        arg1##_input,                                                          \
        arg2##_input,                                                          \
        arg3##_input,                                                          \
        arg4##_input,                                                          \
        arg5##_input                                                           \
      }                                                                        \
    );                                                                         \
                                                                               \
    context->saved_data[#arg1 "_requires_grad"] = arg1##_requires_gradient;    \
    context->saved_data[#arg2 "_requires_grad"] = arg2##_requires_gradient;    \
    context->saved_data[#arg3 "_requires_grad"] = arg3##_requires_gradient;    \
    context->saved_data[#arg4 "_requires_grad"] = arg4##_requires_gradient;    \
    context->saved_data[#arg5 "_requires_grad"] = arg5##_requires_gradient;    \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    static auto op = c10::Dispatcher::singleton()                              \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name "_backward",                                    \
        ""                                                                     \
      )                                                                        \
      .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>( \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &                                                     \
      )>();                                                                    \
                                                                               \
    auto [arg1##_gradient, arg2##_gradient, arg3##_gradient, arg4##_gradient, arg5##_gradient] = op.call( \
      gradient_output,                                                         \
      arg1##_input,                                                            \
      arg2##_input,                                                            \
      arg3##_input,                                                            \
      arg4##_input,                                                            \
      arg5##_input                                                             \
    );                                                                         \
                                                                               \
    return {                                                                   \
      arg1##_gradient,                                                         \
      arg2##_gradient,                                                         \
      arg3##_gradient,                                                         \
      arg4##_gradient,                                                         \
      arg5##_gradient                                                          \
    };                                                                         \
  }                                                                            \
                                                                               \
  static std::vector<at::Tensor> backward(                                     \
    torch::autograd::AutogradContext *context,                                 \
    const std::vector<at::Tensor> &gradient_outputs                            \
  ) {                                                                          \
    torch::autograd::variable_list saved_variables = context->get_saved_variables(); \
                                                                               \
    bool arg1##_requires_grad = context->saved_data[#arg1 "_requires_grad"].toBool(); \
    bool arg2##_requires_grad = context->saved_data[#arg2 "_requires_grad"].toBool(); \
    bool arg3##_requires_grad = context->saved_data[#arg3 "_requires_grad"].toBool(); \
    bool arg4##_requires_grad = context->saved_data[#arg4 "_requires_grad"].toBool(); \
    bool arg5##_requires_grad = context->saved_data[#arg5 "_requires_grad"].toBool(); \
                                                                               \
    if ((!gradient_outputs[0].defined() &&                                     \
         !gradient_outputs[1].defined() &&                                     \
         !gradient_outputs[2].defined() &&                                     \
         !gradient_outputs[3].defined() &&                                     \
         !gradient_outputs[4].defined()) ||                                    \
        (!arg1##_requires_grad &&                                              \
         !arg2##_requires_grad &&                                              \
         !arg3##_requires_grad &&                                              \
         !arg4##_requires_grad &&                                              \
         !arg5##_requires_grad)) {                                             \
      return {                                                                 \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor()                                                           \
      };                                                                       \
    }                                                                          \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    auto arg1##_gg = gradient_outputs[0].defined()                             \
      ? gradient_outputs[0]                                                    \
      : at::zeros_like(saved_variables[1]);                                    \
    auto arg2##_gg = gradient_outputs[1].defined()                             \
      ? gradient_outputs[1]                                                    \
      : at::zeros_like(saved_variables[2]);                                    \
    auto arg3##_gg = gradient_outputs[2].defined()                             \
      ? gradient_outputs[2]                                                    \
      : at::zeros_like(saved_variables[3]);                                    \
    auto arg4##_gg = gradient_outputs[3].defined()                             \
      ? gradient_outputs[3]                                                    \
      : at::zeros_like(saved_variables[4]);                                    \
    auto arg5##_gg = gradient_outputs[4].defined()                             \
      ? gradient_outputs[4]                                                    \
      : at::zeros_like(saved_variables[5]);                                    \
                                                                               \
    auto [                                                                     \
      gradient_gradient_output,                                                \
      arg1##_gradient_output,                                                  \
      arg2##_gradient_output,                                                  \
      arg3##_gradient_output,                                                  \
      arg4##_gradient_output,                                                  \
      arg5##_gradient_output                                                   \
    ] = c10::Dispatcher::singleton()                                           \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name "_backward_backward",                           \
        ""                                                                     \
      )                                                                        \
      .typed<std::tuple<                                                       \
        at::Tensor,                                                            \
        at::Tensor,                                                            \
        at::Tensor,                                                            \
        at::Tensor,                                                            \
        at::Tensor,                                                            \
        at::Tensor                                                             \
      >(                                                                       \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &                                                     \
      )>()                                                                     \
      .call(                                                                   \
        arg1##_gg,                                                             \
        arg2##_gg,                                                             \
        arg3##_gg,                                                             \
        arg4##_gg,                                                             \
        arg5##_gg,                                                             \
        saved_variables[0],                                                    \
        saved_variables[1],                                                    \
        saved_variables[2],                                                    \
        saved_variables[3],                                                    \
        saved_variables[4],                                                    \
        saved_variables[5]                                                     \
      );                                                                       \
                                                                               \
    return {                                                                   \
      gradient_gradient_output,                                                \
      arg1##_gradient_output,                                                  \
      arg2##_gradient_output,                                                  \
      arg3##_gradient_output,                                                  \
      arg4##_gradient_output,                                                  \
      arg5##_gradient_output,                                                  \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor(),                                                            \
      at::Tensor()                                                             \
    };                                                                         \
  }                                                                            \
};                                                                             \
                                                                               \
class Name : public torch::autograd::Function<Name> {                          \
public:                                                                        \
  static at::Tensor forward(                                                   \
    torch::autograd::AutogradContext *context,                                 \
    const at::Tensor &arg1##_input,                                            \
    const at::Tensor &arg2##_input,                                            \
    const at::Tensor &arg3##_input,                                            \
    const at::Tensor &arg4##_input,                                            \
    const at::Tensor &arg5##_input                                             \
  ) {                                                                          \
    context->save_for_backward(                                                \
      {                                                                        \
        arg1##_input,                                                          \
        arg2##_input,                                                          \
        arg3##_input,                                                          \
        arg4##_input,                                                          \
        arg5##_input                                                           \
      }                                                                        \
    );                                                                         \
                                                                               \
    context->saved_data[#arg1 "_requires_grad"] = arg1##_input.requires_grad() && (at::isFloatingType(arg1##_input.scalar_type()) || at::isComplexType(arg1##_input.scalar_type())); \
    context->saved_data[#arg2 "_requires_grad"] = arg2##_input.requires_grad() && (at::isFloatingType(arg2##_input.scalar_type()) || at::isComplexType(arg2##_input.scalar_type())); \
    context->saved_data[#arg3 "_requires_grad"] = arg3##_input.requires_grad() && (at::isFloatingType(arg3##_input.scalar_type()) || at::isComplexType(arg3##_input.scalar_type())); \
    context->saved_data[#arg4 "_requires_grad"] = arg4##_input.requires_grad() && (at::isFloatingType(arg4##_input.scalar_type()) || at::isComplexType(arg4##_input.scalar_type())); \
    context->saved_data[#arg5 "_requires_grad"] = arg5##_input.requires_grad() && (at::isFloatingType(arg5##_input.scalar_type()) || at::isComplexType(arg5##_input.scalar_type())); \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    return c10::Dispatcher::singleton()                                        \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name,                                                \
        ""                                                                     \
      )                                                                        \
      .typed<at::Tensor(                                                       \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &,                                                    \
        const at::Tensor &                                                     \
      )>()                                                                     \
      .call(                                                                   \
        arg1##_input,                                                          \
        arg2##_input,                                                          \
        arg3##_input,                                                          \
        arg4##_input,                                                          \
        arg5##_input                                                           \
      );                                                                       \
  }                                                                            \
                                                                               \
  static torch::autograd::variable_list backward(                              \
    torch::autograd::AutogradContext *context,                                 \
    const torch::autograd::variable_list &gradient_outputs                     \
  ) {                                                                          \
    torch::autograd::variable_list variables = context->get_saved_variables(); \
                                                                               \
    bool arg1##_requires_grad = context->saved_data[#arg1 "_requires_grad"].toBool(); \
    bool arg2##_requires_grad = context->saved_data[#arg2 "_requires_grad"].toBool(); \
    bool arg3##_requires_grad = context->saved_data[#arg3 "_requires_grad"].toBool(); \
    bool arg4##_requires_grad = context->saved_data[#arg4 "_requires_grad"].toBool(); \
    bool arg5##_requires_grad = context->saved_data[#arg5 "_requires_grad"].toBool(); \
                                                                               \
    auto gradients = Name##Backward::apply(                                    \
      gradient_outputs[0],                                                     \
      variables[0],                                                            \
      variables[1],                                                            \
      variables[2],                                                            \
      variables[3],                                                            \
      variables[4],                                                            \
      arg1##_requires_grad,                                                    \
      arg2##_requires_grad,                                                    \
      arg3##_requires_grad,                                                    \
      arg4##_requires_grad,                                                    \
      arg5##_requires_grad                                                     \
    );                                                                         \
                                                                               \
    return {                                                                   \
      arg1##_requires_grad ? gradients[0] : at::Tensor(),                      \
      arg2##_requires_grad ? gradients[1] : at::Tensor(),                      \
      arg3##_requires_grad ? gradients[2] : at::Tensor(),                      \
      arg4##_requires_grad ? gradients[3] : at::Tensor(),                      \
      arg5##_requires_grad ? gradients[4] : at::Tensor()                       \
    };                                                                         \
  }                                                                            \
};                                                                             \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &(arg1),                                                    \
  const at::Tensor &(arg2),                                                    \
  const at::Tensor &(arg3),                                                    \
  const at::Tensor &(arg4),                                                    \
  const at::Tensor &(arg5)                                                     \
) {                                                                            \
  return Name::apply(arg1, arg2, arg3, arg4, arg5);                            \
}                                                                              \
                                                                               \
} /* namespace torchscience::autograd::special_functions */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {                           \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autograd::special_functions::name                            \
  );                                                                           \
}
