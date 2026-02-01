#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

#define TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(name, arg1)                           \
namespace torchscience::meta::special_functions {                              \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &arg1##_input                                               \
) {                                                                            \
  at::Tensor output;                                                           \
                                                                               \
  return at::TensorIteratorConfig()                                            \
    .add_output(output)                                                        \
    .add_const_input(arg1##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build()                                                                   \
    .output();                                                                 \
}                                                                              \
                                                                               \
inline at::Tensor name##_backward(                                             \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input                                               \
) {                                                                            \
  at::Tensor gradient_output;                                                  \
                                                                               \
  return at::TensorIteratorConfig()                                            \
    .add_output(gradient_output)                                               \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build()                                                                   \
    .output();                                                                 \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(            \
  const at::Tensor &gradient_gradient_input,                                   \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input                                               \
) {                                                                            \
  if (!gradient_gradient_input.defined()) {                                    \
    return {};                                                                 \
  }                                                                            \
                                                                               \
  at::Tensor gradient_gradient_output;                                         \
  at::Tensor gradient_output;                                                  \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_gradient_output)                                      \
    .add_output(gradient_output)                                               \
    .add_const_input(gradient_gradient_input)                                  \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
} /* namespace torchscience::meta::special_functions */                        \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                               \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::meta::special_functions::name                                \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward",                                                         \
    torchscience::meta::special_functions::name##_backward                     \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward_backward",                                                \
    torchscience::meta::special_functions::name##_backward_backward            \
  );                                                                           \
}

#define TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(name, arg1, arg2)                    \
namespace torchscience::meta::special_functions {                              \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input                                               \
) {                                                                            \
  at::Tensor output;                                                           \
                                                                               \
  return at::TensorIteratorConfig()                                            \
    .add_output(output)                                                        \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build()                                                                   \
    .output();                                                                 \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor> name##_backward(                     \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input                                               \
) {                                                                            \
  at::Tensor gradient_##arg1;                                                  \
  at::Tensor gradient_##arg2;                                                  \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_##arg1)                                               \
    .add_output(gradient_##arg2)                                               \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> name##_backward_backward(\
  const at::Tensor &gradient_gradient_##arg1##_input,                          \
  const at::Tensor &gradient_gradient_##arg2##_input,                          \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input                                               \
) {                                                                            \
  at::Tensor gradient_gradient_output;                                         \
  at::Tensor gradient_##arg1;                                                  \
  at::Tensor gradient_##arg2;                                                  \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_gradient_output)                                      \
    .add_output(gradient_##arg1)                                               \
    .add_output(gradient_##arg2)                                               \
    .add_const_input(gradient_gradient_##arg1##_input)                         \
    .add_const_input(gradient_gradient_##arg2##_input)                         \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1),                                                        \
    iterator.output(2)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
} /* namespace torchscience::meta::special_functions */                        \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                               \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::meta::special_functions::name                                \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward",                                                         \
    torchscience::meta::special_functions::name##_backward                     \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward_backward",                                                \
    torchscience::meta::special_functions::name##_backward_backward            \
  );                                                                           \
}

#define TORCHSCIENCE_META_POINTWISE_TERNARY_OPERATOR(name, arg1, arg2, arg3)             \
namespace torchscience::meta::special_functions {                              \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input                                               \
) {                                                                            \
  at::Tensor output;                                                           \
                                                                               \
  return at::TensorIteratorConfig()                                            \
    .add_output(output)                                                        \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build()                                                                   \
    .output();                                                                 \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> name##_backward(         \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input                                               \
) {                                                                            \
  at::Tensor gradient_##arg1;                                                  \
  at::Tensor gradient_##arg2;                                                  \
  at::Tensor gradient_##arg3;                                                  \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_##arg1)                                               \
    .add_output(gradient_##arg2)                                               \
    .add_output(gradient_##arg3)                                               \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1),                                                        \
    iterator.output(2)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>              \
name##_backward_backward(                                                      \
  const at::Tensor &gradient_gradient_##arg1##_input,                          \
  const at::Tensor &gradient_gradient_##arg2##_input,                          \
  const at::Tensor &gradient_gradient_##arg3##_input,                          \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input                                               \
) {                                                                            \
  at::Tensor gradient_gradient_output;                                         \
  at::Tensor gradient_##arg1;                                                  \
  at::Tensor gradient_##arg2;                                                  \
  at::Tensor gradient_##arg3;                                                  \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_gradient_output)                                      \
    .add_output(gradient_##arg1)                                               \
    .add_output(gradient_##arg2)                                               \
    .add_output(gradient_##arg3)                                               \
    .add_const_input(gradient_gradient_##arg1##_input)                         \
    .add_const_input(gradient_gradient_##arg2##_input)                         \
    .add_const_input(gradient_gradient_##arg3##_input)                         \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1),                                                        \
    iterator.output(2),                                                        \
    iterator.output(3)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
} /* namespace torchscience::meta::special_functions */                        \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                               \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::meta::special_functions::name                                \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward",                                                         \
    torchscience::meta::special_functions::name##_backward                     \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward_backward",                                                \
    torchscience::meta::special_functions::name##_backward_backward            \
  );                                                                           \
}

#define TORCHSCIENCE_META_POINTWISE_QUATERNARY_OPERATOR(name, arg1, arg2, arg3, arg4)    \
namespace torchscience::meta::special_functions {                              \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input,                                              \
  const at::Tensor &arg4##_input                                               \
) {                                                                            \
  at::Tensor output;                                                           \
                                                                               \
  return at::TensorIteratorConfig()                                            \
    .add_output(output)                                                        \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .add_const_input(arg4##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build()                                                                   \
    .output();                                                                 \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>              \
name##_backward(                                                               \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input,                                              \
  const at::Tensor &arg4##_input                                               \
) {                                                                            \
  at::Tensor gradient_##arg1;                                                  \
  at::Tensor gradient_##arg2;                                                  \
  at::Tensor gradient_##arg3;                                                  \
  at::Tensor gradient_##arg4;                                                  \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_##arg1)                                               \
    .add_output(gradient_##arg2)                                               \
    .add_output(gradient_##arg3)                                               \
    .add_output(gradient_##arg4)                                               \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .add_const_input(arg4##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1),                                                        \
    iterator.output(2),                                                        \
    iterator.output(3)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>  \
name##_backward_backward(                                                      \
  const at::Tensor &gradient_gradient_##arg1##_input,                          \
  const at::Tensor &gradient_gradient_##arg2##_input,                          \
  const at::Tensor &gradient_gradient_##arg3##_input,                          \
  const at::Tensor &gradient_gradient_##arg4##_input,                          \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input,                                              \
  const at::Tensor &arg4##_input                                               \
) {                                                                            \
  at::Tensor gradient_gradient_output;                                         \
  at::Tensor gradient_##arg1;                                                  \
  at::Tensor gradient_##arg2;                                                  \
  at::Tensor gradient_##arg3;                                                  \
  at::Tensor gradient_##arg4;                                                  \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_gradient_output)                                      \
    .add_output(gradient_##arg1)                                               \
    .add_output(gradient_##arg2)                                               \
    .add_output(gradient_##arg3)                                               \
    .add_output(gradient_##arg4)                                               \
    .add_const_input(gradient_gradient_##arg1##_input)                         \
    .add_const_input(gradient_gradient_##arg2##_input)                         \
    .add_const_input(gradient_gradient_##arg3##_input)                         \
    .add_const_input(gradient_gradient_##arg4##_input)                         \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .add_const_input(arg4##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1),                                                        \
    iterator.output(2),                                                        \
    iterator.output(3),                                                        \
    iterator.output(4)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
} /* namespace torchscience::meta::special_functions */                        \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                               \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::meta::special_functions::name                                \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward",                                                         \
    torchscience::meta::special_functions::name##_backward                     \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward_backward",                                                \
    torchscience::meta::special_functions::name##_backward_backward            \
  );                                                                           \
}

#define TORCHSCIENCE_META_POINTWISE_QUINARY_OPERATOR(name, arg1, arg2, arg3, arg4, arg5) \
namespace torchscience::meta::special_functions {                              \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input,                                              \
  const at::Tensor &arg4##_input,                                              \
  const at::Tensor &arg5##_input                                               \
) {                                                                            \
  at::Tensor output;                                                           \
                                                                               \
  return at::TensorIteratorConfig()                                            \
    .add_output(output)                                                        \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .add_const_input(arg4##_input)                                             \
    .add_const_input(arg5##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build()                                                                   \
    .output();                                                                 \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>  \
name##_backward(                                                               \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input,                                              \
  const at::Tensor &arg4##_input,                                              \
  const at::Tensor &arg5##_input                                               \
) {                                                                            \
  at::Tensor gradient_##arg1;                                                  \
  at::Tensor gradient_##arg2;                                                  \
  at::Tensor gradient_##arg3;                                                  \
  at::Tensor gradient_##arg4;                                                  \
  at::Tensor gradient_##arg5;                                                  \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_##arg1)                                               \
    .add_output(gradient_##arg2)                                               \
    .add_output(gradient_##arg3)                                               \
    .add_output(gradient_##arg4)                                               \
    .add_output(gradient_##arg5)                                               \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .add_const_input(arg4##_input)                                             \
    .add_const_input(arg5##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1),                                                        \
    iterator.output(2),                                                        \
    iterator.output(3),                                                        \
    iterator.output(4)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> \
name##_backward_backward(                                                      \
  const at::Tensor &gradient_gradient_##arg1##_input,                          \
  const at::Tensor &gradient_gradient_##arg2##_input,                          \
  const at::Tensor &gradient_gradient_##arg3##_input,                          \
  const at::Tensor &gradient_gradient_##arg4##_input,                          \
  const at::Tensor &gradient_gradient_##arg5##_input,                          \
  const at::Tensor &gradient_input,                                            \
  const at::Tensor &arg1##_input,                                              \
  const at::Tensor &arg2##_input,                                              \
  const at::Tensor &arg3##_input,                                              \
  const at::Tensor &arg4##_input,                                              \
  const at::Tensor &arg5##_input                                               \
) {                                                                            \
  at::Tensor gradient_gradient_output;                                         \
  at::Tensor gradient_##arg1;                                                  \
  at::Tensor gradient_##arg2;                                                  \
  at::Tensor gradient_##arg3;                                                  \
  at::Tensor gradient_##arg4;                                                  \
  at::Tensor gradient_##arg5;                                                  \
                                                                               \
  auto iterator = at::TensorIteratorConfig()                                   \
    .add_output(gradient_gradient_output)                                      \
    .add_output(gradient_##arg1)                                               \
    .add_output(gradient_##arg2)                                               \
    .add_output(gradient_##arg3)                                               \
    .add_output(gradient_##arg4)                                               \
    .add_output(gradient_##arg5)                                               \
    .add_const_input(gradient_gradient_##arg1##_input)                         \
    .add_const_input(gradient_gradient_##arg2##_input)                         \
    .add_const_input(gradient_gradient_##arg3##_input)                         \
    .add_const_input(gradient_gradient_##arg4##_input)                         \
    .add_const_input(gradient_gradient_##arg5##_input)                         \
    .add_const_input(gradient_input)                                           \
    .add_const_input(arg1##_input)                                             \
    .add_const_input(arg2##_input)                                             \
    .add_const_input(arg3##_input)                                             \
    .add_const_input(arg4##_input)                                             \
    .add_const_input(arg5##_input)                                             \
    .promote_inputs_to_common_dtype(true)                                      \
    .cast_common_dtype_to_outputs(true)                                        \
    .build();                                                                  \
                                                                               \
  return {                                                                     \
    iterator.output(0),                                                        \
    iterator.output(1),                                                        \
    iterator.output(2),                                                        \
    iterator.output(3),                                                        \
    iterator.output(4),                                                        \
    iterator.output(5)                                                         \
  };                                                                           \
}                                                                              \
                                                                               \
} /* namespace torchscience::meta::special_functions */                        \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                               \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::meta::special_functions::name                                \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward",                                                         \
    torchscience::meta::special_functions::name##_backward                     \
  );                                                                           \
                                                                               \
  module.impl(                                                                 \
    #name "_backward_backward",                                                \
    torchscience::meta::special_functions::name##_backward_backward            \
  );                                                                           \
}
