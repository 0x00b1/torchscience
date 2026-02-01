#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

#define TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(name, arg1)                       \
namespace torchscience::autocast::special_functions {                          \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &(arg1)                                                     \
) {                                                                            \
  c10::impl::ExcludeDispatchKeyGuard no_autocast(                              \
    c10::DispatchKey::Autocast                                                 \
  );                                                                           \
                                                                               \
  return c10::Dispatcher::singleton()                                          \
    .findSchemaOrThrow(                                                        \
      "torchscience::" #name,                                                  \
      ""                                                                       \
    ).typed<at::Tensor(                                                        \
      const at::Tensor &                                                       \
    )>().call(                                                                 \
      at::autocast::cached_cast(                                               \
        at::kFloat,                                                            \
        arg1                                                                   \
      )                                                                        \
    );                                                                         \
}                                                                              \
                                                                               \
} /* namespace torchscience::autocast::special_functions */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {                           \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autocast::special_functions::name                            \
  );                                                                           \
}

#define TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(name, arg1, arg2)                \
namespace torchscience::autocast::special_functions {                          \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &(arg1),                                                    \
  const at::Tensor &(arg2)                                                     \
) {                                                                            \
  c10::impl::ExcludeDispatchKeyGuard no_autocast(                              \
    c10::DispatchKey::Autocast                                                 \
  );                                                                           \
                                                                               \
  auto device_type = arg1.device().type();                                      \
  auto target_type = at::autocast::promote_type(                               \
    at::kFloat,                                                                \
    device_type,                                                               \
    arg1,                                                                      \
    arg2                                                                       \
  );                                                                           \
                                                                               \
  return c10::Dispatcher::singleton()                                          \
    .findSchemaOrThrow(                                                        \
      "torchscience::" #name,                                                  \
      ""                                                                       \
    ).typed<at::Tensor(                                                        \
      const at::Tensor &,                                                      \
      const at::Tensor &                                                       \
    )>().call(                                                                 \
      at::autocast::cached_cast(target_type, arg1),                            \
      at::autocast::cached_cast(target_type, arg2)                             \
    );                                                                         \
}                                                                              \
                                                                               \
} /* namespace torchscience::autocast::special_functions */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {                           \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autocast::special_functions::name                            \
  );                                                                           \
}

#define TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(name, arg1, arg2, arg3)         \
namespace torchscience::autocast::special_functions {                          \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &(arg1),                                                    \
  const at::Tensor &(arg2),                                                    \
  const at::Tensor &(arg3)                                                     \
) {                                                                            \
  c10::impl::ExcludeDispatchKeyGuard no_autocast(                              \
    c10::DispatchKey::Autocast                                                 \
  );                                                                           \
                                                                               \
  auto device_type = arg1.device().type();                                      \
  auto target_type = at::autocast::promote_type(                               \
    at::kFloat,                                                                \
    device_type,                                                               \
    arg1,                                                                      \
    arg2,                                                                      \
    arg3                                                                       \
  );                                                                           \
                                                                               \
  return c10::Dispatcher::singleton()                                          \
    .findSchemaOrThrow(                                                        \
      "torchscience::" #name,                                                  \
      ""                                                                       \
    ).typed<at::Tensor(                                                        \
      const at::Tensor &,                                                      \
      const at::Tensor &,                                                      \
      const at::Tensor &                                                       \
    )>().call(                                                                 \
      at::autocast::cached_cast(target_type, arg1),                            \
      at::autocast::cached_cast(target_type, arg2),                            \
      at::autocast::cached_cast(target_type, arg3)                             \
    );                                                                         \
}                                                                              \
                                                                               \
} /* namespace torchscience::autocast::special_functions */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {                           \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autocast::special_functions::name                            \
  );                                                                           \
}

#define TORCHSCIENCE_AUTOCAST_POINTWISE_QUATERNARY_OPERATOR(name, arg1, arg2, arg3, arg4) \
namespace torchscience::autocast::special_functions {                          \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &(arg1),                                                    \
  const at::Tensor &(arg2),                                                    \
  const at::Tensor &(arg3),                                                    \
  const at::Tensor &(arg4)                                                     \
) {                                                                            \
  c10::impl::ExcludeDispatchKeyGuard no_autocast(                              \
    c10::DispatchKey::Autocast                                                 \
  );                                                                           \
                                                                               \
  auto device_type = arg1.device().type();                                      \
  auto target_type = at::autocast::promote_type(                               \
    at::kFloat,                                                                \
    device_type,                                                               \
    arg1,                                                                      \
    arg2,                                                                      \
    arg3,                                                                      \
    arg4                                                                       \
  );                                                                           \
                                                                               \
  return c10::Dispatcher::singleton()                                          \
    .findSchemaOrThrow(                                                        \
      "torchscience::" #name,                                                  \
      ""                                                                       \
    ).typed<at::Tensor(                                                        \
      const at::Tensor &,                                                      \
      const at::Tensor &,                                                      \
      const at::Tensor &,                                                      \
      const at::Tensor &                                                       \
    )>().call(                                                                 \
      at::autocast::cached_cast(target_type, arg1),                            \
      at::autocast::cached_cast(target_type, arg2),                            \
      at::autocast::cached_cast(target_type, arg3),                            \
      at::autocast::cached_cast(target_type, arg4)                             \
    );                                                                         \
}                                                                              \
                                                                               \
} /* namespace torchscience::autocast::special_functions */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {                           \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autocast::special_functions::name                            \
  );                                                                           \
}

#define TORCHSCIENCE_AUTOCAST_POINTWISE_QUINARY_OPERATOR(name, arg1, arg2, arg3, arg4, arg5) \
namespace torchscience::autocast::special_functions {                          \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &(arg1),                                                    \
  const at::Tensor &(arg2),                                                    \
  const at::Tensor &(arg3),                                                    \
  const at::Tensor &(arg4),                                                    \
  const at::Tensor &(arg5)                                                     \
) {                                                                            \
  c10::impl::ExcludeDispatchKeyGuard no_autocast(                              \
    c10::DispatchKey::Autocast                                                 \
  );                                                                           \
                                                                               \
  auto device_type = arg1.device().type();                                      \
  auto target_type = at::autocast::promote_type(                               \
    at::kFloat,                                                                \
    device_type,                                                               \
    arg1,                                                                      \
    arg2,                                                                      \
    arg3,                                                                      \
    arg4,                                                                      \
    arg5                                                                       \
  );                                                                           \
                                                                               \
  return c10::Dispatcher::singleton()                                          \
    .findSchemaOrThrow(                                                        \
      "torchscience::" #name,                                                  \
      ""                                                                       \
    ).typed<at::Tensor(                                                        \
      const at::Tensor &,                                                      \
      const at::Tensor &,                                                      \
      const at::Tensor &,                                                      \
      const at::Tensor &,                                                      \
      const at::Tensor &                                                       \
    )>().call(                                                                 \
      at::autocast::cached_cast(target_type, arg1),                            \
      at::autocast::cached_cast(target_type, arg2),                            \
      at::autocast::cached_cast(target_type, arg3),                            \
      at::autocast::cached_cast(target_type, arg4),                            \
      at::autocast::cached_cast(target_type, arg5)                             \
    );                                                                         \
}                                                                              \
                                                                               \
} /* namespace torchscience::autocast::special_functions */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {                           \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autocast::special_functions::name                            \
  );                                                                           \
}
