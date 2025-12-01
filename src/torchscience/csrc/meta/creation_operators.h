#pragma once

#include <vector>
#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta {

namespace {

inline void check_size_nonnegative(const std::vector<int64_t>& shape, const char* op_name) {
    for (auto s : shape) {
        TORCH_CHECK(s >= 0, op_name, ": size must be non-negative, got ", s);
    }
}

}  // anonymous namespace

// MetaCreationTraits only needs output_shape - no kernel dispatch
template<typename CreationTraits>
struct MetaCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        check_size_nonnegative(shape_vec, "meta_creation_op");

        auto options = at::TensorOptions()
            .dtype(dtype.value_or(
                c10::typeMetaToScalarType(at::get_default_dtype())
            ))
            .layout(layout.value_or(at::kStrided))
            .device(at::kMeta)
            .requires_grad(requires_grad);

        return at::empty(shape_vec, options);
    }
};

#define REGISTER_META_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::meta::MetaCreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::meta
