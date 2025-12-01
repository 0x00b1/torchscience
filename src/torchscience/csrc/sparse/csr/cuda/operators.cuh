#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::sparse::csr::cuda {

// ============================================================================
// Sparse CSR CUDA operators delegate to dense operations on values
// ============================================================================

namespace detail {

inline at::Tensor make_sparse_csr(
    const at::Tensor& crow_indices,
    const at::Tensor& col_indices,
    const at::Tensor& values,
    at::IntArrayRef sizes,
    const at::TensorOptions& options
) {
    return at::sparse_csr_tensor(
        crow_indices, col_indices, values, sizes, options.dtype(values.scalar_type())
    );
}

}  // namespace detail

// ============================================================================
// SparseCsrCudaUnaryOperator
// ============================================================================

struct SparseCsrCudaUnaryOperator {
    static at::Tensor forward(
        const at::Tensor& input,
        const char* schema_name
    ) {
        TORCH_CHECK(input.is_sparse_csr(), "expects sparse CSR tensor");

        at::Tensor values = input.values();

        at::Tensor new_values = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<at::Tensor(const at::Tensor&)>()
            .call(values);

        return detail::make_sparse_csr(
            input.crow_indices(), input.col_indices(), new_values,
            input.sizes(), input.options()
        );
    }

    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        const char* schema_name
    ) {
        TORCH_CHECK(grad_output.is_sparse_csr(), "expects sparse CSR gradient");
        TORCH_CHECK(input.is_sparse_csr(), "expects sparse CSR input");

        at::Tensor new_grad_values = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
            .call(grad_output.values(), input.values());

        return detail::make_sparse_csr(
            input.crow_indices(), input.col_indices(), new_grad_values,
            input.sizes(), input.options()
        );
    }

    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        const char* schema_name
    ) {
        if (!grad_grad_input.defined()) {
            return {};
        }

        TORCH_CHECK(grad_grad_input.is_sparse_csr(), "expects sparse CSR grad_grad");
        TORCH_CHECK(grad_output.is_sparse_csr(), "expects sparse CSR gradient");
        TORCH_CHECK(input.is_sparse_csr(), "expects sparse CSR input");

        auto [new_gg_out_values, new_grad_values] = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_grad_input.values(), grad_output.values(), input.values());

        at::Tensor grad_grad_output;
        at::Tensor grad_input;

        if (new_gg_out_values.defined()) {
            grad_grad_output = detail::make_sparse_csr(
                grad_output.crow_indices(), grad_output.col_indices(),
                new_gg_out_values, grad_output.sizes(), grad_output.options()
            );
        }

        if (new_grad_values.defined()) {
            grad_input = detail::make_sparse_csr(
                input.crow_indices(), input.col_indices(),
                new_grad_values, input.sizes(), input.options()
            );
        }

        return {grad_grad_output, grad_input};
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* backward_backward_name,
        const char* schema_name,
        const char* schema_backward_name,
        const char* schema_backward_backward_name
    ) {
        module.impl(name, [schema_name](const at::Tensor& input) {
            return forward(input, schema_name);
        });
        module.impl(backward_name, [schema_backward_name](
            const at::Tensor& grad_output, const at::Tensor& input
        ) {
            return backward(grad_output, input, schema_backward_name);
        });
        module.impl(backward_backward_name, [schema_backward_backward_name](
            const at::Tensor& grad_grad_input,
            const at::Tensor& grad_output,
            const at::Tensor& input
        ) {
            return backward_backward(
                grad_grad_input, grad_output, input, schema_backward_backward_name
            );
        });
    }
};

#define REGISTER_SPARSE_CSR_CUDA_UNARY(module, name) \
    ::torchscience::sparse::csr::cuda::SparseCsrCudaUnaryOperator::register_all( \
        module, #name, #name "_backward", #name "_backward_backward", \
        "torchscience::" #name, \
        "torchscience::" #name "_backward", \
        "torchscience::" #name "_backward_backward")

// ============================================================================
// SparseCsrCudaBinaryOperator
// ============================================================================

struct SparseCsrCudaBinaryOperator {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const char* schema_name
    ) {
        TORCH_CHECK(input1.is_sparse_csr() && input2.is_sparse_csr(),
                    "expects sparse CSR tensors");

        at::Tensor new_values = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
            .call(input1.values(), input2.values());

        return detail::make_sparse_csr(
            input1.crow_indices(), input1.col_indices(), new_values,
            input1.sizes(), input1.options()
        );
    }

    static std::tuple<at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const char* schema_name
    ) {
        TORCH_CHECK(grad_output.is_sparse_csr(), "expects sparse CSR gradient");
        TORCH_CHECK(input1.is_sparse_csr() && input2.is_sparse_csr(),
                    "expects sparse CSR inputs");

        auto [new_grad1, new_grad2] = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_output.values(), input1.values(), input2.values());

        at::Tensor grad_input1, grad_input2;

        if (new_grad1.defined()) {
            grad_input1 = detail::make_sparse_csr(
                input1.crow_indices(), input1.col_indices(), new_grad1,
                input1.sizes(), input1.options()
            );
        }

        if (new_grad2.defined()) {
            grad_input2 = detail::make_sparse_csr(
                input2.crow_indices(), input2.col_indices(), new_grad2,
                input2.sizes(), input2.options()
            );
        }

        return {grad_input1, grad_input2};
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& gg_input1,
        const at::Tensor& gg_input2,
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const char* schema_name
    ) {
        at::Tensor gg1_values = gg_input1.defined() && gg_input1.is_sparse_csr()
            ? gg_input1.values() : gg_input1;
        at::Tensor gg2_values = gg_input2.defined() && gg_input2.is_sparse_csr()
            ? gg_input2.values() : gg_input2;

        auto [new_gg_out, new_grad1, new_grad2] = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&
            )>()
            .call(gg1_values, gg2_values, grad_output.values(),
                  input1.values(), input2.values());

        at::Tensor grad_grad_output, grad_input1, grad_input2;

        if (new_gg_out.defined()) {
            grad_grad_output = detail::make_sparse_csr(
                grad_output.crow_indices(), grad_output.col_indices(), new_gg_out,
                grad_output.sizes(), grad_output.options()
            );
        }

        if (new_grad1.defined()) {
            grad_input1 = detail::make_sparse_csr(
                input1.crow_indices(), input1.col_indices(), new_grad1,
                input1.sizes(), input1.options()
            );
        }

        if (new_grad2.defined()) {
            grad_input2 = detail::make_sparse_csr(
                input2.crow_indices(), input2.col_indices(), new_grad2,
                input2.sizes(), input2.options()
            );
        }

        return {grad_grad_output, grad_input1, grad_input2};
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* backward_backward_name,
        const char* schema_name,
        const char* schema_backward_name,
        const char* schema_backward_backward_name
    ) {
        module.impl(name, [schema_name](
            const at::Tensor& input1, const at::Tensor& input2
        ) {
            return forward(input1, input2, schema_name);
        });
        module.impl(backward_name, [schema_backward_name](
            const at::Tensor& grad_output,
            const at::Tensor& input1,
            const at::Tensor& input2
        ) {
            return backward(grad_output, input1, input2, schema_backward_name);
        });
        module.impl(backward_backward_name, [schema_backward_backward_name](
            const at::Tensor& gg_input1,
            const at::Tensor& gg_input2,
            const at::Tensor& grad_output,
            const at::Tensor& input1,
            const at::Tensor& input2
        ) {
            return backward_backward(
                gg_input1, gg_input2, grad_output, input1, input2,
                schema_backward_backward_name
            );
        });
    }
};

#define REGISTER_SPARSE_CSR_CUDA_BINARY(module, name) \
    ::torchscience::sparse::csr::cuda::SparseCsrCudaBinaryOperator::register_all( \
        module, #name, #name "_backward", #name "_backward_backward", \
        "torchscience::" #name, \
        "torchscience::" #name "_backward", \
        "torchscience::" #name "_backward_backward")

// ============================================================================
// SparseCsrCudaTernaryOperator
// ============================================================================

struct SparseCsrCudaTernaryOperator {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const char* schema_name
    ) {
        TORCH_CHECK(
            input1.is_sparse_csr() && input2.is_sparse_csr() && input3.is_sparse_csr(),
            "expects sparse CSR tensors"
        );

        at::Tensor new_values = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<at::Tensor(
                const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(input1.values(), input2.values(), input3.values());

        return detail::make_sparse_csr(
            input1.crow_indices(), input1.col_indices(), new_values,
            input1.sizes(), input1.options()
        );
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const char* schema_name
    ) {
        TORCH_CHECK(grad_output.is_sparse_csr(), "expects sparse CSR gradient");

        auto [new_grad1, new_grad2, new_grad3] = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_output.values(), input1.values(),
                  input2.values(), input3.values());

        at::Tensor grad1, grad2, grad3;

        if (new_grad1.defined()) {
            grad1 = detail::make_sparse_csr(
                input1.crow_indices(), input1.col_indices(), new_grad1,
                input1.sizes(), input1.options()
            );
        }
        if (new_grad2.defined()) {
            grad2 = detail::make_sparse_csr(
                input2.crow_indices(), input2.col_indices(), new_grad2,
                input2.sizes(), input2.options()
            );
        }
        if (new_grad3.defined()) {
            grad3 = detail::make_sparse_csr(
                input3.crow_indices(), input3.col_indices(), new_grad3,
                input3.sizes(), input3.options()
            );
        }

        return {grad1, grad2, grad3};
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* schema_name,
        const char* schema_backward_name
    ) {
        module.impl(name, [schema_name](
            const at::Tensor& input1,
            const at::Tensor& input2,
            const at::Tensor& input3
        ) {
            return forward(input1, input2, input3, schema_name);
        });
        module.impl(backward_name, [schema_backward_name](
            const at::Tensor& grad_output,
            const at::Tensor& input1,
            const at::Tensor& input2,
            const at::Tensor& input3
        ) {
            return backward(grad_output, input1, input2, input3, schema_backward_name);
        });
    }
};

#define REGISTER_SPARSE_CSR_CUDA_TERNARY(module, name) \
    ::torchscience::sparse::csr::cuda::SparseCsrCudaTernaryOperator::register_all( \
        module, #name, #name "_backward", \
        "torchscience::" #name, \
        "torchscience::" #name "_backward")

// ============================================================================
// SparseCsrCudaQuaternaryOperator
// ============================================================================

struct SparseCsrCudaQuaternaryOperator {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const at::Tensor& input4,
        const char* schema_name
    ) {
        TORCH_CHECK(
            input1.is_sparse_csr() && input2.is_sparse_csr() &&
            input3.is_sparse_csr() && input4.is_sparse_csr(),
            "expects sparse CSR tensors"
        );

        at::Tensor new_values = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<at::Tensor(
                const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&
            )>()
            .call(input1.values(), input2.values(),
                  input3.values(), input4.values());

        return detail::make_sparse_csr(
            input1.crow_indices(), input1.col_indices(), new_values,
            input1.sizes(), input1.options()
        );
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const at::Tensor& input4,
        const char* schema_name
    ) {
        TORCH_CHECK(grad_output.is_sparse_csr(), "expects sparse CSR gradient");

        auto [new_grad1, new_grad2, new_grad3, new_grad4] =
            c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_output.values(), input1.values(), input2.values(),
                  input3.values(), input4.values());

        at::Tensor grad1, grad2, grad3, grad4;

        if (new_grad1.defined()) {
            grad1 = detail::make_sparse_csr(
                input1.crow_indices(), input1.col_indices(), new_grad1,
                input1.sizes(), input1.options()
            );
        }
        if (new_grad2.defined()) {
            grad2 = detail::make_sparse_csr(
                input2.crow_indices(), input2.col_indices(), new_grad2,
                input2.sizes(), input2.options()
            );
        }
        if (new_grad3.defined()) {
            grad3 = detail::make_sparse_csr(
                input3.crow_indices(), input3.col_indices(), new_grad3,
                input3.sizes(), input3.options()
            );
        }
        if (new_grad4.defined()) {
            grad4 = detail::make_sparse_csr(
                input4.crow_indices(), input4.col_indices(), new_grad4,
                input4.sizes(), input4.options()
            );
        }

        return {grad1, grad2, grad3, grad4};
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* schema_name,
        const char* schema_backward_name
    ) {
        module.impl(name, [schema_name](
            const at::Tensor& input1,
            const at::Tensor& input2,
            const at::Tensor& input3,
            const at::Tensor& input4
        ) {
            return forward(input1, input2, input3, input4, schema_name);
        });
        module.impl(backward_name, [schema_backward_name](
            const at::Tensor& grad_output,
            const at::Tensor& input1,
            const at::Tensor& input2,
            const at::Tensor& input3,
            const at::Tensor& input4
        ) {
            return backward(
                grad_output, input1, input2, input3, input4, schema_backward_name
            );
        });
    }
};

#define REGISTER_SPARSE_CSR_CUDA_QUATERNARY(module, name) \
    ::torchscience::sparse::csr::cuda::SparseCsrCudaQuaternaryOperator::register_all( \
        module, #name, #name "_backward", \
        "torchscience::" #name, \
        "torchscience::" #name "_backward")

}  // namespace torchscience::sparse::csr::cuda
