#pragma once

#include <ATen/core/Tensor.h>
#include <torch/extension.h>

namespace torchscience::composite::test_functions {

namespace {

inline void check_rosenbrock_input(const at::Tensor& x, const char* fn_name) {
    TORCH_CHECK(
        at::isFloatingType(x.scalar_type()) || at::isComplexType(x.scalar_type()),
        fn_name, " requires floating-point or complex input, got ",
        x.scalar_type()
    );
    TORCH_CHECK(
        x.dim() >= 1 && x.size(-1) >= 2,
        fn_name, " requires at least 2 dimensions in the last axis, got shape ",
        x.sizes()
    );
}

}  // anonymous namespace

inline at::Tensor rosenbrock(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    check_rosenbrock_input(x, "rosenbrock");

    // x_i and x_{i+1} components along last dimension
    at::Tensor x_i = x.narrow(-1, 0, x.size(-1) - 1);
    at::Tensor x_i_plus_1 = x.narrow(-1, 1, x.size(-1) - 1);

    // Rosenbrock formula: sum of (a - x_i)^2 + b*(x_{i+1} - x_i^2)^2
    at::Tensor term1 = at::pow(a - x_i, 2);
    at::Tensor term2 = b * at::pow(x_i_plus_1 - at::pow(x_i, 2), 2);

    return at::sum(term1 + term2, -1);
}

inline at::Tensor sphere(
    const at::Tensor& x
) {
    TORCH_CHECK(
        at::isFloatingType(x.scalar_type()) || at::isComplexType(x.scalar_type()),
        "sphere requires floating-point or complex input, got ",
        x.scalar_type()
    );
    TORCH_CHECK(
        x.dim() >= 1,
        "sphere requires at least 1 dimension, got ",
        x.dim()
    );
    return at::sum(at::pow(x, 2), -1);
}

inline at::Tensor booth(
    const at::Tensor& x1,
    const at::Tensor& x2
) {
    TORCH_CHECK(
        at::isFloatingType(x1.scalar_type()) || at::isComplexType(x1.scalar_type()),
        "booth requires floating-point or complex input for x1, got ",
        x1.scalar_type()
    );
    TORCH_CHECK(
        at::isFloatingType(x2.scalar_type()) || at::isComplexType(x2.scalar_type()),
        "booth requires floating-point or complex input for x2, got ",
        x2.scalar_type()
    );
    return at::pow(x1 + 2 * x2 - 7, 2) + at::pow(2 * x1 + x2 - 5, 2);
}

inline at::Tensor beale(
    const at::Tensor& x1,
    const at::Tensor& x2
) {
    TORCH_CHECK(
        at::isFloatingType(x1.scalar_type()) || at::isComplexType(x1.scalar_type()),
        "beale requires floating-point or complex input for x1, got ",
        x1.scalar_type()
    );
    TORCH_CHECK(
        at::isFloatingType(x2.scalar_type()) || at::isComplexType(x2.scalar_type()),
        "beale requires floating-point or complex input for x2, got ",
        x2.scalar_type()
    );
    at::Tensor term1 = at::pow(1.5 - x1 + x1 * x2, 2);
    at::Tensor term2 = at::pow(2.25 - x1 + x1 * at::pow(x2, 2), 2);
    at::Tensor term3 = at::pow(2.625 - x1 + x1 * at::pow(x2, 3), 2);
    return term1 + term2 + term3;
}

inline at::Tensor himmelblau(
    const at::Tensor& x1,
    const at::Tensor& x2
) {
    TORCH_CHECK(
        at::isFloatingType(x1.scalar_type()) || at::isComplexType(x1.scalar_type()),
        "himmelblau requires floating-point or complex input for x1, got ",
        x1.scalar_type()
    );
    TORCH_CHECK(
        at::isFloatingType(x2.scalar_type()) || at::isComplexType(x2.scalar_type()),
        "himmelblau requires floating-point or complex input for x2, got ",
        x2.scalar_type()
    );
    return at::pow(at::pow(x1, 2) + x2 - 11, 2) + at::pow(x1 + at::pow(x2, 2) - 7, 2);
}

inline at::Tensor rastrigin(
    const at::Tensor& x
) {
    TORCH_CHECK(
        at::isFloatingType(x.scalar_type()) || at::isComplexType(x.scalar_type()),
        "rastrigin requires floating-point or complex input, got ",
        x.scalar_type()
    );
    TORCH_CHECK(
        x.dim() >= 1,
        "rastrigin requires at least 1 dimension, got ",
        x.dim()
    );
    auto n = static_cast<double>(x.size(-1));
    return 10.0 * n + at::sum(at::pow(x, 2) - 10.0 * at::cos(2.0 * M_PI * x), -1);
}

inline at::Tensor ackley(
    const at::Tensor& x
) {
    TORCH_CHECK(
        at::isFloatingType(x.scalar_type()) || at::isComplexType(x.scalar_type()),
        "ackley requires floating-point or complex input, got ",
        x.scalar_type()
    );
    TORCH_CHECK(
        x.dim() >= 1,
        "ackley requires at least 1 dimension, got ",
        x.dim()
    );
    constexpr double a = 20.0;
    constexpr double b = 0.2;
    constexpr double c = 2.0 * M_PI;
    at::Tensor mean_sq = at::mean(at::pow(x, 2), -1);
    at::Tensor mean_cos = at::mean(at::cos(c * x), -1);
    return -a * at::exp(-b * at::sqrt(mean_sq)) - at::exp(mean_cos) + a + M_E;
}

}  // namespace torchscience::composite::test_functions

TORCH_LIBRARY_IMPL(torchscience, CompositeImplicitAutograd, module) {
    module.impl(
        "rosenbrock",
        &torchscience::composite::test_functions::rosenbrock
    );
    module.impl(
        "sphere",
        &torchscience::composite::test_functions::sphere
    );
    module.impl(
        "booth",
        &torchscience::composite::test_functions::booth
    );
    module.impl(
        "beale",
        &torchscience::composite::test_functions::beale
    );
    module.impl(
        "himmelblau",
        &torchscience::composite::test_functions::himmelblau
    );
    module.impl(
        "rastrigin",
        &torchscience::composite::test_functions::rastrigin
    );
    module.impl(
        "ackley",
        &torchscience::composite::test_functions::ackley
    );
}
