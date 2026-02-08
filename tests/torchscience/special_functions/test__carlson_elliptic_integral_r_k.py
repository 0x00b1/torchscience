import math

import mpmath
import torch
import torch.testing

import torchscience.special_functions
from torchscience.testing import (
    IdentitySpec,
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
    ToleranceConfig,
)


def _homogeneity(func):
    """Check R_K(ax, ay) = R_K(x, y) / sqrt(a)."""
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    y = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    a = 4.0
    left = func(a * x, a * y)
    right = func(x, y) / math.sqrt(a)
    return left, right


def _relation_to_rf(func):
    """Check R_K(x, y) = R_F(0, x, y)."""
    x = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    y = torch.tensor([2.0, 3.0, 1.5], dtype=torch.float64)
    left = func(x, y)
    zero = torch.zeros_like(x)
    right = torchscience.special_functions.carlson_elliptic_integral_r_f(
        zero, x, y
    )
    return left, right


class TestCarlsonEllipticIntegralRK(OpTestCase):
    """Tests for Carlson's elliptic integral R_K."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="carlson_elliptic_integral_r_k",
            func=torchscience.special_functions.carlson_elliptic_integral_r_k,
            arity=2,
            input_specs=[
                InputSpec(name="x", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(name="y", position=1, default_real_range=(0.5, 5.0)),
            ],
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradgradcheck_real",
                "test_gradgradcheck_complex",
            },
            special_values=[
                SpecialValue(
                    inputs=(1.0, 1.0),
                    expected=math.pi / 2,
                    description="R_K(1, 1) = pi/2",
                ),
                SpecialValue(
                    inputs=(4.0, 4.0),
                    expected=math.pi / 4,
                    description="R_K(4, 4) = pi/(2*sqrt(4)) = pi/4",
                ),
            ],
            functional_identities=[
                IdentitySpec(
                    name="homogeneity",
                    identity_fn=_homogeneity,
                    description="R_K(ax, ay) = R_K(x, y) / sqrt(a)",
                ),
                IdentitySpec(
                    name="relation_to_rf",
                    identity_fn=_relation_to_rf,
                    description="R_K(x, y) = R_F(0, x, y)",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_forward_against_mpmath(self):
        """Compare against mpmath.elliprf(0, x, y)."""
        test_cases = [
            (1.0, 1.0),
            (1.0, 2.0),
            (2.0, 3.0),
            (0.5, 1.5),
            (4.0, 4.0),
        ]
        for x_val, y_val in test_cases:
            x = torch.tensor([x_val], dtype=torch.float64)
            y = torch.tensor([y_val], dtype=torch.float64)
            result = (
                torchscience.special_functions.carlson_elliptic_integral_r_k(
                    x, y
                )
            )
            expected = float(mpmath.elliprf(0, x_val, y_val))
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_equal_args(self):
        """R_K(a, a) = pi/(2*sqrt(a))."""
        x = torch.tensor([1.0, 2.0, 4.0, 9.0], dtype=torch.float64)
        result = torchscience.special_functions.carlson_elliptic_integral_r_k(
            x, x
        )
        expected = math.pi / (2.0 * torch.sqrt(x))
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_symmetry(self):
        """R_K(x, y) = R_K(y, x) (symmetric in both arguments)."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        y = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        rk_xy = torchscience.special_functions.carlson_elliptic_integral_r_k(
            x, y
        )
        rk_yx = torchscience.special_functions.carlson_elliptic_integral_r_k(
            y, x
        )
        torch.testing.assert_close(rk_xy, rk_yx, rtol=1e-10, atol=1e-10)
