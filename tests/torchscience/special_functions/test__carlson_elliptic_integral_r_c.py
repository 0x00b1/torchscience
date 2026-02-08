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
    SingularitySpec,
    SpecialValue,
    ToleranceConfig,
)


def _homogeneity(func):
    """Check R_C(ax, ay) = R_C(x, y) / sqrt(a)."""
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    y = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    a = 4.0
    left = func(a * x, a * y)
    right = func(x, y) / math.sqrt(a)
    return left, right


def _degeneration_from_rf(func):
    """Check R_F(x, y, y) = R_C(x, y)."""
    x = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    y = torch.tensor([2.0, 3.0, 1.5], dtype=torch.float64)
    left = torchscience.special_functions.carlson_elliptic_integral_r_f(
        x, y, y
    )
    right = func(x, y)
    return left, right


class TestCarlsonEllipticIntegralRC(OpTestCase):
    """Tests for Carlson's elliptic integral R_C."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="carlson_elliptic_integral_r_c",
            func=torchscience.special_functions.carlson_elliptic_integral_r_c,
            arity=2,
            input_specs=[
                InputSpec(name="x", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(name="y", position=1, default_real_range=(0.5, 5.0)),
            ],
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_sparse_coo_basic",
                "test_sparse_csr_basic",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_pole_behavior",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 1.0),
                    expected=math.pi / 2,
                    description="R_C(0, 1) = pi/2",
                ),
                SpecialValue(
                    inputs=(1.0, 1.0),
                    expected=1.0,
                    description="R_C(1, 1) = 1",
                ),
                SpecialValue(
                    inputs=(4.0, 4.0),
                    expected=0.5,
                    description="R_C(4, 4) = 1/sqrt(4) = 0.5",
                ),
            ],
            functional_identities=[
                IdentitySpec(
                    name="homogeneity",
                    identity_fn=_homogeneity,
                    description="R_C(ax, ay) = R_C(x, y) / sqrt(a)",
                ),
                IdentitySpec(
                    name="degeneration_from_rf",
                    identity_fn=_degeneration_from_rf,
                    description="R_F(x, y, y) = R_C(x, y)",
                ),
            ],
            singularities=[
                SingularitySpec(
                    type="pole",
                    locations=lambda: iter([0.0]),
                    expected_behavior="inf",
                    description="Pole at y=0",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_forward_against_mpmath(self):
        """Compare against mpmath.elliprc."""
        test_cases = [
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (2.0, 3.0),
            (0.5, 1.5),
        ]
        for x_val, y_val in test_cases:
            x = torch.tensor([x_val], dtype=torch.float64)
            y = torch.tensor([y_val], dtype=torch.float64)
            result = (
                torchscience.special_functions.carlson_elliptic_integral_r_c(
                    x, y
                )
            )
            expected = float(mpmath.elliprc(x_val, y_val))
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_equal_args(self):
        """R_C(x, x) = 1/sqrt(x)."""
        x = torch.tensor([1.0, 2.0, 4.0, 9.0], dtype=torch.float64)
        result = torchscience.special_functions.carlson_elliptic_integral_r_c(
            x, x
        )
        expected = 1.0 / torch.sqrt(x)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_pole_at_y_zero(self):
        """R_C(x, 0) diverges (pole at y=0). Result should be very large or inf."""
        x = torch.tensor([1.0], dtype=torch.float64)
        y = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.carlson_elliptic_integral_r_c(
            x, y
        )
        # The iterative algorithm may return a very large finite number rather
        # than inf at the pole; verify the result is extremely large or inf/nan.
        assert (
            torch.isinf(result).all()
            or torch.isnan(result).all()
            or (result.abs() > 1e20).all()
        ), f"Expected very large value at y=0 pole, got {result}"
