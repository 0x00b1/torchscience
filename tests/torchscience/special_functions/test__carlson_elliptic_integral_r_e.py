import torch
import torch.testing

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
    ToleranceConfig,
)

# Optional mpmath import for reference tests
try:
    import mpmath

    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


def mpmath_re(x, y, z):
    """Reference implementation using mpmath via R_D relationship.

    R_E(x, y, z) = (3/2) * z * R_D(x, y, z) + sqrt(xy/z)
    """
    rd = mpmath.elliprd(x, y, z)
    return float(1.5 * z * rd + mpmath.sqrt(x * y / z))


class TestCarlsonEllipticIntegralRE(OpTestCase):
    """Tests for Carlson's elliptic integral R_E."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="carlson_elliptic_integral_r_e",
            func=torchscience.special_functions.carlson_elliptic_integral_r_e,
            arity=3,
            input_specs=[
                InputSpec(name="x", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(name="y", position=1, default_real_range=(0.5, 5.0)),
                InputSpec(name="z", position=2, default_real_range=(0.5, 5.0)),
            ],
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradgradcheck_real",
                "test_gradgradcheck_complex",
            },
            special_values=[
                SpecialValue(
                    inputs=(1.0, 1.0, 1.0),
                    expected=2.5,
                    description="R_E(1,1,1) = 2.5",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_equal_args(self):
        """R_E(x, x, x) = (3/2)*x*R_D(x,x,x) + sqrt(x*x/x) = 3/2*x^{-1/2} + x^{1/2}."""
        x = torch.tensor([1.0, 2.0, 4.0, 9.0], dtype=torch.float64)
        result = torchscience.special_functions.carlson_elliptic_integral_r_e(
            x, x, x
        )
        # R_D(x,x,x) = x^{-3/2}, so R_E(x,x,x) = 3/2 * x^{-1/2} + x^{1/2}
        expected = 1.5 * x ** (-0.5) + x**0.5
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_symmetry_xy(self):
        """R_E is symmetric in x and y: R_E(x, y, z) = R_E(y, x, z)."""
        x = torch.tensor([1.0], dtype=torch.float64)
        y = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([3.0], dtype=torch.float64)

        re_xyz = torchscience.special_functions.carlson_elliptic_integral_r_e(
            x, y, z
        )
        re_yxz = torchscience.special_functions.carlson_elliptic_integral_r_e(
            y, x, z
        )
        torch.testing.assert_close(re_xyz, re_yxz, rtol=1e-10, atol=1e-10)

    def test_relationship_to_rd(self):
        """R_E(x, y, z) = (3/2)*z*R_D(x, y, z) + sqrt(xy/z)."""
        x = torch.tensor([1.0], dtype=torch.float64)
        y = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([3.0], dtype=torch.float64)

        re = torchscience.special_functions.carlson_elliptic_integral_r_e(
            x, y, z
        )
        rd = torchscience.special_functions.carlson_elliptic_integral_r_d(
            x, y, z
        )
        re_from_rd = 1.5 * z * rd + torch.sqrt(x * y / z)
        torch.testing.assert_close(re, re_from_rd, rtol=1e-10, atol=1e-10)

    def test_forward_against_mpmath(self):
        """Compare against mpmath reference."""
        if not HAS_MPMATH:
            return
        test_cases = [
            (1.0, 1.0, 1.0),
            (1.0, 2.0, 3.0),
            (0.5, 1.5, 2.5),
            (2.0, 3.0, 4.0),
            (0.1, 0.5, 1.0),
        ]
        for x_val, y_val, z_val in test_cases:
            x = torch.tensor([x_val], dtype=torch.float64)
            y = torch.tensor([y_val], dtype=torch.float64)
            z = torch.tensor([z_val], dtype=torch.float64)
            result = (
                torchscience.special_functions.carlson_elliptic_integral_r_e(
                    x, y, z
                )
            )
            expected = mpmath_re(x_val, y_val, z_val)
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_gradient(self):
        """First-order gradient via gradcheck."""
        x = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(
            torchscience.special_functions.carlson_elliptic_integral_r_e,
            (x, y, z),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )
