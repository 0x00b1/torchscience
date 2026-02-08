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
    """Check R_G(ax, ay, az) = a^(1/2) R_G(x, y, z)."""
    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([2.0, 3.0], dtype=torch.float64)
    z = torch.tensor([3.0, 4.0], dtype=torch.float64)
    a = 4.0
    left = func(a * x, a * y, a * z)
    right = func(x, y, z) * math.sqrt(a)
    return left, right


def _symmetry(func):
    """Check R_G is symmetric in all arguments."""
    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([2.0, 3.0], dtype=torch.float64)
    z = torch.tensor([3.0, 4.0], dtype=torch.float64)
    left = func(x, y, z)
    right = func(y, z, x)
    return left, right


class TestCarlsonEllipticIntegralRG(OpTestCase):
    """Tests for Carlson's elliptic integral R_G."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="carlson_elliptic_integral_r_g",
            func=torchscience.special_functions.carlson_elliptic_integral_r_g,
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
                    expected=1.0,
                    description="R_G(1,1,1) = sqrt(1) = 1",
                ),
                SpecialValue(
                    inputs=(4.0, 4.0, 4.0),
                    expected=2.0,
                    description="R_G(4,4,4) = sqrt(4) = 2",
                ),
            ],
            functional_identities=[
                IdentitySpec(
                    name="homogeneity",
                    identity_fn=_homogeneity,
                    description="R_G(ax, ay, az) = sqrt(a) R_G(x, y, z)",
                ),
                IdentitySpec(
                    name="symmetry",
                    identity_fn=_symmetry,
                    description="R_G is symmetric in all arguments",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_forward_against_mpmath(self):
        """Compare against mpmath.elliprg."""
        test_cases = [
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (1.0, 2.0, 3.0),
            (0.5, 1.5, 2.5),
        ]
        for x_val, y_val, z_val in test_cases:
            x = torch.tensor([x_val], dtype=torch.float64)
            y = torch.tensor([y_val], dtype=torch.float64)
            z = torch.tensor([z_val], dtype=torch.float64)
            result = (
                torchscience.special_functions.carlson_elliptic_integral_r_g(
                    x, y, z
                )
            )
            expected = float(mpmath.elliprg(x_val, y_val, z_val))
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_equal_args(self):
        """R_G(x, x, x) = sqrt(x)."""
        x = torch.tensor([1.0, 2.0, 4.0, 9.0], dtype=torch.float64)
        result = torchscience.special_functions.carlson_elliptic_integral_r_g(
            x, x, x
        )
        expected = torch.sqrt(x)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)
