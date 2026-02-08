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
    """Check R_D(ax, ay, az) = a^(-3/2) R_D(x, y, z)."""
    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([2.0, 3.0], dtype=torch.float64)
    z = torch.tensor([3.0, 4.0], dtype=torch.float64)
    a = 4.0
    left = func(a * x, a * y, a * z)
    right = func(x, y, z) / (a**1.5)
    return left, right


def _symmetry_xy(func):
    """Check R_D(x, y, z) = R_D(y, x, z)."""
    x = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    y = torch.tensor([2.0, 3.0, 1.5], dtype=torch.float64)
    z = torch.tensor([3.0, 4.0, 2.5], dtype=torch.float64)
    left = func(x, y, z)
    right = func(y, x, z)
    return left, right


class TestCarlsonEllipticIntegralRD(OpTestCase):
    """Tests for Carlson's elliptic integral R_D."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="carlson_elliptic_integral_r_d",
            func=torchscience.special_functions.carlson_elliptic_integral_r_d,
            arity=3,
            input_specs=[
                InputSpec(name="x", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(name="y", position=1, default_real_range=(0.5, 5.0)),
                InputSpec(name="z", position=2, default_real_range=(0.5, 5.0)),
            ],
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradcheck_real",
                "test_gradcheck_complex",
                "test_gradgradcheck_real",
                "test_gradgradcheck_complex",
            },
            special_values=[
                SpecialValue(
                    inputs=(1.0, 1.0, 1.0),
                    expected=1.0,
                    description="R_D(1,1,1) = 1^(-3/2) = 1",
                ),
                SpecialValue(
                    inputs=(4.0, 4.0, 4.0),
                    expected=0.125,
                    description="R_D(4,4,4) = 4^(-3/2) = 0.125",
                ),
            ],
            functional_identities=[
                IdentitySpec(
                    name="homogeneity",
                    identity_fn=_homogeneity,
                    description="R_D(ax, ay, az) = a^(-3/2) R_D(x, y, z)",
                ),
                IdentitySpec(
                    name="symmetry_xy",
                    identity_fn=_symmetry_xy,
                    description="R_D(x, y, z) = R_D(y, x, z)",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_forward_against_mpmath(self):
        """Compare against mpmath.elliprd."""
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
                torchscience.special_functions.carlson_elliptic_integral_r_d(
                    x, y, z
                )
            )
            expected = float(mpmath.elliprd(x_val, y_val, z_val))
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_degeneration_from_rj(self):
        """R_J(x, y, z, z) = R_D(x, y, z)."""
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        y = torch.tensor([2.0, 3.0], dtype=torch.float64)
        z = torch.tensor([3.0, 4.0], dtype=torch.float64)
        rd = torchscience.special_functions.carlson_elliptic_integral_r_d(
            x, y, z
        )
        rj = torchscience.special_functions.carlson_elliptic_integral_r_j(
            x, y, z, z
        )
        torch.testing.assert_close(rd, rj, rtol=1e-8, atol=1e-10)
