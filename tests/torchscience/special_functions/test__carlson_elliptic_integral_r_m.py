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


def _symmetry_xy(func):
    """Check R_M(x, y, z) = R_M(y, x, z) (symmetric in first two args)."""
    x = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    y = torch.tensor([2.0, 3.0, 1.5], dtype=torch.float64)
    z = torch.tensor([3.0, 4.0, 2.5], dtype=torch.float64)
    left = func(x, y, z)
    right = func(y, x, z)
    return left, right


class TestCarlsonEllipticIntegralRM(OpTestCase):
    """Tests for Carlson's elliptic integral R_M."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="carlson_elliptic_integral_r_m",
            func=torchscience.special_functions.carlson_elliptic_integral_r_m,
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
                "test_low_precision_forward",
            },
            special_values=[
                SpecialValue(
                    inputs=(1.0, 1.0, 1.0),
                    expected=2.0 / 3.0,
                    description="R_M(1,1,1) = (1/sqrt(1) + 3*1) / 6 = 2/3",
                ),
            ],
            functional_identities=[
                IdentitySpec(
                    name="symmetry_xy",
                    identity_fn=_symmetry_xy,
                    description="R_M(x, y, z) = R_M(y, x, z)",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_equal_args(self):
        """R_M(x, x, x) = (1/sqrt(x) + 3x) / 6."""
        x = torch.tensor([1.0, 2.0, 4.0, 9.0], dtype=torch.float64)
        result = torchscience.special_functions.carlson_elliptic_integral_r_m(
            x, x, x
        )
        expected = (1.0 / torch.sqrt(x) + 3.0 * x) / 6.0
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_symmetry(self):
        """R_M is symmetric in x and y."""
        x = torch.tensor([1.0], dtype=torch.float64)
        y = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([3.0], dtype=torch.float64)
        rm_xyz = torchscience.special_functions.carlson_elliptic_integral_r_m(
            x, y, z
        )
        rm_yxz = torchscience.special_functions.carlson_elliptic_integral_r_m(
            y, x, z
        )
        torch.testing.assert_close(rm_xyz, rm_yxz, rtol=1e-10, atol=1e-10)

    def test_gradient(self):
        """First-order gradient via gradcheck."""
        # Use values where x + y - z is well away from 0 to avoid singularity
        # in the sqrt(xyz / (x + y - z)) term.
        x = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(
            torchscience.special_functions.carlson_elliptic_integral_r_m,
            (x, y, z),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )
