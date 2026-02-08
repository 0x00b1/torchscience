import scipy.special
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


def _gradient_identity(func):
    """Check Y_0'(z) = -Y_1(z)."""
    z = torch.tensor(
        [1.0, 2.0, 5.0, 10.0], dtype=torch.float64, requires_grad=True
    )
    y = func(z)
    grad = torch.autograd.grad(y.sum(), z)[0]
    expected = -torchscience.special_functions.bessel_y_1(z.detach())
    return grad, expected


class TestBesselY0(OpTestCase):
    """Tests for the Bessel function of the second kind Y_0."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="bessel_y_0",
            func=torchscience.special_functions.bessel_y_0,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z",
                    position=0,
                    default_real_range=(0.5, 20.0),
                ),
            ],
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_sparse_coo_basic",
                "test_sparse_csr_basic",
                "test_low_precision_forward",
                "test_gradgradcheck_complex",
            },
            special_values=[
                SpecialValue(
                    inputs=(1.0,),
                    expected=float(scipy.special.y0(1.0)),
                    description="Y_0(1)",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=float(scipy.special.y0(2.0)),
                    description="Y_0(2)",
                ),
                SpecialValue(
                    inputs=(5.0,),
                    expected=float(scipy.special.y0(5.0)),
                    description="Y_0(5)",
                ),
            ],
            functional_identities=[
                IdentitySpec(
                    name="gradient_identity",
                    identity_fn=_gradient_identity,
                    rtol=1e-6,
                    atol=1e-6,
                    description="Y_0'(z) = -Y_1(z)",
                ),
            ],
            singularities=[
                SingularitySpec(
                    type="pole",
                    locations=lambda: iter([0.0]),
                    expected_behavior="inf",
                    description="Logarithmic singularity at z=0",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_forward_against_scipy(self):
        """Compare against scipy.special.y0."""
        z_vals = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]
        z = torch.tensor(z_vals, dtype=torch.float64)
        result = torchscience.special_functions.bessel_y_0(z)
        expected = torch.tensor(
            [scipy.special.y0(v) for v in z_vals], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_known_zeros(self):
        """Test near known zeros of Y_0."""
        zeros = torch.tensor(
            [0.893577, 3.957678, 7.086051], dtype=torch.float64
        )
        result = torchscience.special_functions.bessel_y_0(zeros)
        torch.testing.assert_close(
            result,
            torch.zeros_like(result),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_negative_real_returns_nan(self):
        """Test that negative real arguments return NaN."""
        z = torch.tensor([-1.0, -2.0, -5.0], dtype=torch.float64)
        result = torchscience.special_functions.bessel_y_0(z)
        assert result.isnan().all()

    def test_recurrence_with_bessel_y(self):
        """Test Y_{n-1}(z) + Y_{n+1}(z) = (2n/z) Y_n(z) for n=1."""
        z = torch.tensor([1.0, 2.0, 5.0, 10.0], dtype=torch.float64)
        y0 = torchscience.special_functions.bessel_y_0(z)
        n_tensor = torch.tensor([2.0], dtype=torch.float64)
        y2 = torchscience.special_functions.bessel_y(n_tensor, z)
        y1 = torchscience.special_functions.bessel_y_1(z)
        torch.testing.assert_close(
            y0 + y2, (2.0 / z) * y1, rtol=1e-8, atol=1e-10
        )
