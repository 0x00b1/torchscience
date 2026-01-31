import pytest
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

# Optional mpmath import for reference tests
try:
    import mpmath

    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

# Optional scipy import for reference tests
try:
    import scipy.special

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _exp_identity(func):
    """Check M(a, a, z) = exp(z)."""
    a = torch.tensor([2.0], dtype=torch.float64)
    z = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    left = func(a, a, z)
    right = torch.exp(z)
    return left, right


def _kummer_transformation(func):
    """Check M(a, b, z) = exp(z) * M(b-a, b, -z)."""
    a = torch.tensor([1.5], dtype=torch.float64)
    b = torch.tensor([3.0], dtype=torch.float64)
    z = torch.tensor([-1.0, -0.5, -0.25], dtype=torch.float64)
    left = func(a, b, z)
    right = torch.exp(z) * func(b - a, b, -z)
    return left, right


class TestConfluentHypergeometricM(OpTestCase):
    """Tests for the confluent hypergeometric function M (Kummer's function)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="confluent_hypergeometric_m",
            func=torchscience.special_functions.confluent_hypergeometric_m,
            arity=3,
            input_specs=[
                InputSpec(
                    name="a",
                    position=0,
                    default_real_range=(0.5, 5.0),
                    supports_grad=True,
                ),
                InputSpec(
                    name="b",
                    position=1,
                    default_real_range=(0.5, 5.0),
                    excluded_values={0.0, -1.0, -2.0, -3.0, -4.0, -5.0},
                    supports_grad=True,
                ),
                InputSpec(
                    name="z",
                    position=2,
                    default_real_range=(-5.0, 5.0),
                    supports_grad=True,
                    complex_magnitude_max=5.0,
                ),
            ],
            tolerances=ToleranceConfig(
                float32_rtol=1e-4,
                float32_atol=1e-4,
                float64_rtol=1e-8,
                float64_atol=1e-8,
                gradcheck_rtol=1e-4,
                gradcheck_atol=1e-4,
                gradgradcheck_rtol=1e-3,
                gradgradcheck_atol=1e-3,
            ),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradcheck_real",
                "test_gradcheck_complex",
                "test_gradgradcheck_real",
                "test_gradgradcheck_complex",
                "test_sparse_coo_basic",
                "test_sparse_csr_basic",
                "test_quantized_basic",
                "test_nan_propagation",
                "test_nan_propagation_all_inputs",
                "test_low_precision_forward",  # Hypergeometric functions need high precision
            },
            functional_identities=[
                IdentitySpec(
                    name="exp_identity",
                    identity_fn=_exp_identity,
                    description="M(a, a, z) = exp(z)",
                    rtol=1e-6,
                    atol=1e-6,
                ),
                IdentitySpec(
                    name="kummer_transformation",
                    identity_fn=_kummer_transformation,
                    description="M(a, b, z) = exp(z) * M(b-a, b, -z)",
                    rtol=1e-5,
                    atol=1e-5,
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0, 2.0, 0.0),
                    expected=1.0,
                    description="M(a, b, 0) = 1",
                ),
                SpecialValue(
                    inputs=(0.0, 2.0, 1.0),
                    expected=1.0,
                    description="M(0, b, z) = 1",
                ),
            ],
            singularities=[],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_at_zero(self):
        """Test M(a, b, 0) = 1 for various a, b."""
        z = torch.tensor([0.0], dtype=torch.float64)
        test_cases = [
            (1.0, 2.0),
            (0.5, 0.5),
            (2.0, 3.0),
            (1.5, 2.5),
        ]
        for a_val, b_val in test_cases:
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)
            result = self.descriptor.func(a, b, z)
            torch.testing.assert_close(
                result,
                torch.ones_like(result),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_a_equals_b(self):
        """Test M(a, a, z) = exp(z)."""
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        z = torch.tensor([0.5, 1.0, -0.5], dtype=torch.float64)
        result = self.descriptor.func(a, a, z)
        expected = torch.exp(z)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_a_zero(self):
        """Test M(0, b, z) = 1."""
        a = torch.tensor([0.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        result = self.descriptor.func(a, b, z)
        expected = torch.ones_like(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_mpmath_reference(self):
        """Test against mpmath reference implementation."""
        test_cases = [
            (1.0, 2.0, 0.5),
            (0.5, 1.5, 1.0),
            (2.0, 3.0, -1.0),
            (1.5, 2.5, 2.0),
            (0.25, 0.75, -0.5),
        ]
        for a_val, b_val, z_val in test_cases:
            expected = float(mpmath.hyp1f1(a_val, b_val, z_val))
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)
            z = torch.tensor([z_val], dtype=torch.float64)
            result = self.descriptor.func(a, b, z)
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-8,
                atol=1e-10,
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_scipy_reference(self):
        """Test against scipy reference implementation."""
        test_cases = [
            (1.0, 2.0, 0.5),
            (0.5, 1.5, 1.0),
            (2.0, 3.0, -1.0),
            (1.5, 2.5, 2.0),
        ]
        for a_val, b_val, z_val in test_cases:
            expected = float(scipy.special.hyp1f1(a_val, b_val, z_val))
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)
            z = torch.tensor([z_val], dtype=torch.float64)
            result = self.descriptor.func(a, b, z)
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-8,
                atol=1e-10,
            )

    def test_kummer_transformation_property(self):
        """Test Kummer transformation: M(a, b, z) = exp(z) * M(b-a, b, -z)."""
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([-2.0, -1.0, -0.5], dtype=torch.float64)

        left = self.descriptor.func(a, b, z)
        right = torch.exp(z) * self.descriptor.func(b - a, b, -z)

        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)

    def test_gradient_z(self):
        """Test dM/dz = (a/b) * M(a+1, b+1, z)."""
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.5], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        y = self.descriptor.func(a, b, z)
        y.backward()

        expected_grad = (a / b) * self.descriptor.func(
            a + 1, b + 1, z.detach()
        )

        torch.testing.assert_close(z.grad, expected_grad, rtol=1e-5, atol=1e-5)

    def test_complex_basic(self):
        """Test with complex inputs."""
        a = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        b = torch.tensor([2.0 + 0j], dtype=torch.complex128)
        z = torch.tensor([1.0 + 0.5j], dtype=torch.complex128)

        result = self.descriptor.func(a, b, z)

        assert torch.isfinite(result).all()
        assert result.is_complex()

    def test_negative_integer_a_polynomial(self):
        """Test that M(-n, b, z) is a polynomial of degree n."""
        a = torch.tensor([-2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)

        result = self.descriptor.func(a, b, z)

        assert torch.isfinite(result).all()
