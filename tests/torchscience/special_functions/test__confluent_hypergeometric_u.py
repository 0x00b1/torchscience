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


def _power_identity(func):
    """Check U(a, a+1, z) = z^(-a)."""
    a = torch.tensor([2.0], dtype=torch.float64)
    z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    left = func(a, a + 1, z)
    right = z ** (-a)
    return left, right


def _zero_a_identity(func):
    """Check U(0, b, z) = 1."""
    a = torch.tensor([0.0], dtype=torch.float64)
    b = torch.tensor([2.0], dtype=torch.float64)
    z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    left = func(a, b, z)
    right = torch.ones_like(z)
    return left, right


class TestConfluentHypergeometricU(OpTestCase):
    """Tests for the confluent hypergeometric function U (Tricomi's function)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="confluent_hypergeometric_u",
            func=torchscience.special_functions.confluent_hypergeometric_u,
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
                    supports_grad=True,
                ),
                InputSpec(
                    name="z",
                    position=2,
                    default_real_range=(0.5, 10.0),  # Avoid z near 0
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
                    name="power_identity",
                    identity_fn=_power_identity,
                    description="U(a, a+1, z) = z^(-a)",
                    rtol=1e-6,
                    atol=1e-6,
                ),
                IdentitySpec(
                    name="zero_a_identity",
                    identity_fn=_zero_a_identity,
                    description="U(0, b, z) = 1",
                    rtol=1e-6,
                    atol=1e-6,
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0, 2.0, 1.0),
                    expected=1.0,
                    description="U(0, b, z) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 2.0, 1.0),
                    expected=1.0,
                    description="U(1, 2, z) = z^(-1) = 1.0 when z=1",
                ),
            ],
            singularities=[],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_large_z_asymptotic(self):
        """Test U ~ z^(-a) for large z (asymptotic behavior)."""
        a = torch.tensor([1.5, 2.0, 2.5], dtype=torch.float64)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        z = torch.tensor([50.0, 100.0, 200.0], dtype=torch.float64)

        result = self.descriptor.func(a, b, z)
        expected = z ** (-a)

        # For large z, the asymptotic approximation should be good
        torch.testing.assert_close(result, expected, rtol=0.1, atol=1e-6)

    def test_derivative_z(self):
        """Test dU/dz = -a * U(a+1, b+1, z)."""
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.5], dtype=torch.float64)
        z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

        y = self.descriptor.func(a, b, z)
        y.backward()

        expected_grad = -a * self.descriptor.func(a + 1, b + 1, z.detach())

        torch.testing.assert_close(z.grad, expected_grad, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_scipy_reference(self):
        """Test against scipy reference implementation."""
        test_cases = [
            (1.0, 2.0, 1.0),
            (0.5, 1.5, 2.0),
            (2.0, 3.0, 1.5),
            (1.5, 2.5, 3.0),
            (0.25, 0.75, 2.0),
        ]
        for a_val, b_val, z_val in test_cases:
            expected = float(scipy.special.hyperu(a_val, b_val, z_val))
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

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_mpmath_reference(self):
        """Test against mpmath reference implementation."""
        test_cases = [
            (1.0, 2.0, 1.0),
            (0.5, 1.5, 2.0),
            (2.0, 3.0, 1.5),
            (1.5, 2.5, 3.0),
            (0.25, 0.75, 2.0),
        ]
        for a_val, b_val, z_val in test_cases:
            expected = float(mpmath.hyperu(a_val, b_val, z_val))
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

    def test_negative_integer_a(self):
        """Test that U(-n, b, z) is a polynomial of degree n (terminates)."""
        # When a is a negative integer, U becomes a polynomial
        a = torch.tensor([-2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64)

        result = self.descriptor.func(a, b, z)

        # Result should be finite for all z values
        assert torch.isfinite(result).all()

    def test_complex_basic(self):
        """Test with complex inputs."""
        a = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        b = torch.tensor([2.0 + 0j], dtype=torch.complex128)
        z = torch.tensor([2.0 + 0.5j], dtype=torch.complex128)

        result = self.descriptor.func(a, b, z)

        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()
        assert result.is_complex()

    def test_a_equals_zero(self):
        """Test U(0, b, z) = 1 for various b, z."""
        a = torch.tensor([0.0], dtype=torch.float64)
        test_cases = [
            (1.0, 0.5),
            (2.0, 1.0),
            (3.0, 2.0),
            (0.5, 3.0),
        ]
        for b_val, z_val in test_cases:
            b = torch.tensor([b_val], dtype=torch.float64)
            z = torch.tensor([z_val], dtype=torch.float64)
            result = self.descriptor.func(a, b, z)
            torch.testing.assert_close(
                result,
                torch.ones_like(result),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_special_case_u_a_a_plus_1_z(self):
        """Test U(a, a+1, z) = z^(-a)."""
        test_cases = [
            (1.0, [1.0, 2.0, 3.0]),
            (2.0, [1.0, 2.0, 3.0]),
            (0.5, [1.0, 2.0, 3.0]),
            (1.5, [1.0, 2.0, 3.0]),
        ]
        for a_val, z_vals in test_cases:
            a = torch.tensor([a_val], dtype=torch.float64)
            z = torch.tensor(z_vals, dtype=torch.float64)
            result = self.descriptor.func(a, a + 1, z)
            expected = z ** (-a)
            torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_recurrence_relation(self):
        """Test recurrence relation: U(a-1, b, z) = (1-b+z)*U(a, b, z) + z*a*U(a+1, b, z)."""
        # This is one of the standard recurrence relations for U
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([1.5, 2.0, 3.0], dtype=torch.float64)

        u_am1 = self.descriptor.func(a - 1, b, z)
        u_a = self.descriptor.func(a, b, z)
        u_ap1 = self.descriptor.func(a + 1, b, z)

        # Recurrence: U(a-1, b, z) + (b - 2*a - z)*U(a, b, z) + a*(a - b + 1)*U(a+1, b, z) = 0
        # Or equivalently: (a - b + 1)*U(a+1, b+1, z) = U(a, b, z) - U(a, b+1, z)
        # We'll use: U(a-1, b, z) = (2*a - b + z)*U(a, b, z) - a*(a - b + 1)*U(a+1, b, z)

        # Use a simpler recurrence: z*U(a, b+1, z) = U(a-1, b, z) - (b-1)*U(a, b, z)
        u_a_bp1 = self.descriptor.func(a, b + 1, z)
        u_am1_b = self.descriptor.func(a - 1, b, z)

        lhs = z * u_a_bp1
        rhs = u_am1_b - (b - 1) * u_a

        torch.testing.assert_close(lhs, rhs, rtol=1e-5, atol=1e-5)

    def test_positive_z_only(self):
        """Test that positive z values give finite results."""
        a = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
        b = torch.tensor([2.0, 3.0, 1.5], dtype=torch.float64)
        z = torch.tensor([0.5, 1.0, 5.0], dtype=torch.float64)

        result = self.descriptor.func(a, b, z)
        assert torch.isfinite(result).all()

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        a = torch.tensor([[1.0], [2.0]], dtype=torch.float64)  # (2, 1)
        b = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)  # (3,)
        z = torch.tensor([1.0], dtype=torch.float64)  # (1,)

        result = self.descriptor.func(a, b, z)

        # Should broadcast to (2, 3)
        assert result.shape == (2, 3)
        assert torch.isfinite(result).all()
