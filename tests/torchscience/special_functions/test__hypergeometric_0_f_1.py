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


def _zero_identity(func):
    """Check 0F1(;b;0) = 1."""
    b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    z = torch.tensor([0.0], dtype=torch.float64)
    left = func(b, z)
    right = torch.ones_like(left)
    return left, right


class TestHypergeometric0F1(OpTestCase):
    """Tests for the hypergeometric function 0F1."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="hypergeometric_0_f_1",
            func=torchscience.special_functions.hypergeometric_0_f_1,
            arity=2,
            input_specs=[
                InputSpec(
                    name="b",
                    position=0,
                    default_real_range=(0.5, 5.0),
                    excluded_values={0.0, -1.0, -2.0, -3.0, -4.0, -5.0},
                    supports_grad=True,
                ),
                InputSpec(
                    name="z",
                    position=1,
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
                "test_low_precision_forward",
            },
            functional_identities=[
                IdentitySpec(
                    name="zero_identity",
                    identity_fn=_zero_identity,
                    description="0F1(;b;0) = 1",
                    rtol=1e-10,
                    atol=1e-10,
                ),
            ],
            special_values=[
                SpecialValue(
                    input_values={"b": 1.0, "z": 0.0},
                    expected_value=1.0,
                    description="0F1(;1;0) = 1",
                ),
            ],
            singularities=[],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_zero_returns_one(self):
        """Test that 0F1(;b;0) = 1 for any b."""
        b = torch.tensor([0.5, 1.0, 2.0, 3.5], dtype=torch.float64)
        z = torch.tensor([0.0], dtype=torch.float64)
        result = self.descriptor.func(b, z)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_mpmath_reference(self):
        """Test against mpmath reference implementation."""
        test_cases = [
            (1.0, 1.0),
            (2.0, 0.5),
            (1.5, 2.0),
            (3.0, -1.0),
        ]
        for b_val, z_val in test_cases:
            expected = float(mpmath.hyp0f1(b_val, z_val))
            b = torch.tensor([b_val], dtype=torch.float64)
            z = torch.tensor([z_val], dtype=torch.float64)
            result = self.descriptor.func(b, z)
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
            (1.0, 1.0),
            (2.0, 0.5),
            (1.5, 2.0),
            (3.0, -1.0),
        ]
        for b_val, z_val in test_cases:
            expected = scipy.special.hyp0f1(b_val, z_val)
            b = torch.tensor([b_val], dtype=torch.float64)
            z = torch.tensor([z_val], dtype=torch.float64)
            result = self.descriptor.func(b, z)
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-8,
                atol=1e-10,
            )

    def test_bessel_relation(self):
        """Test relationship with Bessel functions.

        I_v(z) = (z/2)^v / Gamma(v+1) * 0F1(;v+1;z^2/4)
        """
        v = 1.0
        z_bessel = 2.0

        # Compute 0F1(;v+1;z^2/4)
        b = torch.tensor([v + 1], dtype=torch.float64)
        z = torch.tensor([z_bessel**2 / 4], dtype=torch.float64)
        hyp_result = self.descriptor.func(b, z)

        # Expected: I_v(z) * Gamma(v+1) / (z/2)^v
        import math

        bessel_i = torchscience.special_functions.modified_bessel_i(
            torch.tensor([v], dtype=torch.float64),
            torch.tensor([z_bessel], dtype=torch.float64),
        )
        gamma_v1 = math.gamma(v + 1)
        z_half_v = (z_bessel / 2) ** v

        expected = bessel_i * gamma_v1 / z_half_v

        torch.testing.assert_close(hyp_result, expected, rtol=1e-6, atol=1e-8)

    def test_pole_at_nonpositive_integer_b(self):
        """Test that poles occur when b is a non-positive integer."""
        z = torch.tensor([1.0], dtype=torch.float64)

        for b_val in [0.0, -1.0, -2.0]:
            b = torch.tensor([b_val], dtype=torch.float64)
            result = self.descriptor.func(b, z)
            assert torch.isinf(result).all(), f"Expected inf for b={b_val}"

    def test_positive_z_finite(self):
        """Test that the function is finite for positive z and valid b."""
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        z = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        result = self.descriptor.func(b, z)
        assert torch.isfinite(result).all()

    def test_negative_z_finite(self):
        """Test that the function is finite for negative z and valid b."""
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        z = torch.tensor([-0.5, -1.0, -2.0], dtype=torch.float64)
        result = self.descriptor.func(b, z)
        assert torch.isfinite(result).all()
