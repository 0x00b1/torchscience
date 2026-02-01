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


def _zero_identity(func):
    """Check 1F2(a;b1,b2;0) = 1."""
    a = torch.tensor([1.0, 2.0], dtype=torch.float64)
    b1 = torch.tensor([2.0, 3.0], dtype=torch.float64)
    b2 = torch.tensor([3.0, 4.0], dtype=torch.float64)
    z = torch.tensor([0.0, 0.0], dtype=torch.float64)
    left = func(a, b1, b2, z)
    right = torch.ones_like(left)
    return left, right


class TestHypergeometric1F2(OpTestCase):
    """Tests for the hypergeometric function 1F2."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="hypergeometric_1_f_2",
            func=torchscience.special_functions.hypergeometric_1_f_2,
            arity=4,
            input_specs=[
                InputSpec(
                    name="a",
                    position=0,
                    default_real_range=(-3.0, 5.0),
                    supports_grad=True,
                ),
                InputSpec(
                    name="b1",
                    position=1,
                    default_real_range=(0.5, 5.0),
                    excluded_values={0.0, -1.0, -2.0, -3.0, -4.0, -5.0},
                    supports_grad=True,
                ),
                InputSpec(
                    name="b2",
                    position=2,
                    default_real_range=(0.5, 5.0),
                    excluded_values={0.0, -1.0, -2.0, -3.0, -4.0, -5.0},
                    supports_grad=True,
                ),
                InputSpec(
                    name="z",
                    position=3,
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
                    description="1F2(a;b1,b2;0) = 1",
                    rtol=1e-10,
                    atol=1e-10,
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0, 2.0, 3.0, 0.0),
                    expected=1.0,
                    description="1F2(1;2,3;0) = 1",
                ),
            ],
            singularities=[],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_zero_returns_one(self):
        """Test that 1F2(a;b1,b2;0) = 1 for any valid a, b1, b2."""
        a = torch.tensor([0.5, 1.0, 2.0, 3.5], dtype=torch.float64)
        b1 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        b2 = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        z = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = self.descriptor.func(a, b1, b2, z)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_mpmath_reference(self):
        """Test against mpmath reference implementation."""
        test_cases = [
            (1.0, 2.0, 3.0, 0.5),
            (0.5, 1.5, 2.5, 1.0),
            (2.0, 3.0, 4.0, -0.5),
            (1.0, 1.0, 2.0, 0.25),
        ]
        for a_val, b1_val, b2_val, z_val in test_cases:
            expected = float(mpmath.hyp1f2(a_val, b1_val, b2_val, z_val))
            a = torch.tensor([a_val], dtype=torch.float64)
            b1 = torch.tensor([b1_val], dtype=torch.float64)
            b2 = torch.tensor([b2_val], dtype=torch.float64)
            z = torch.tensor([z_val], dtype=torch.float64)
            result = self.descriptor.func(a, b1, b2, z)
            torch.testing.assert_close(
                result,
                torch.tensor([expected], dtype=torch.float64),
                rtol=1e-8,
                atol=1e-10,
            )

    def test_pole_at_nonpositive_integer_b1(self):
        """Test that poles occur when b1 is a non-positive integer."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b2 = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64)

        for b1_val in [0.0, -1.0, -2.0]:
            b1 = torch.tensor([b1_val], dtype=torch.float64)
            result = self.descriptor.func(a, b1, b2, z)
            assert torch.isinf(result).all(), f"Expected inf for b1={b1_val}"

    def test_pole_at_nonpositive_integer_b2(self):
        """Test that poles occur when b2 is a non-positive integer."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b1 = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([1.0], dtype=torch.float64)

        for b2_val in [0.0, -1.0, -2.0]:
            b2 = torch.tensor([b2_val], dtype=torch.float64)
            result = self.descriptor.func(a, b1, b2, z)
            assert torch.isinf(result).all(), f"Expected inf for b2={b2_val}"

    def test_polynomial_when_a_nonpositive_integer(self):
        """Test that series terminates when a is a non-positive integer.

        When a = -m for m >= 0, the series becomes a polynomial of degree m.
        """
        # a = 0 -> polynomial of degree 0, result = 1
        a = torch.tensor([0.0], dtype=torch.float64)
        b1 = torch.tensor([2.0], dtype=torch.float64)
        b2 = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([10.0], dtype=torch.float64)
        result = self.descriptor.func(a, b1, b2, z)
        torch.testing.assert_close(
            result,
            torch.tensor([1.0], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_positive_z_finite(self):
        """Test that the function is finite for positive z and valid parameters."""
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        b1 = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        b2 = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)
        z = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        result = self.descriptor.func(a, b1, b2, z)
        assert torch.isfinite(result).all()

    def test_negative_z_finite(self):
        """Test that the function is finite for negative z and valid parameters."""
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        b1 = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        b2 = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)
        z = torch.tensor([-0.5, -1.0, -2.0], dtype=torch.float64)
        result = self.descriptor.func(a, b1, b2, z)
        assert torch.isfinite(result).all()

    def test_broadcasting(self):
        """Test that inputs broadcast correctly."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b1 = torch.tensor([2.0, 3.0], dtype=torch.float64)
        b2 = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.5], dtype=torch.float64)
        result = self.descriptor.func(a, b1, b2, z)
        assert result.shape == (2,)
        assert torch.isfinite(result).all()
