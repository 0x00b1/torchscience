import pytest
import torch
import torch.testing

import torchscience.special_functions

# Optional mpmath import for reference tests
try:
    import mpmath

    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


class TestHypergeometricPFQ:
    """Tests for the generalized hypergeometric function pFq."""

    def test_zero_returns_one(self):
        """Test that pFq(a;b;0) = 1 for any valid a, b."""
        # 2F1 case
        a = torch.tensor([1.0, 2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor(0.0, dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_p_f_q(a, b, z)
        torch.testing.assert_close(
            result,
            torch.tensor(1.0, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

        # 1F2 case
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0, 3.0], dtype=torch.float64)
        z = torch.tensor(0.0, dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_p_f_q(a, b, z)
        torch.testing.assert_close(
            result,
            torch.tensor(1.0, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_0f1_mpmath_reference(self):
        """Test 0F1 case against mpmath."""
        a = torch.tensor([], dtype=torch.float64)  # p=0
        b = torch.tensor([2.0], dtype=torch.float64)  # q=1
        z = torch.tensor(1.0, dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_p_f_q(a, b, z)
        expected = float(mpmath.hyp0f1(2.0, 1.0))

        torch.testing.assert_close(
            result,
            torch.tensor(expected, dtype=torch.float64),
            rtol=1e-8,
            atol=1e-10,
        )

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_1f1_mpmath_reference(self):
        """Test 1F1 case against mpmath."""
        a = torch.tensor([1.0], dtype=torch.float64)  # p=1
        b = torch.tensor([2.0], dtype=torch.float64)  # q=1
        z = torch.tensor(0.5, dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_p_f_q(a, b, z)
        expected = float(mpmath.hyp1f1(1.0, 2.0, 0.5))

        torch.testing.assert_close(
            result,
            torch.tensor(expected, dtype=torch.float64),
            rtol=1e-8,
            atol=1e-10,
        )

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_2f1_mpmath_reference(self):
        """Test 2F1 case against mpmath."""
        a = torch.tensor([1.0, 2.0], dtype=torch.float64)  # p=2
        b = torch.tensor([3.0], dtype=torch.float64)  # q=1
        z = torch.tensor(0.5, dtype=torch.float64)  # |z| < 1 required

        result = torchscience.special_functions.hypergeometric_p_f_q(a, b, z)
        expected = float(mpmath.hyp2f1(1.0, 2.0, 3.0, 0.5))

        torch.testing.assert_close(
            result,
            torch.tensor(expected, dtype=torch.float64),
            rtol=1e-6,
            atol=1e-8,
        )

    @pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not available")
    def test_1f2_mpmath_reference(self):
        """Test 1F2 case against mpmath."""
        a = torch.tensor([1.0], dtype=torch.float64)  # p=1
        b = torch.tensor([2.0, 3.0], dtype=torch.float64)  # q=2
        z = torch.tensor(0.5, dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_p_f_q(a, b, z)
        expected = float(mpmath.hyp1f2(1.0, 2.0, 3.0, 0.5))

        torch.testing.assert_close(
            result,
            torch.tensor(expected, dtype=torch.float64),
            rtol=1e-8,
            atol=1e-10,
        )

    def test_pole_at_nonpositive_integer_b(self):
        """Test that poles occur when b contains a non-positive integer."""
        a = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor(1.0, dtype=torch.float64)

        for b_val in [0.0, -1.0, -2.0]:
            b = torch.tensor([b_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_p_f_q(
                a, b, z
            )
            assert torch.isinf(result).item(), f"Expected inf for b={b_val}"

    def test_polynomial_when_a_nonpositive_integer(self):
        """Test that series terminates when a contains a non-positive integer.

        When any a[i] = -m for m >= 0, the series becomes a polynomial.
        """
        # a = [0] -> polynomial of degree 0, result = 1
        a = torch.tensor([0.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor(10.0, dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_p_f_q(a, b, z)
        torch.testing.assert_close(
            result,
            torch.tensor(1.0, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_convergence_p_less_than_q(self):
        """Test convergence when p < q (entire function)."""
        # 0F1: p=0, q=1
        a = torch.tensor([], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor(10.0, dtype=torch.float64)  # Large z is fine
        result = torchscience.special_functions.hypergeometric_p_f_q(a, b, z)
        assert torch.isfinite(result).item()

        # 1F2: p=1, q=2
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0, 3.0], dtype=torch.float64)
        z = torch.tensor(10.0, dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_p_f_q(a, b, z)
        assert torch.isfinite(result).item()

    def test_convergence_p_equals_q_plus_1(self):
        """Test convergence when p = q + 1 (converges for |z| < 1)."""
        # 2F1: p=2, q=1
        a = torch.tensor([1.0, 2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)

        # Should converge for |z| < 1
        z_small = torch.tensor(0.5, dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_p_f_q(
            a, b, z_small
        )
        assert torch.isfinite(result).item()

        # Should diverge for |z| > 1 (returns NaN)
        z_large = torch.tensor(2.0, dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_p_f_q(
            a, b, z_large
        )
        assert torch.isnan(result).item()

    def test_divergence_p_greater_than_q_plus_1(self):
        """Test divergence when p > q + 1."""
        # 3F1: p=3, q=1 (diverges)
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        b = torch.tensor([4.0], dtype=torch.float64)
        z = torch.tensor(0.5, dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_p_f_q(a, b, z)
        assert torch.isnan(result).item()

    def test_batched_computation(self):
        """Test batched computation with multiple values."""
        a = torch.tensor(
            [[1.0, 2.0], [0.5, 1.5]], dtype=torch.float64
        )  # Shape [2, 2]
        b = torch.tensor([[3.0], [2.0]], dtype=torch.float64)  # Shape [2, 1]
        z = torch.tensor([0.5, 0.25], dtype=torch.float64)  # Shape [2]

        result = torchscience.special_functions.hypergeometric_p_f_q(a, b, z)
        assert result.shape == (2,)
        assert torch.isfinite(result).all()

    def test_dtype_preservation(self):
        """Test that dtype is preserved."""
        a = torch.tensor([1.0, 2.0], dtype=torch.float32)
        b = torch.tensor([3.0], dtype=torch.float32)
        z = torch.tensor(0.5, dtype=torch.float32)
        result = torchscience.special_functions.hypergeometric_p_f_q(a, b, z)
        assert result.dtype == torch.float32

        a = torch.tensor([1.0, 2.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor(0.5, dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_p_f_q(a, b, z)
        assert result.dtype == torch.float64

    def test_consistency_with_hypergeometric_1_f_2(self):
        """Test that pFq with p=1, q=2 matches hypergeometric_1_f_2."""
        a_val, b1_val, b2_val, z_val = 1.5, 2.0, 3.0, 0.5

        # Using pFq
        a = torch.tensor([a_val], dtype=torch.float64)
        b = torch.tensor([b1_val, b2_val], dtype=torch.float64)
        z = torch.tensor(z_val, dtype=torch.float64)
        result_pfq = torchscience.special_functions.hypergeometric_p_f_q(
            a, b, z
        )

        # Using specialized 1F2
        a_tensor = torch.tensor(a_val, dtype=torch.float64)
        b1_tensor = torch.tensor(b1_val, dtype=torch.float64)
        b2_tensor = torch.tensor(b2_val, dtype=torch.float64)
        z_tensor = torch.tensor(z_val, dtype=torch.float64)
        result_1f2 = torchscience.special_functions.hypergeometric_1_f_2(
            a_tensor, b1_tensor, b2_tensor, z_tensor
        )

        torch.testing.assert_close(
            result_pfq, result_1f2, rtol=1e-8, atol=1e-10
        )

    def test_empty_upper_parameters(self):
        """Test with no upper parameters (0Fq case)."""
        a = torch.tensor([], dtype=torch.float64)  # p=0
        b = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor(1.0, dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_p_f_q(a, b, z)
        assert torch.isfinite(result).item()
