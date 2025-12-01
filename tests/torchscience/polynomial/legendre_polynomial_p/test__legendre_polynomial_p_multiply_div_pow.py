"""Tests for Legendre polynomial multiply, divmod, div, mod, and pow."""

import numpy as np
import pytest
import torch

from torchscience.polynomial import (
    legendre_polynomial_p,
    legendre_polynomial_p_evaluate,
)
from torchscience.polynomial._legendre_polynomial_p import (
    legendre_polynomial_p_div,
    legendre_polynomial_p_divmod,
    legendre_polynomial_p_mod,
    legendre_polynomial_p_multiply,
    legendre_polynomial_p_pow,
)


class TestLegendrePolynomialPMultiply:
    """Tests for legendre_polynomial_p_multiply."""

    def test_multiply_degree_0(self):
        """Multiply by constant."""
        a = legendre_polynomial_p(torch.tensor([2.0]))  # 2*P_0
        b = legendre_polynomial_p(torch.tensor([3.0]))  # 3*P_0
        c = legendre_polynomial_p_multiply(a, b)
        torch.testing.assert_close(c.coeffs, torch.tensor([6.0]))

    def test_multiply_degree_1(self):
        """P_1 * P_1 = (1/3)*P_0 + (2/3)*P_2."""
        a = legendre_polynomial_p(torch.tensor([0.0, 1.0]))  # P_1
        b = legendre_polynomial_p(torch.tensor([0.0, 1.0]))  # P_1
        c = legendre_polynomial_p_multiply(a, b)
        # P_1^2 = x^2 = (2*P_2 + P_0)/3
        expected = torch.tensor([1.0 / 3, 0.0, 2.0 / 3])
        torch.testing.assert_close(c.coeffs, expected, atol=1e-6, rtol=1e-6)

    def test_multiply_vs_numpy(self):
        """Verify against NumPy's legmul."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        b = legendre_polynomial_p(torch.tensor([4.0, 5.0]))
        c = legendre_polynomial_p_multiply(a, b)
        expected = np.polynomial.legendre.legmul([1, 2, 3], [4, 5])
        torch.testing.assert_close(
            c.coeffs, torch.tensor(expected).float(), atol=1e-5, rtol=1e-5
        )

    def test_multiply_operator(self):
        """Test * operator."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        b = legendre_polynomial_p(torch.tensor([3.0, 4.0]))
        c = a * b
        expected = np.polynomial.legendre.legmul([1, 2], [3, 4])
        torch.testing.assert_close(
            c.coeffs, torch.tensor(expected).float(), atol=1e-5, rtol=1e-5
        )

    def test_multiply_commutativity(self):
        """Verify a * b = b * a."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        b = legendre_polynomial_p(torch.tensor([4.0, 5.0]))
        c1 = legendre_polynomial_p_multiply(a, b)
        c2 = legendre_polynomial_p_multiply(b, a)
        torch.testing.assert_close(c1.coeffs, c2.coeffs, atol=1e-6, rtol=1e-6)

    def test_multiply_identity(self):
        """Multiplying by P_0 = 1 leaves series unchanged."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        one = legendre_polynomial_p(torch.tensor([1.0]))
        result = legendre_polynomial_p_multiply(a, one)
        torch.testing.assert_close(
            result.coeffs, a.coeffs, atol=1e-6, rtol=1e-6
        )

    def test_multiply_zero(self):
        """Multiplying by zero gives zero."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        zero = legendre_polynomial_p(torch.tensor([0.0]))
        result = legendre_polynomial_p_multiply(a, zero)
        # Result should be all zeros
        assert torch.allclose(result.coeffs, torch.zeros_like(result.coeffs))


class TestLegendrePolynomialPDivmod:
    """Tests for legendre_polynomial_p_divmod."""

    def test_divmod_basic(self):
        """Test basic division with remainder."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        b = legendre_polynomial_p(torch.tensor([1.0, 1.0]))
        q, r = legendre_polynomial_p_divmod(a, b)
        # Verify: a = b * q + r by evaluating at sample points
        x = torch.linspace(-0.9, 0.9, 20)
        lhs = legendre_polynomial_p_evaluate(a, x)
        bq = legendre_polynomial_p_multiply(b, q)
        bq_plus_r = bq + r
        rhs = legendre_polynomial_p_evaluate(bq_plus_r, x)
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_divmod_exact_division(self):
        """Test case where division is exact (no remainder)."""
        # Create b * c, then divide by b, should get c back
        b = legendre_polynomial_p(torch.tensor([1.0, 1.0]))
        c = legendre_polynomial_p(torch.tensor([2.0, 3.0]))
        a = legendre_polynomial_p_multiply(b, c)
        q, r = legendre_polynomial_p_divmod(a, b)
        # Quotient should be close to c
        torch.testing.assert_close(q.coeffs, c.coeffs, atol=1e-5, rtol=1e-5)
        # Remainder should be near zero
        assert r.coeffs.abs().max() < 1e-10

    def test_divmod_vs_numpy(self):
        """Verify against NumPy's legdiv."""
        a_coeffs = [1.0, 2.0, 3.0]
        b_coeffs = [1.0, 1.0]
        a = legendre_polynomial_p(torch.tensor(a_coeffs))
        b = legendre_polynomial_p(torch.tensor(b_coeffs))
        q, r = legendre_polynomial_p_divmod(a, b)
        q_np, r_np = np.polynomial.legendre.legdiv(a_coeffs, b_coeffs)
        torch.testing.assert_close(
            q.coeffs, torch.tensor(q_np).float(), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            r.coeffs, torch.tensor(r_np).float(), atol=1e-5, rtol=1e-5
        )

    def test_div_operator(self):
        """Test // operator returns quotient."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        b = legendre_polynomial_p(torch.tensor([1.0, 1.0]))
        q = a // b
        q_expected, _ = legendre_polynomial_p_divmod(a, b)
        torch.testing.assert_close(q.coeffs, q_expected.coeffs)

    def test_mod_operator(self):
        """Test % operator returns remainder."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        b = legendre_polynomial_p(torch.tensor([1.0, 1.0]))
        r = a % b
        _, r_expected = legendre_polynomial_p_divmod(a, b)
        torch.testing.assert_close(r.coeffs, r_expected.coeffs)


class TestLegendrePolynomialPDiv:
    """Tests for legendre_polynomial_p_div."""

    def test_div_returns_quotient(self):
        """Test that div returns only quotient."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        b = legendre_polynomial_p(torch.tensor([1.0, 1.0]))
        q = legendre_polynomial_p_div(a, b)
        q_expected, _ = legendre_polynomial_p_divmod(a, b)
        torch.testing.assert_close(q.coeffs, q_expected.coeffs)


class TestLegendrePolynomialPMod:
    """Tests for legendre_polynomial_p_mod."""

    def test_mod_returns_remainder(self):
        """Test that mod returns only remainder."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        b = legendre_polynomial_p(torch.tensor([1.0, 1.0]))
        r = legendre_polynomial_p_mod(a, b)
        _, r_expected = legendre_polynomial_p_divmod(a, b)
        torch.testing.assert_close(r.coeffs, r_expected.coeffs)


class TestLegendrePolynomialPPow:
    """Tests for legendre_polynomial_p_pow."""

    def test_pow_0(self):
        """p^0 = 1 (constant polynomial P_0)."""
        p = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        result = legendre_polynomial_p_pow(p, 0)
        torch.testing.assert_close(result.coeffs, torch.tensor([1.0]))

    def test_pow_1(self):
        """p^1 = p."""
        p = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        result = legendre_polynomial_p_pow(p, 1)
        torch.testing.assert_close(result.coeffs, p.coeffs)

    def test_pow_2(self):
        """p^2 = p * p."""
        p = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        result = legendre_polynomial_p_pow(p, 2)
        expected = legendre_polynomial_p_multiply(p, p)
        torch.testing.assert_close(
            result.coeffs, expected.coeffs, atol=1e-5, rtol=1e-5
        )

    def test_pow_3(self):
        """p^3 = p * p * p."""
        p = legendre_polynomial_p(torch.tensor([1.0, 1.0]))  # P_0 + P_1
        result = legendre_polynomial_p_pow(p, 3)
        p2 = legendre_polynomial_p_multiply(p, p)
        expected = legendre_polynomial_p_multiply(p2, p)
        torch.testing.assert_close(
            result.coeffs, expected.coeffs, atol=1e-5, rtol=1e-5
        )

    def test_pow_4(self):
        """p^4 via binary exponentiation."""
        p = legendre_polynomial_p(torch.tensor([1.0, 1.0]))
        result = legendre_polynomial_p_pow(p, 4)
        # Manual calculation: p^4 = ((p^2)^2)
        p2 = legendre_polynomial_p_multiply(p, p)
        expected = legendre_polynomial_p_multiply(p2, p2)
        torch.testing.assert_close(
            result.coeffs, expected.coeffs, atol=1e-5, rtol=1e-5
        )

    def test_pow_operator(self):
        """Test ** operator."""
        p = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        result = p**2
        expected = legendre_polynomial_p_multiply(p, p)
        torch.testing.assert_close(
            result.coeffs, expected.coeffs, atol=1e-5, rtol=1e-5
        )

    def test_pow_negative_raises(self):
        """Negative exponent should raise ValueError."""
        p = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        with pytest.raises(ValueError, match="non-negative"):
            legendre_polynomial_p_pow(p, -1)

    def test_pow_consistency_via_evaluation(self):
        """Verify p^n evaluates consistently."""
        p = legendre_polynomial_p(
            torch.tensor([0.5, 0.5])
        )  # 0.5*P_0 + 0.5*P_1
        result = legendre_polynomial_p_pow(p, 3)
        x = torch.tensor([0.0, 0.5, 1.0])
        # p(x) = 0.5*1 + 0.5*x = 0.5*(1 + x)
        # p^3(x) = 0.5^3 * (1 + x)^3
        p_vals = legendre_polynomial_p_evaluate(p, x)
        p3_vals = legendre_polynomial_p_evaluate(result, x)
        expected = p_vals**3
        torch.testing.assert_close(p3_vals, expected, atol=1e-5, rtol=1e-5)
