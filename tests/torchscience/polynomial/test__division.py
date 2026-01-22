"""Tests for polynomial division operations."""

import numpy as np
import pytest
import torch

from torchscience.polynomial import (
    DegreeError,
    polynomial,
    polynomial_add,
    polynomial_div,
    polynomial_divmod,
    polynomial_equal,
    polynomial_mod,
    polynomial_multiply,
)


class TestPolynomialDivmod:
    """Tests for polynomial_divmod."""

    def test_exact_division(self):
        """(x^3 - 1) / (x - 1) = x^2 + x + 1 with remainder 0."""
        # x^3 - 1 = [-1, 0, 0, 1]
        p = polynomial(torch.tensor([-1.0, 0.0, 0.0, 1.0]))
        # x - 1 = [-1, 1]
        q = polynomial(torch.tensor([-1.0, 1.0]))

        quotient, remainder = polynomial_divmod(p, q)

        # Expected: x^2 + x + 1 = [1, 1, 1]
        expected_q = polynomial(torch.tensor([1.0, 1.0, 1.0]))
        assert polynomial_equal(quotient, expected_q, tol=1e-6)
        # Remainder should be zero (or near-zero)
        assert torch.allclose(remainder, torch.zeros(1), atol=1e-6)

    def test_division_with_remainder(self):
        """(x^2 + 1) / (x - 1) = x + 1 with remainder 2."""
        # x^2 + 1 = [1, 0, 1]
        p = polynomial(torch.tensor([1.0, 0.0, 1.0]))
        # x - 1 = [-1, 1]
        q = polynomial(torch.tensor([-1.0, 1.0]))

        quotient, remainder = polynomial_divmod(p, q)

        # Expected quotient: x + 1 = [1, 1]
        expected_q = polynomial(torch.tensor([1.0, 1.0]))
        assert polynomial_equal(quotient, expected_q, tol=1e-6)
        # Expected remainder: 2
        assert torch.allclose(remainder, torch.tensor([2.0]), atol=1e-6)

    def test_dividend_smaller_degree(self):
        """When deg(p) < deg(q), quotient is 0, remainder is p."""
        p = polynomial(torch.tensor([1.0, 2.0]))  # 1 + 2x
        q = polynomial(torch.tensor([1.0, 0.0, 1.0]))  # 1 + x^2

        quotient, remainder = polynomial_divmod(p, q)

        # Quotient should be zero polynomial
        assert torch.allclose(quotient, torch.zeros(1), atol=1e-6)
        # Remainder should be p
        assert polynomial_equal(remainder, p, tol=1e-6)

    def test_division_by_constant(self):
        """Division by constant c is equivalent to scaling by 1/c."""
        p = polynomial(torch.tensor([2.0, 4.0, 6.0]))  # 2 + 4x + 6x^2
        q = polynomial(torch.tensor([2.0]))  # constant 2

        quotient, remainder = polynomial_divmod(p, q)

        expected_q = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        assert polynomial_equal(quotient, expected_q, tol=1e-6)
        assert torch.allclose(remainder, torch.zeros(1), atol=1e-6)

    def test_division_identity(self):
        """p / p = 1 with remainder 0."""
        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))

        quotient, remainder = polynomial_divmod(p, p)

        expected_q = polynomial(torch.tensor([1.0]))
        assert polynomial_equal(quotient, expected_q, tol=1e-6)
        assert torch.allclose(remainder, torch.zeros(1), atol=1e-6)

    def test_division_verification(self):
        """Verify p = q * quotient + remainder."""
        p = polynomial(torch.tensor([1.0, -3.0, 2.0, 5.0, -1.0]))
        q = polynomial(torch.tensor([1.0, 2.0, 1.0]))

        quotient, remainder = polynomial_divmod(p, q)

        # Reconstruct: q * quotient + remainder should equal p
        reconstructed = polynomial_add(
            polynomial_multiply(q, quotient), remainder
        )
        assert polynomial_equal(reconstructed, p, tol=1e-6)


class TestPolynomialDivMod:
    """Tests for polynomial_div and polynomial_mod convenience functions."""

    def test_div_returns_quotient(self):
        """polynomial_div returns only quotient."""
        p = polynomial(torch.tensor([-1.0, 0.0, 0.0, 1.0]))
        q = polynomial(torch.tensor([-1.0, 1.0]))

        quotient = polynomial_div(p, q)

        expected = polynomial(torch.tensor([1.0, 1.0, 1.0]))
        assert polynomial_equal(quotient, expected, tol=1e-6)

    def test_mod_returns_remainder(self):
        """polynomial_mod returns only remainder."""
        p = polynomial(torch.tensor([1.0, 0.0, 1.0]))
        q = polynomial(torch.tensor([-1.0, 1.0]))

        remainder = polynomial_mod(p, q)

        assert torch.allclose(remainder, torch.tensor([2.0]), atol=1e-6)


class TestDivisionErrors:
    """Tests for division error handling."""

    def test_division_by_zero_polynomial(self):
        """Division by zero polynomial raises DegreeError."""

        p = polynomial(torch.tensor([1.0, 2.0, 3.0]))
        q = polynomial(torch.tensor([0.0]))

        with pytest.raises(DegreeError):
            polynomial_divmod(p, q)


class TestDivisionNumpy:
    """Tests comparing against NumPy."""

    def test_divmod_vs_numpy(self):
        """Compare polynomial_divmod against np.polydiv."""
        # Random polynomial coefficients
        p_coeffs = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
        q_coeffs = torch.tensor([1.0, 2.0, 1.0])

        p = polynomial(p_coeffs)
        q = polynomial(q_coeffs)

        quotient, remainder = polynomial_divmod(p, q)

        # NumPy uses descending order, so reverse
        np_quot, np_rem = np.polydiv(
            p_coeffs.numpy()[::-1], q_coeffs.numpy()[::-1]
        )

        # Reverse back to ascending order (copy to avoid negative stride)
        np_quot = np_quot[::-1].copy()
        np_rem = np_rem[::-1].copy()

        assert torch.allclose(
            quotient,
            torch.tensor(np_quot, dtype=torch.float32),
            atol=1e-5,
        )

        # Also check remainder
        # Pad np_rem if needed to match shape
        if len(np_rem) == 0:
            np_rem = np.array([0.0])
        assert torch.allclose(
            remainder,
            torch.tensor(np_rem, dtype=torch.float32),
            atol=1e-5,
        )


class TestDivisionAutograd:
    """Tests for autograd support."""

    def test_divmod_gradcheck(self):
        """Verify gradients through polynomial_divmod."""
        p_coeffs = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], requires_grad=True, dtype=torch.float64
        )
        q_coeffs = torch.tensor(
            [1.0, 1.0], requires_grad=True, dtype=torch.float64
        )

        def divmod_sum(p_c, q_c):
            p = polynomial(p_c)
            q = polynomial(q_c)
            quot, rem = polynomial_divmod(p, q)
            return quot.sum() + rem.sum()

        assert torch.autograd.gradcheck(
            divmod_sum, (p_coeffs, q_coeffs), eps=1e-6
        )

    @pytest.mark.xfail(
        reason="Second-order backward for polynomial division needs work"
    )
    def test_divmod_gradgradcheck(self):
        """Verify second-order gradients through polynomial_divmod."""
        p_coeffs = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], requires_grad=True, dtype=torch.float64
        )
        q_coeffs = torch.tensor(
            [1.0, 1.0], requires_grad=True, dtype=torch.float64
        )

        def divmod_sum(p_c, q_c):
            p = polynomial(p_c)
            q = polynomial(q_c)
            quot, rem = polynomial_divmod(p, q)
            return quot.sum() + rem.sum()

        assert torch.autograd.gradgradcheck(
            divmod_sum, (p_coeffs, q_coeffs), eps=1e-6
        )


class TestDivisionBatched:
    """Tests for batched division."""

    def test_batched_division(self):
        """Division with batch dimensions."""
        # Batch of 2 polynomials
        p_coeffs = torch.tensor(
            [[1.0, 0.0, 1.0], [2.0, 0.0, 2.0]]
        )  # x^2+1, 2x^2+2
        q_coeffs = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])  # x-1, x-1

        p = polynomial(p_coeffs)
        q = polynomial(q_coeffs)

        quotient, remainder = polynomial_divmod(p, q)

        # Both should have quotient [1, 1] or [2, 2] and remainder 2 or 4
        assert quotient.shape[0] == 2
        assert remainder.shape[0] == 2

        # First polynomial: (x^2+1)/(x-1) = x+1 remainder 2
        assert torch.allclose(quotient[0], torch.tensor([1.0, 1.0]), atol=1e-6)
        assert torch.allclose(remainder[0], torch.tensor([2.0]), atol=1e-6)

        # Second polynomial: (2x^2+2)/(x-1) = 2x+2 remainder 4
        assert torch.allclose(quotient[1], torch.tensor([2.0, 2.0]), atol=1e-6)
        assert torch.allclose(remainder[1], torch.tensor([4.0]), atol=1e-6)


class TestDivisionOperators:
    """Tests for // and % operators."""

    def test_floordiv_operator(self):
        """Test p // q operator."""
        p = polynomial(torch.tensor([-1.0, 0.0, 0.0, 1.0]))
        q = polynomial(torch.tensor([-1.0, 1.0]))

        result = p // q

        expected = polynomial(torch.tensor([1.0, 1.0, 1.0]))
        assert polynomial_equal(result, expected, tol=1e-6)

    def test_mod_operator(self):
        """Test p % q operator."""
        p = polynomial(torch.tensor([1.0, 0.0, 1.0]))
        q = polynomial(torch.tensor([-1.0, 1.0]))

        result = p % q

        assert torch.allclose(result, torch.tensor([2.0]), atol=1e-6)
