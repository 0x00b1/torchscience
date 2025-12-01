"""Tests for ChebyshevPolynomialT utility functions."""

import numpy as np
import torch
from numpy.polynomial import chebyshev as np_cheb

from torchscience.polynomial import (
    ChebyshevPolynomialT,
    chebyshev_polynomial_t,
    chebyshev_polynomial_t_add,
    chebyshev_polynomial_t_degree,
    chebyshev_polynomial_t_divmod,
    chebyshev_polynomial_t_equal,
    chebyshev_polynomial_t_multiply,
    chebyshev_polynomial_t_trim,
    chebyshev_polynomial_t_weight,
)


class TestChebyshevPolynomialTDegree:
    """Tests for chebyshev_polynomial_t_degree."""

    def test_degree_constant(self):
        """Degree of constant is 0."""
        c = chebyshev_polynomial_t(torch.tensor([5.0]))
        assert chebyshev_polynomial_t_degree(c).item() == 0

    def test_degree_linear(self):
        """Degree of linear is 1."""
        c = chebyshev_polynomial_t(torch.tensor([1.0, 2.0]))
        assert chebyshev_polynomial_t_degree(c).item() == 1

    def test_degree_quadratic(self):
        """Degree of quadratic is 2."""
        c = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        assert chebyshev_polynomial_t_degree(c).item() == 2

    def test_degree_with_trailing_zeros(self):
        """Degree counts all coefficients."""
        c = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 0.0]))
        assert chebyshev_polynomial_t_degree(c).item() == 2  # Still degree 2


class TestChebyshevPolynomialTTrim:
    """Tests for chebyshev_polynomial_t_trim."""

    def test_trim_no_change(self):
        """Trim doesn't change non-zero trailing."""
        c = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        t = chebyshev_polynomial_t_trim(c)
        torch.testing.assert_close(t.coeffs, c.coeffs)

    def test_trim_trailing_zeros(self):
        """Trim removes trailing zeros."""
        c = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 0.0, 0.0]))
        t = chebyshev_polynomial_t_trim(c)
        torch.testing.assert_close(t.coeffs, torch.tensor([1.0, 2.0]))

    def test_trim_near_zero(self):
        """Trim removes near-zero trailing coefficients."""
        c = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 1e-15, 1e-16]))
        t = chebyshev_polynomial_t_trim(c, tol=1e-10)
        torch.testing.assert_close(t.coeffs, torch.tensor([1.0, 2.0]))

    def test_trim_preserves_constant(self):
        """Trim preserves at least one coefficient."""
        c = chebyshev_polynomial_t(torch.tensor([0.0, 0.0, 0.0]))
        t = chebyshev_polynomial_t_trim(c)
        assert t.coeffs.shape[-1] >= 1

    def test_trim_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebtrim."""
        coeffs = [1.0, 2.0, 0.0, 1e-16, 0.0]

        c = chebyshev_polynomial_t(torch.tensor(coeffs))
        t = chebyshev_polynomial_t_trim(c, tol=1e-10)

        t_np = np_cheb.chebtrim(coeffs, tol=1e-10)

        np.testing.assert_allclose(t.coeffs.numpy(), t_np, rtol=1e-6)


class TestChebyshevPolynomialTEqual:
    """Tests for chebyshev_polynomial_t_equal."""

    def test_equal_same(self):
        """Equal series are equal."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        b = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        assert chebyshev_polynomial_t_equal(a, b)

    def test_equal_different(self):
        """Different series are not equal."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        b = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 4.0]))
        assert not chebyshev_polynomial_t_equal(a, b)

    def test_equal_different_length_padded(self):
        """Series with trailing zeros are equal."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0]))
        b = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 0.0]))
        assert chebyshev_polynomial_t_equal(a, b)

    def test_equal_with_tolerance(self):
        """Near-equal series within tolerance."""
        a = chebyshev_polynomial_t(
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        )
        b = chebyshev_polynomial_t(
            torch.tensor([1.0, 2.0, 3.0 + 1e-8], dtype=torch.float64)
        )
        assert chebyshev_polynomial_t_equal(a, b, tol=1e-6)
        assert not chebyshev_polynomial_t_equal(a, b, tol=1e-10)


class TestChebyshevPolynomialTWeight:
    """Tests for chebyshev_polynomial_t_weight."""

    def test_weight_at_zero(self):
        """w(0) = 1."""
        x = torch.tensor([0.0])
        w = chebyshev_polynomial_t_weight(x)
        torch.testing.assert_close(w, torch.tensor([1.0]))

    def test_weight_symmetric(self):
        """w(-x) = w(x)."""
        x = torch.tensor([0.3, 0.5, 0.7])
        w_pos = chebyshev_polynomial_t_weight(x)
        w_neg = chebyshev_polynomial_t_weight(-x)
        torch.testing.assert_close(w_pos, w_neg)

    def test_weight_formula(self):
        """w(x) = 1/sqrt(1-x^2)."""
        x = torch.tensor([0.0, 0.3, 0.5, 0.7])
        w = chebyshev_polynomial_t_weight(x)
        expected = 1.0 / torch.sqrt(1.0 - x**2)
        torch.testing.assert_close(w, expected)

    def test_weight_near_boundary(self):
        """Weight increases near |x|=1."""
        x = torch.tensor([0.9, 0.99, 0.999])
        w = chebyshev_polynomial_t_weight(x)
        assert w[0] < w[1] < w[2]


class TestChebyshevPolynomialTDivision:
    """Tests for division operations."""

    def test_divmod_exact(self):
        """Exact division leaves no remainder."""
        # (1 + T_1)^2 / (1 + T_1) = (1 + T_1)
        a = chebyshev_polynomial_t(torch.tensor([1.0, 1.0]))  # 1 + T_1
        a_squared = chebyshev_polynomial_t_multiply(a, a)

        q, r = chebyshev_polynomial_t_divmod(a_squared, a)

        # Quotient should be (1 + T_1)
        torch.testing.assert_close(
            q.coeffs, torch.tensor([1.0, 1.0]), atol=1e-5, rtol=1e-5
        )

        # Remainder should be ~0
        assert torch.abs(r.coeffs).max() < 1e-5

    def test_divmod_with_remainder(self):
        """Division with non-zero remainder."""
        a = chebyshev_polynomial_t(
            torch.tensor([1.0, 2.0, 3.0, 4.0])
        )  # degree 3
        b = chebyshev_polynomial_t(torch.tensor([1.0, 1.0]))  # degree 1

        q, r = chebyshev_polynomial_t_divmod(a, b)

        # Verify: a = b*q + r
        reconstructed = chebyshev_polynomial_t_add(
            chebyshev_polynomial_t_multiply(b, q), r
        )

        # Pad for comparison
        n_max = max(a.coeffs.shape[-1], reconstructed.coeffs.shape[-1])
        a_padded = torch.zeros(n_max)
        a_padded[: a.coeffs.shape[-1]] = a.coeffs
        r_padded = torch.zeros(n_max)
        r_padded[: reconstructed.coeffs.shape[-1]] = reconstructed.coeffs

        torch.testing.assert_close(a_padded, r_padded, atol=1e-5, rtol=1e-5)

    def test_div_operator(self):
        """Test // operator."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        b = chebyshev_polynomial_t(torch.tensor([1.0, 1.0]))
        q = a // b
        assert isinstance(q, ChebyshevPolynomialT)

    def test_mod_operator(self):
        """Test % operator."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        b = chebyshev_polynomial_t(torch.tensor([1.0, 1.0]))
        r = a % b
        assert isinstance(r, ChebyshevPolynomialT)
