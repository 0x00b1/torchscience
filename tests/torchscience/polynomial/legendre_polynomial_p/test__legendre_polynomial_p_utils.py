"""Tests for LegendrePolynomialP utility functions."""

import numpy as np
import torch
from numpy.polynomial import legendre as np_leg

from torchscience.polynomial import (
    legendre_polynomial_p,
    legendre_polynomial_p_degree,
    legendre_polynomial_p_equal,
    legendre_polynomial_p_evaluate,
    legendre_polynomial_p_mulx,
    legendre_polynomial_p_trim,
)


class TestLegendrePolynomialPDegree:
    """Tests for legendre_polynomial_p_degree."""

    def test_degree_constant(self):
        """Degree of constant is 0."""
        c = legendre_polynomial_p(torch.tensor([5.0]))
        assert legendre_polynomial_p_degree(c).item() == 0

    def test_degree_linear(self):
        """Degree of linear is 1."""
        c = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        assert legendre_polynomial_p_degree(c).item() == 1

    def test_degree_quadratic(self):
        """Degree of quadratic is 2."""
        c = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        assert legendre_polynomial_p_degree(c).item() == 2

    def test_degree_with_trailing_zeros(self):
        """Degree counts all coefficients."""
        c = legendre_polynomial_p(torch.tensor([1.0, 2.0, 0.0]))
        assert legendre_polynomial_p_degree(c).item() == 2  # Still degree 2


class TestLegendrePolynomialPTrim:
    """Tests for legendre_polynomial_p_trim."""

    def test_trim_no_change(self):
        """Trim doesn't change non-zero trailing."""
        c = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        t = legendre_polynomial_p_trim(c)
        torch.testing.assert_close(t.coeffs, c.coeffs)

    def test_trim_trailing_zeros(self):
        """Trim removes trailing zeros."""
        c = legendre_polynomial_p(torch.tensor([1.0, 2.0, 0.0, 0.0]))
        t = legendre_polynomial_p_trim(c)
        torch.testing.assert_close(t.coeffs, torch.tensor([1.0, 2.0]))

    def test_trim_near_zero(self):
        """Trim removes near-zero trailing coefficients."""
        c = legendre_polynomial_p(torch.tensor([1.0, 2.0, 1e-15, 1e-16]))
        t = legendre_polynomial_p_trim(c, tol=1e-10)
        torch.testing.assert_close(t.coeffs, torch.tensor([1.0, 2.0]))

    def test_trim_preserves_constant(self):
        """Trim preserves at least one coefficient."""
        c = legendre_polynomial_p(torch.tensor([0.0, 0.0, 0.0]))
        t = legendre_polynomial_p_trim(c)
        assert t.coeffs.shape[-1] >= 1

    def test_trim_vs_numpy(self):
        """Compare with numpy.polynomial.legendre.legtrim."""
        coeffs = [1.0, 2.0, 0.0, 1e-16, 0.0]

        c = legendre_polynomial_p(torch.tensor(coeffs))
        t = legendre_polynomial_p_trim(c, tol=1e-10)

        t_np = np_leg.legtrim(coeffs, tol=1e-10)

        np.testing.assert_allclose(t.coeffs.numpy(), t_np, rtol=1e-6)


class TestLegendrePolynomialPEqual:
    """Tests for legendre_polynomial_p_equal."""

    def test_equal_same(self):
        """Equal series are equal."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        b = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        assert legendre_polynomial_p_equal(a, b)

    def test_equal_different(self):
        """Different series are not equal."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        b = legendre_polynomial_p(torch.tensor([1.0, 2.0, 4.0]))
        assert not legendre_polynomial_p_equal(a, b)

    def test_equal_different_length_padded(self):
        """Series with trailing zeros are equal."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        b = legendre_polynomial_p(torch.tensor([1.0, 2.0, 0.0]))
        assert legendre_polynomial_p_equal(a, b)

    def test_equal_with_tolerance(self):
        """Near-equal series within tolerance."""
        a = legendre_polynomial_p(
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        )
        b = legendre_polynomial_p(
            torch.tensor([1.0, 2.0, 3.0 + 1e-8], dtype=torch.float64)
        )
        assert legendre_polynomial_p_equal(a, b, tol=1e-6)
        assert not legendre_polynomial_p_equal(a, b, tol=1e-10)


class TestLegendrePolynomialPMulx:
    """Tests for legendre_polynomial_p_mulx."""

    def test_mulx_p0(self):
        """x * P_0 = P_1."""
        c = legendre_polynomial_p(torch.tensor([1.0]))  # P_0
        result = legendre_polynomial_p_mulx(c)

        # x * 1 = x = P_1
        expected = torch.tensor([0.0, 1.0])
        torch.testing.assert_close(result.coeffs, expected)

    def test_mulx_p1(self):
        """x * P_1 = (2*P_2 + P_0) / 3."""
        c = legendre_polynomial_p(torch.tensor([0.0, 1.0]))  # P_1
        result = legendre_polynomial_p_mulx(c)

        # x * P_1 = (P_0 + 2*P_2) / 3
        expected = torch.tensor([1.0 / 3.0, 0.0, 2.0 / 3.0])
        torch.testing.assert_close(result.coeffs, expected)

    def test_mulx_verify_evaluation(self):
        """Verify x * c(x) = result(x) for various x."""
        coeffs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        c = legendre_polynomial_p(coeffs)
        result = legendre_polynomial_p_mulx(c)

        x = torch.linspace(-1, 1, 20, dtype=torch.float64)
        y_c = legendre_polynomial_p_evaluate(c, x)
        y_result = legendre_polynomial_p_evaluate(result, x)
        y_expected = x * y_c

        torch.testing.assert_close(
            y_result, y_expected, atol=1e-10, rtol=1e-10
        )

    def test_mulx_vs_numpy(self):
        """Compare with numpy.polynomial.legendre.legmulx."""
        coeffs = [1.0, 2.0, 3.0, 4.0]

        c = legendre_polynomial_p(torch.tensor(coeffs, dtype=torch.float64))
        result = legendre_polynomial_p_mulx(c)

        result_np = np_leg.legmulx(coeffs)

        np.testing.assert_allclose(
            result.coeffs.numpy(), result_np, rtol=1e-10
        )

    def test_mulx_degree_increases(self):
        """Multiplying by x increases degree by 1."""
        c = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))  # degree 2
        result = legendre_polynomial_p_mulx(c)
        assert result.coeffs.shape[-1] == 4  # degree 3


class TestLegendrePolynomialPMulxAutograd:
    """Tests for autograd support in mulx."""

    def test_mulx_gradcheck(self):
        """Gradcheck for legendre_polynomial_p_mulx."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            return legendre_polynomial_p_mulx(legendre_polynomial_p(c)).coeffs

        assert torch.autograd.gradcheck(fn, (coeffs,), raise_exception=True)
