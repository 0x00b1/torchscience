"""Tests for LegendrePolynomialP arithmetic operations."""

import numpy as np
import torch
from numpy.polynomial import legendre as np_leg

from torchscience.polynomial import (
    legendre_polynomial_p,
    legendre_polynomial_p_add,
    legendre_polynomial_p_negate,
    legendre_polynomial_p_scale,
    legendre_polynomial_p_subtract,
)


class TestLegendrePolynomialPAdd:
    """Tests for legendre_polynomial_p_add."""

    def test_add_same_degree(self):
        """Add series of same degree."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        b = legendre_polynomial_p(torch.tensor([4.0, 5.0, 6.0]))
        c = legendre_polynomial_p_add(a, b)
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), torch.tensor([5.0, 7.0, 9.0])
        )

    def test_add_different_degree(self):
        """Add series of different degrees (zero-pad shorter)."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        b = legendre_polynomial_p(torch.tensor([3.0, 4.0, 5.0]))
        c = legendre_polynomial_p_add(a, b)
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), torch.tensor([4.0, 6.0, 5.0])
        )

    def test_add_operator(self):
        """Test + operator."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        b = legendre_polynomial_p(torch.tensor([3.0, 4.0]))
        c = a + b
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), torch.tensor([4.0, 6.0])
        )

    def test_add_vs_numpy(self):
        """Compare with numpy.polynomial.legendre.legadd."""
        a_coeffs = [1.0, 2.0, 3.0]
        b_coeffs = [4.0, 5.0]

        a = legendre_polynomial_p(torch.tensor(a_coeffs))
        b = legendre_polynomial_p(torch.tensor(b_coeffs))
        c = legendre_polynomial_p_add(a, b)

        c_np = np_leg.legadd(a_coeffs, b_coeffs)

        np.testing.assert_allclose(
            c.as_subclass(torch.Tensor).numpy(), c_np, rtol=1e-6
        )


class TestLegendrePolynomialPSubtract:
    """Tests for legendre_polynomial_p_subtract."""

    def test_subtract_same_degree(self):
        """Subtract series of same degree."""
        a = legendre_polynomial_p(torch.tensor([5.0, 7.0, 9.0]))
        b = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        c = legendre_polynomial_p_subtract(a, b)
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), torch.tensor([4.0, 5.0, 6.0])
        )

    def test_subtract_different_degree(self):
        """Subtract series of different degrees."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        b = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        c = legendre_polynomial_p_subtract(a, b)
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), torch.tensor([0.0, 0.0, 3.0])
        )

    def test_subtract_operator(self):
        """Test - operator."""
        a = legendre_polynomial_p(torch.tensor([5.0, 6.0]))
        b = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        c = a - b
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), torch.tensor([4.0, 4.0])
        )

    def test_subtract_vs_numpy(self):
        """Compare with numpy.polynomial.legendre.legsub."""
        a_coeffs = [5.0, 4.0, 3.0]
        b_coeffs = [1.0, 2.0]

        a = legendre_polynomial_p(torch.tensor(a_coeffs))
        b = legendre_polynomial_p(torch.tensor(b_coeffs))
        c = legendre_polynomial_p_subtract(a, b)

        c_np = np_leg.legsub(a_coeffs, b_coeffs)

        np.testing.assert_allclose(
            c.as_subclass(torch.Tensor).numpy(), c_np, rtol=1e-6
        )


class TestLegendrePolynomialPNegate:
    """Tests for legendre_polynomial_p_negate."""

    def test_negate(self):
        """Negate series."""
        a = legendre_polynomial_p(torch.tensor([1.0, -2.0, 3.0]))
        b = legendre_polynomial_p_negate(a)
        torch.testing.assert_close(
            b.as_subclass(torch.Tensor), torch.tensor([-1.0, 2.0, -3.0])
        )

    def test_negate_operator(self):
        """Test unary - operator."""
        a = legendre_polynomial_p(torch.tensor([1.0, -2.0]))
        b = -a
        torch.testing.assert_close(
            b.as_subclass(torch.Tensor), torch.tensor([-1.0, 2.0])
        )


class TestLegendrePolynomialPScale:
    """Tests for legendre_polynomial_p_scale."""

    def test_scale_by_scalar(self):
        """Scale by scalar tensor."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        b = legendre_polynomial_p_scale(a, torch.tensor(2.0))
        torch.testing.assert_close(
            b.as_subclass(torch.Tensor), torch.tensor([2.0, 4.0, 6.0])
        )

    def test_scale_operator(self):
        """Test * operator with scalar."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        b = a * torch.tensor(3.0)
        torch.testing.assert_close(
            b.as_subclass(torch.Tensor), torch.tensor([3.0, 6.0])
        )

    def test_scale_rmul_operator(self):
        """Test scalar * series."""
        a = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        b = torch.tensor(3.0) * a
        torch.testing.assert_close(
            b.as_subclass(torch.Tensor), torch.tensor([3.0, 6.0])
        )


class TestLegendrePolynomialPArithmeticAutograd:
    """Tests for autograd support in arithmetic operations."""

    def test_add_gradcheck(self):
        """Gradcheck for add."""
        a_coeffs = torch.tensor(
            [1.0, 2.0], dtype=torch.float64, requires_grad=True
        )
        b_coeffs = torch.tensor(
            [3.0, 4.0, 5.0], dtype=torch.float64, requires_grad=True
        )

        def fn(a, b):
            return legendre_polynomial_p_add(
                legendre_polynomial_p(a), legendre_polynomial_p(b)
            ).as_subclass(torch.Tensor)

        assert torch.autograd.gradcheck(
            fn, (a_coeffs, b_coeffs), raise_exception=True
        )

    def test_subtract_gradcheck(self):
        """Gradcheck for subtract."""
        a_coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        b_coeffs = torch.tensor(
            [4.0, 5.0], dtype=torch.float64, requires_grad=True
        )

        def fn(a, b):
            return legendre_polynomial_p_subtract(
                legendre_polynomial_p(a), legendre_polynomial_p(b)
            ).as_subclass(torch.Tensor)

        assert torch.autograd.gradcheck(
            fn, (a_coeffs, b_coeffs), raise_exception=True
        )

    def test_scale_gradcheck(self):
        """Gradcheck for scale."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        scalar = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

        def fn(c, s):
            return legendre_polynomial_p_scale(
                legendre_polynomial_p(c), s
            ).as_subclass(torch.Tensor)

        assert torch.autograd.gradcheck(
            fn, (coeffs, scalar), raise_exception=True
        )

    def test_negate_gradcheck(self):
        """Gradcheck for negate."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            return legendre_polynomial_p_negate(
                legendre_polynomial_p(c)
            ).as_subclass(torch.Tensor)

        assert torch.autograd.gradcheck(fn, (coeffs,), raise_exception=True)
