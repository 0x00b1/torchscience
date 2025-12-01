"""Tests for ChebyshevPolynomialT basis conversion."""

import numpy as np
import torch
from numpy.polynomial import chebyshev as np_cheb

from torchscience.polynomial import (
    chebyshev_polynomial_t,
    chebyshev_polynomial_t_evaluate,
    chebyshev_polynomial_t_to_polynomial,
    polynomial,
    polynomial_evaluate,
    polynomial_to_chebyshev_polynomial_t,
)


class TestChebyshevPolynomialTToPolynomial:
    """Tests for chebyshev_polynomial_t_to_polynomial."""

    def test_t0_to_power(self):
        """T_0 = 1."""
        c = chebyshev_polynomial_t(torch.tensor([1.0]))  # T_0
        p = chebyshev_polynomial_t_to_polynomial(c)
        torch.testing.assert_close(p.coeffs, torch.tensor([1.0]))

    def test_t1_to_power(self):
        """T_1 = x."""
        c = chebyshev_polynomial_t(torch.tensor([0.0, 1.0]))  # T_1
        p = chebyshev_polynomial_t_to_polynomial(c)
        torch.testing.assert_close(p.coeffs, torch.tensor([0.0, 1.0]))

    def test_t2_to_power(self):
        """T_2 = 2x^2 - 1."""
        c = chebyshev_polynomial_t(torch.tensor([0.0, 0.0, 1.0]))  # T_2
        p = chebyshev_polynomial_t_to_polynomial(c)
        torch.testing.assert_close(p.coeffs, torch.tensor([-1.0, 0.0, 2.0]))

    def test_t3_to_power(self):
        """T_3 = 4x^3 - 3x."""
        c = chebyshev_polynomial_t(torch.tensor([0.0, 0.0, 0.0, 1.0]))  # T_3
        p = chebyshev_polynomial_t_to_polynomial(c)
        torch.testing.assert_close(
            p.coeffs, torch.tensor([0.0, -3.0, 0.0, 4.0])
        )

    def test_linear_combination(self):
        """1 + 2*T_1 + 3*T_2 in power basis."""
        # = 1 + 2x + 3*(2x^2 - 1) = 1 + 2x + 6x^2 - 3 = -2 + 2x + 6x^2
        c = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        p = chebyshev_polynomial_t_to_polynomial(c)
        torch.testing.assert_close(p.coeffs, torch.tensor([-2.0, 2.0, 6.0]))

    def test_evaluation_consistency(self):
        """Chebyshev and power give same values."""
        coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0])
        c = chebyshev_polynomial_t(coeffs)
        p = chebyshev_polynomial_t_to_polynomial(c)

        x = torch.linspace(-1, 1, 20)
        y_cheb = chebyshev_polynomial_t_evaluate(c, x)
        y_power = polynomial_evaluate(p, x)

        torch.testing.assert_close(y_cheb, y_power, atol=1e-5, rtol=1e-5)

    def test_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.cheb2poly."""
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]

        c = chebyshev_polynomial_t(torch.tensor(coeffs, dtype=torch.float64))
        p = chebyshev_polynomial_t_to_polynomial(c)

        p_np = np_cheb.cheb2poly(coeffs)

        np.testing.assert_allclose(p.coeffs.numpy(), p_np, rtol=1e-10)


class TestPolynomialToChebyshevPolynomialT:
    """Tests for polynomial_to_chebyshev_polynomial_t."""

    def test_constant_to_cheb(self):
        """Constant 1 = T_0."""
        p = polynomial(torch.tensor([1.0]))
        c = polynomial_to_chebyshev_polynomial_t(p)
        torch.testing.assert_close(c.coeffs, torch.tensor([1.0]))

    def test_x_to_cheb(self):
        """x = T_1."""
        p = polynomial(torch.tensor([0.0, 1.0]))  # x
        c = polynomial_to_chebyshev_polynomial_t(p)
        torch.testing.assert_close(c.coeffs, torch.tensor([0.0, 1.0]))

    def test_x2_to_cheb(self):
        """x^2 = (T_2 + T_0)/2 = 0.5*T_0 + 0.5*T_2."""
        p = polynomial(torch.tensor([0.0, 0.0, 1.0]))  # x^2
        c = polynomial_to_chebyshev_polynomial_t(p)
        torch.testing.assert_close(c.coeffs, torch.tensor([0.5, 0.0, 0.5]))

    def test_x3_to_cheb(self):
        """x^3 = (3*T_1 + T_3)/4 = 0.75*T_1 + 0.25*T_3."""
        p = polynomial(torch.tensor([0.0, 0.0, 0.0, 1.0]))  # x^3
        c = polynomial_to_chebyshev_polynomial_t(p)
        torch.testing.assert_close(
            c.coeffs, torch.tensor([0.0, 0.75, 0.0, 0.25])
        )

    def test_roundtrip_cheb_power_cheb(self):
        """cheb -> power -> cheb preserves coefficients."""
        coeffs_orig = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        c = chebyshev_polynomial_t(coeffs_orig)
        p = chebyshev_polynomial_t_to_polynomial(c)
        c_back = polynomial_to_chebyshev_polynomial_t(p)

        torch.testing.assert_close(
            c_back.coeffs, coeffs_orig, atol=1e-10, rtol=1e-10
        )

    def test_roundtrip_power_cheb_power(self):
        """power -> cheb -> power preserves coefficients."""
        coeffs_orig = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        p = polynomial(coeffs_orig)
        c = polynomial_to_chebyshev_polynomial_t(p)
        p_back = chebyshev_polynomial_t_to_polynomial(c)

        torch.testing.assert_close(
            p_back.coeffs, coeffs_orig, atol=1e-10, rtol=1e-10
        )

    def test_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.poly2cheb."""
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]

        p = polynomial(torch.tensor(coeffs, dtype=torch.float64))
        c = polynomial_to_chebyshev_polynomial_t(p)

        c_np = np_cheb.poly2cheb(coeffs)

        np.testing.assert_allclose(c.coeffs.numpy(), c_np, rtol=1e-10)


class TestConversionAutograd:
    """Tests for autograd support in conversion."""

    def test_to_polynomial_gradcheck(self):
        """Gradcheck for chebyshev_polynomial_t_to_polynomial."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            return chebyshev_polynomial_t_to_polynomial(
                chebyshev_polynomial_t(c)
            ).coeffs

        assert torch.autograd.gradcheck(fn, (coeffs,), raise_exception=True)

    def test_from_polynomial_gradcheck(self):
        """Gradcheck for polynomial_to_chebyshev_polynomial_t."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            return polynomial_to_chebyshev_polynomial_t(polynomial(c)).coeffs

        assert torch.autograd.gradcheck(fn, (coeffs,), raise_exception=True)

    def test_roundtrip_gradgradcheck(self):
        """Second-order gradients through roundtrip."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            cheb = chebyshev_polynomial_t(c)
            poly = chebyshev_polynomial_t_to_polynomial(cheb)
            back = polynomial_to_chebyshev_polynomial_t(poly)
            return back.coeffs.sum()

        assert torch.autograd.gradgradcheck(
            fn, (coeffs,), raise_exception=True
        )
