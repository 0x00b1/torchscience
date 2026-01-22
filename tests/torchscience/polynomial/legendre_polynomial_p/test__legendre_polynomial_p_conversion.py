"""Tests for LegendrePolynomialP basis conversion."""

import numpy as np
import torch
from numpy.polynomial import legendre as np_leg

from torchscience.polynomial import (
    legendre_polynomial_p,
    legendre_polynomial_p_evaluate,
    legendre_polynomial_p_to_polynomial,
    polynomial,
    polynomial_evaluate,
    polynomial_to_legendre_polynomial_p,
)


class TestLegendrePolynomialPToPolynomial:
    """Tests for legendre_polynomial_p_to_polynomial."""

    def test_p0_to_power(self):
        """P_0 = 1."""
        c = legendre_polynomial_p(torch.tensor([1.0]))  # P_0
        p = legendre_polynomial_p_to_polynomial(c)
        torch.testing.assert_close(
            p.as_subclass(torch.Tensor), torch.tensor([1.0])
        )

    def test_p1_to_power(self):
        """P_1 = x."""
        c = legendre_polynomial_p(torch.tensor([0.0, 1.0]))  # P_1
        p = legendre_polynomial_p_to_polynomial(c)
        torch.testing.assert_close(
            p.as_subclass(torch.Tensor), torch.tensor([0.0, 1.0])
        )

    def test_p2_to_power(self):
        """P_2 = (3x^2 - 1) / 2."""
        c = legendre_polynomial_p(torch.tensor([0.0, 0.0, 1.0]))  # P_2
        p = legendre_polynomial_p_to_polynomial(c)
        torch.testing.assert_close(
            p.as_subclass(torch.Tensor), torch.tensor([-0.5, 0.0, 1.5])
        )

    def test_p3_to_power(self):
        """P_3 = (5x^3 - 3x) / 2."""
        c = legendre_polynomial_p(torch.tensor([0.0, 0.0, 0.0, 1.0]))  # P_3
        p = legendre_polynomial_p_to_polynomial(c)
        torch.testing.assert_close(
            p.as_subclass(torch.Tensor), torch.tensor([0.0, -1.5, 0.0, 2.5])
        )

    def test_linear_combination(self):
        """1 + 2*P_1 + 3*P_2 in power basis."""
        # = 1 + 2x + 3*(3x^2 - 1)/2 = 1 + 2x + 4.5x^2 - 1.5 = -0.5 + 2x + 4.5x^2
        c = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        p = legendre_polynomial_p_to_polynomial(c)
        torch.testing.assert_close(
            p.as_subclass(torch.Tensor), torch.tensor([-0.5, 2.0, 4.5])
        )

    def test_evaluation_consistency(self):
        """Legendre and power give same values."""
        coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0])
        c = legendre_polynomial_p(coeffs)
        p = legendre_polynomial_p_to_polynomial(c)

        x = torch.linspace(-1, 1, 20)
        y_leg = legendre_polynomial_p_evaluate(c, x)
        y_power = polynomial_evaluate(p, x)

        torch.testing.assert_close(y_leg, y_power, atol=1e-5, rtol=1e-5)

    def test_vs_numpy(self):
        """Compare with numpy.polynomial.legendre.leg2poly."""
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]

        c = legendre_polynomial_p(torch.tensor(coeffs, dtype=torch.float64))
        p = legendre_polynomial_p_to_polynomial(c)

        p_np = np_leg.leg2poly(coeffs)

        np.testing.assert_allclose(
            p.as_subclass(torch.Tensor).numpy(), p_np, rtol=1e-10
        )


class TestPolynomialToLegendrePolynomialP:
    """Tests for polynomial_to_legendre_polynomial_p."""

    def test_constant_to_leg(self):
        """Constant 1 = P_0."""
        p = polynomial(torch.tensor([1.0]))
        c = polynomial_to_legendre_polynomial_p(p)
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), torch.tensor([1.0])
        )

    def test_x_to_leg(self):
        """x = P_1."""
        p = polynomial(torch.tensor([0.0, 1.0]))  # x
        c = polynomial_to_legendre_polynomial_p(p)
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor), torch.tensor([0.0, 1.0])
        )

    def test_x2_to_leg(self):
        """x^2 = (2*P_2 + P_0) / 3."""
        p = polynomial(torch.tensor([0.0, 0.0, 1.0]))  # x^2
        c = polynomial_to_legendre_polynomial_p(p)
        # x^2 = (P_0 + 2*P_2) / 3 = 1/3 * P_0 + 2/3 * P_2
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor),
            torch.tensor([1.0 / 3.0, 0.0, 2.0 / 3.0]),
        )

    def test_x3_to_leg(self):
        """x^3 = (2*P_1 + 3*P_3) / 5."""
        p = polynomial(torch.tensor([0.0, 0.0, 0.0, 1.0]))  # x^3
        c = polynomial_to_legendre_polynomial_p(p)
        # x^3 = (3*P_1 + 2*P_3) / 5
        torch.testing.assert_close(
            c.as_subclass(torch.Tensor),
            torch.tensor([0.0, 3.0 / 5.0, 0.0, 2.0 / 5.0]),
        )

    def test_roundtrip_leg_power_leg(self):
        """leg -> power -> leg preserves coefficients."""
        coeffs_orig = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        c = legendre_polynomial_p(coeffs_orig)
        p = legendre_polynomial_p_to_polynomial(c)
        c_back = polynomial_to_legendre_polynomial_p(p)

        torch.testing.assert_close(
            c_back.as_subclass(torch.Tensor),
            coeffs_orig,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_roundtrip_power_leg_power(self):
        """power -> leg -> power preserves coefficients."""
        coeffs_orig = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        p = polynomial(coeffs_orig)
        c = polynomial_to_legendre_polynomial_p(p)
        p_back = legendre_polynomial_p_to_polynomial(c)

        torch.testing.assert_close(
            p_back.as_subclass(torch.Tensor),
            coeffs_orig,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_vs_numpy(self):
        """Compare with numpy.polynomial.legendre.poly2leg."""
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]

        p = polynomial(torch.tensor(coeffs, dtype=torch.float64))
        c = polynomial_to_legendre_polynomial_p(p)

        c_np = np_leg.poly2leg(coeffs)

        np.testing.assert_allclose(
            c.as_subclass(torch.Tensor).numpy(), c_np, rtol=1e-10
        )


class TestConversionAutograd:
    """Tests for autograd support in conversion."""

    def test_to_polynomial_gradcheck(self):
        """Gradcheck for legendre_polynomial_p_to_polynomial."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            return legendre_polynomial_p_to_polynomial(
                legendre_polynomial_p(c)
            ).as_subclass(torch.Tensor)

        assert torch.autograd.gradcheck(fn, (coeffs,), raise_exception=True)

    def test_from_polynomial_gradcheck(self):
        """Gradcheck for polynomial_to_legendre_polynomial_p."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            return polynomial_to_legendre_polynomial_p(
                polynomial(c)
            ).as_subclass(torch.Tensor)

        assert torch.autograd.gradcheck(fn, (coeffs,), raise_exception=True)

    def test_roundtrip_gradgradcheck(self):
        """Second-order gradients through roundtrip."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            leg = legendre_polynomial_p(c)
            poly = legendre_polynomial_p_to_polynomial(leg)
            back = polynomial_to_legendre_polynomial_p(poly)
            return back.as_subclass(torch.Tensor).sum()

        assert torch.autograd.gradgradcheck(
            fn, (coeffs,), raise_exception=True
        )
