"""Tests for LegendrePolynomialP calculus operations."""

import numpy as np
import pytest
import torch
from numpy.polynomial import legendre as np_leg

from torchscience.polynomial import (
    legendre_polynomial_p,
    legendre_polynomial_p_antiderivative,
    legendre_polynomial_p_derivative,
    legendre_polynomial_p_evaluate,
    legendre_polynomial_p_integral,
)


class TestLegendrePolynomialPDerivative:
    """Tests for legendre_polynomial_p_derivative."""

    def test_derivative_constant(self):
        """Derivative of constant is zero."""
        p = legendre_polynomial_p(torch.tensor([5.0]))
        dp = legendre_polynomial_p_derivative(p)
        torch.testing.assert_close(dp.coeffs, torch.tensor([0.0]))

    def test_derivative_p1(self):
        """d/dx P_1(x) = d/dx(x) = 1 = P_0."""
        p = legendre_polynomial_p(torch.tensor([0.0, 1.0]))  # P_1
        dp = legendre_polynomial_p_derivative(p)
        torch.testing.assert_close(
            dp.coeffs, torch.tensor([1.0]), atol=1e-6, rtol=1e-6
        )

    def test_derivative_p2(self):
        """d/dx P_2(x) = d/dx((3x^2-1)/2) = 3x = 3*P_1."""
        # P_2 = (3x^2 - 1)/2, P'_2 = 3x = 3*P_1
        p = legendre_polynomial_p(torch.tensor([0.0, 0.0, 1.0]))  # P_2
        dp = legendre_polynomial_p_derivative(p)
        torch.testing.assert_close(
            dp.coeffs, torch.tensor([0.0, 3.0]), atol=1e-6, rtol=1e-6
        )

    def test_derivative_vs_numpy(self):
        """Compare with numpy.polynomial.legendre.legder."""
        coeffs = [1.0, 2.0, 3.0, 4.0]

        p = legendre_polynomial_p(torch.tensor(coeffs))
        dp = legendre_polynomial_p_derivative(p)

        dp_np = np_leg.legder(coeffs)

        np.testing.assert_allclose(dp.coeffs.numpy(), dp_np, rtol=1e-6)

    def test_derivative_order_2(self):
        """Second derivative matches numpy."""
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]

        p = legendre_polynomial_p(torch.tensor(coeffs))
        dp2 = legendre_polynomial_p_derivative(p, order=2)

        dp2_np = np_leg.legder(coeffs, m=2)

        np.testing.assert_allclose(dp2.coeffs.numpy(), dp2_np, rtol=1e-5)

    def test_derivative_order_0(self):
        """Order 0 returns original."""
        p = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
        dp = legendre_polynomial_p_derivative(p, order=0)
        torch.testing.assert_close(dp.coeffs, p.coeffs)

    def test_derivative_linear_combination(self):
        """Derivative of linear combination."""
        # 1*P_0 + 2*P_1 + 3*P_2 = 1 + 2x + 3*(3x^2-1)/2 = -0.5 + 2x + 4.5x^2
        # d/dx = 2 + 9x = 2*P_0 + 9*P_1
        coeffs = [1.0, 2.0, 3.0]

        p = legendre_polynomial_p(torch.tensor(coeffs))
        dp = legendre_polynomial_p_derivative(p)

        dp_np = np_leg.legder(coeffs)

        np.testing.assert_allclose(dp.coeffs.numpy(), dp_np, rtol=1e-6)

    def test_derivative_high_order(self):
        """Derivative of high-degree polynomial."""
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

        p = legendre_polynomial_p(torch.tensor(coeffs))
        dp = legendre_polynomial_p_derivative(p)

        dp_np = np_leg.legder(coeffs)

        np.testing.assert_allclose(dp.coeffs.numpy(), dp_np, rtol=1e-6)


class TestLegendrePolynomialPAntiderivative:
    """Tests for legendre_polynomial_p_antiderivative."""

    def test_antiderivative_constant(self):
        """Antiderivative of P_0 = 1 is P_1 = x (with constant set correctly)."""
        p = legendre_polynomial_p(torch.tensor([1.0]))  # P_0 = 1
        ip = legendre_polynomial_p_antiderivative(p, constant=0.0)

        # Compare with numpy
        ip_np = np_leg.legint([1.0], k=[0.0])

        np.testing.assert_allclose(ip.coeffs.numpy(), ip_np, rtol=1e-6)

    def test_antiderivative_vs_numpy(self):
        """Compare with numpy.polynomial.legendre.legint."""
        coeffs = [1.0, 2.0, 3.0]

        p = legendre_polynomial_p(torch.tensor(coeffs))
        ip = legendre_polynomial_p_antiderivative(p, constant=0.0)

        ip_np = np_leg.legint(coeffs, k=[0.0])

        np.testing.assert_allclose(ip.coeffs.numpy(), ip_np, rtol=1e-5)

    def test_antiderivative_with_constant(self):
        """Antiderivative with integration constant."""
        coeffs = [1.0, 2.0]

        p = legendre_polynomial_p(torch.tensor(coeffs))
        ip = legendre_polynomial_p_antiderivative(p, constant=5.0)

        ip_np = np_leg.legint(coeffs, k=[5.0])

        np.testing.assert_allclose(ip.coeffs.numpy(), ip_np, rtol=1e-5)

    def test_derivative_antiderivative_roundtrip(self):
        """derivative(antiderivative(p)) = p."""
        coeffs = [1.0, 2.0, 3.0]
        p = legendre_polynomial_p(torch.tensor(coeffs))
        ip = legendre_polynomial_p_antiderivative(p)
        dip = legendre_polynomial_p_derivative(ip)
        torch.testing.assert_close(dip.coeffs, p.coeffs, atol=1e-5, rtol=1e-5)

    def test_antiderivative_order_2(self):
        """Second antiderivative matches numpy."""
        coeffs = [1.0, 2.0, 3.0]

        p = legendre_polynomial_p(torch.tensor(coeffs))
        i2p = legendre_polynomial_p_antiderivative(p, order=2, constant=0.0)

        i2p_np = np_leg.legint(coeffs, m=2, k=[0.0, 0.0])

        np.testing.assert_allclose(i2p.coeffs.numpy(), i2p_np, rtol=1e-5)

    def test_antiderivative_p1(self):
        """Antiderivative of P_1 matches numpy."""
        coeffs = [0.0, 1.0]  # P_1

        p = legendre_polynomial_p(torch.tensor(coeffs))
        ip = legendre_polynomial_p_antiderivative(p, constant=0.0)

        ip_np = np_leg.legint(coeffs, k=[0.0])

        np.testing.assert_allclose(ip.coeffs.numpy(), ip_np, rtol=1e-6)


class TestLegendrePolynomialPIntegral:
    """Tests for legendre_polynomial_p_integral (definite integral)."""

    def test_integral_constant(self):
        """Integral of P_0 = 1 over [-1, 1] is 2."""
        p = legendre_polynomial_p(torch.tensor([1.0]))  # P_0
        result = legendre_polynomial_p_integral(
            p, torch.tensor(-1.0), torch.tensor(1.0)
        )
        torch.testing.assert_close(
            result, torch.tensor(2.0), atol=1e-6, rtol=1e-6
        )

    def test_integral_p1(self):
        """Integral of P_1 = x over [-1, 1] is 0 (odd function)."""
        p = legendre_polynomial_p(torch.tensor([0.0, 1.0]))  # P_1
        result = legendre_polynomial_p_integral(
            p, torch.tensor(-1.0), torch.tensor(1.0)
        )
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )

    def test_integral_p2(self):
        """Integral of P_2 over [-1, 1] is 0 (orthogonality)."""
        # Legendre polynomials P_n for n >= 1 integrate to 0 over [-1, 1]
        p = legendre_polynomial_p(torch.tensor([0.0, 0.0, 1.0]))  # P_2
        result = legendre_polynomial_p_integral(
            p, torch.tensor(-1.0), torch.tensor(1.0)
        )
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=1e-5, rtol=1e-5
        )

    def test_integral_partial_domain(self):
        """Integral over [0, 1]."""
        p = legendre_polynomial_p(torch.tensor([1.0]))  # P_0
        result = legendre_polynomial_p_integral(
            p, torch.tensor(0.0), torch.tensor(1.0)
        )
        torch.testing.assert_close(
            result, torch.tensor(1.0), atol=1e-6, rtol=1e-6
        )

    def test_integral_warns_outside_domain(self):
        """Warning when integration bounds exceed natural domain."""
        p = legendre_polynomial_p(torch.tensor([1.0, 2.0]))
        with pytest.warns(UserWarning, match="Integration bounds"):
            legendre_polynomial_p_integral(
                p, torch.tensor(-2.0), torch.tensor(2.0)
            )

    def test_integral_custom_limits(self):
        """Integral over custom limits [a, b]."""
        # integral_{0}^{1} P_0 dx = 1
        p = legendre_polynomial_p(torch.tensor([1.0]))
        result = legendre_polynomial_p_integral(
            p, torch.tensor(0.0), torch.tensor(1.0)
        )
        torch.testing.assert_close(result, torch.tensor(1.0))

    def test_integral_vs_numpy(self):
        """Compare with numerical integration via numpy."""
        coeffs = [1.0, 2.0, 3.0, 4.0]
        p = legendre_polynomial_p(torch.tensor(coeffs))

        # Compute using our integral
        result = legendre_polynomial_p_integral(
            p, torch.tensor(-1.0), torch.tensor(1.0)
        )

        # Compute using numpy antiderivative and evaluate
        ip_np = np_leg.legint(coeffs)
        result_np = np_leg.legval(1.0, ip_np) - np_leg.legval(-1.0, ip_np)

        np.testing.assert_allclose(result.item(), result_np, rtol=1e-6)


class TestLegendrePolynomialPCalculusAutograd:
    """Tests for autograd support in calculus operations."""

    def test_derivative_gradcheck(self):
        """Gradcheck for derivative."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            return legendre_polynomial_p_derivative(
                legendre_polynomial_p(c)
            ).coeffs

        assert torch.autograd.gradcheck(fn, (coeffs,), raise_exception=True)

    def test_antiderivative_gradcheck(self):
        """Gradcheck for antiderivative."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            return legendre_polynomial_p_antiderivative(
                legendre_polynomial_p(c), constant=0.0
            ).coeffs

        assert torch.autograd.gradcheck(fn, (coeffs,), raise_exception=True)

    def test_integral_gradcheck_coeffs(self):
        """Gradcheck for integral w.r.t. coefficients."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            return legendre_polynomial_p_integral(
                legendre_polynomial_p(c),
                torch.tensor(-1.0, dtype=torch.float64),
                torch.tensor(1.0, dtype=torch.float64),
            )

        assert torch.autograd.gradcheck(fn, (coeffs,), raise_exception=True)

    def test_integral_gradcheck_limits(self):
        """Gradcheck for integral w.r.t. limits."""
        lower = torch.tensor(-0.5, dtype=torch.float64, requires_grad=True)
        upper = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        coeffs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        def fn(lo, hi):
            return legendre_polynomial_p_integral(
                legendre_polynomial_p(coeffs), lo, hi
            )

        assert torch.autograd.gradcheck(
            fn, (lower, upper), raise_exception=True
        )

    def test_derivative_gradgradcheck(self):
        """Second-order gradients for derivative."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            return legendre_polynomial_p_derivative(
                legendre_polynomial_p(c)
            ).coeffs.sum()

        assert torch.autograd.gradgradcheck(
            fn, (coeffs,), raise_exception=True
        )

    def test_integral_gradgradcheck(self):
        """Second-order gradients for integral."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            return legendre_polynomial_p_integral(
                legendre_polynomial_p(c),
                torch.tensor(-1.0, dtype=torch.float64),
                torch.tensor(1.0, dtype=torch.float64),
            )

        assert torch.autograd.gradgradcheck(
            fn, (coeffs,), raise_exception=True
        )


class TestLegendrePolynomialPDerivativeEvaluation:
    """Tests checking derivative via evaluation consistency."""

    def test_derivative_evaluation_consistency(self):
        """Numerical derivative matches symbolic derivative."""
        p = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        dp = legendre_polynomial_p_derivative(p)

        x = torch.tensor([0.0, 0.3, 0.7], requires_grad=True)
        y = legendre_polynomial_p_evaluate(p, x)

        # Compute numerical derivative
        grad_y = torch.autograd.grad(y.sum(), x, create_graph=True)[0]

        # Evaluate symbolic derivative
        dy_symbolic = legendre_polynomial_p_evaluate(dp, x.detach())

        torch.testing.assert_close(grad_y, dy_symbolic, atol=1e-5, rtol=1e-5)
