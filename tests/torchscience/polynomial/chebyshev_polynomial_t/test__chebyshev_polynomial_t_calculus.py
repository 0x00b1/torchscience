"""Tests for ChebyshevPolynomialT calculus operations."""

import numpy as np
import torch
from numpy.polynomial import chebyshev as np_cheb

from torchscience.polynomial import (
    chebyshev_polynomial_t,
    chebyshev_polynomial_t_antiderivative,
    chebyshev_polynomial_t_derivative,
    chebyshev_polynomial_t_evaluate,
    chebyshev_polynomial_t_integral,
)


class TestChebyshevPolynomialTDerivative:
    """Tests for chebyshev_polynomial_t_derivative."""

    def test_derivative_constant(self):
        """Derivative of constant is zero."""
        a = chebyshev_polynomial_t(torch.tensor([5.0]))
        da = chebyshev_polynomial_t_derivative(a)
        torch.testing.assert_close(da, torch.tensor([0.0]))

    def test_derivative_t1(self):
        """d/dx T_1(x) = 1 = T_0."""
        a = chebyshev_polynomial_t(torch.tensor([0.0, 1.0]))  # T_1 = x
        da = chebyshev_polynomial_t_derivative(a)
        torch.testing.assert_close(da, torch.tensor([1.0]))

    def test_derivative_t2(self):
        """d/dx T_2(x) = 4x = 4*T_1."""
        # T_2 = 2x^2 - 1, d/dx = 4x = 4*T_1
        a = chebyshev_polynomial_t(torch.tensor([0.0, 0.0, 1.0]))  # T_2
        da = chebyshev_polynomial_t_derivative(a)
        torch.testing.assert_close(da, torch.tensor([0.0, 4.0]))

    def test_derivative_t3(self):
        """d/dx T_3(x) = 3*U_2(x) = 12x^2 - 3 = 6*T_2 + 3*T_0."""
        # T_3 = 4x^3 - 3x, d/dx = 12x^2 - 3
        # In Chebyshev basis: 12x^2 - 3 = 12*(T_2+1)/2 - 3 = 6*T_2 + 6 - 3 = 3 + 6*T_2
        # = 3*T_0 + 6*T_2
        a = chebyshev_polynomial_t(torch.tensor([0.0, 0.0, 0.0, 1.0]))  # T_3
        da = chebyshev_polynomial_t_derivative(a)
        torch.testing.assert_close(da, torch.tensor([3.0, 0.0, 6.0]))

    def test_derivative_linear_combination(self):
        """Derivative of 1 + 2*T_1 + 3*T_2."""
        # d/dx (1 + 2*T_1 + 3*T_2) = 2*1 + 3*4*T_1 = 2 + 12*T_1
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        da = chebyshev_polynomial_t_derivative(a)
        torch.testing.assert_close(da, torch.tensor([2.0, 12.0]))

    def test_derivative_second_order(self):
        """Second derivative."""
        a = chebyshev_polynomial_t(
            torch.tensor([1.0, 2.0, 3.0, 4.0])
        )  # degree 3
        d2a = chebyshev_polynomial_t_derivative(a, order=2)
        # First derivative: degree 2
        # Second derivative: degree 1
        assert d2a.shape[-1] <= 3

    def test_derivative_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebder."""
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]

        a = chebyshev_polynomial_t(torch.tensor(coeffs))
        da = chebyshev_polynomial_t_derivative(a)

        da_np = np_cheb.chebder(coeffs)

        np.testing.assert_allclose(da.numpy(), da_np, rtol=1e-6)

    def test_derivative_evaluation_consistency(self):
        """Numerical derivative matches symbolic derivative."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        da = chebyshev_polynomial_t_derivative(a)

        x = torch.tensor([0.0, 0.3, 0.7], requires_grad=True)
        y = chebyshev_polynomial_t_evaluate(a, x)

        # Compute numerical derivative
        grad_y = torch.autograd.grad(y.sum(), x, create_graph=True)[0]

        # Evaluate symbolic derivative
        dy_symbolic = chebyshev_polynomial_t_evaluate(da, x.detach())

        torch.testing.assert_close(grad_y, dy_symbolic, atol=1e-5, rtol=1e-5)


class TestChebyshevPolynomialTAntiderivative:
    """Tests for chebyshev_polynomial_t_antiderivative."""

    def test_antiderivative_constant(self):
        """Antiderivative of constant."""
        # integral(1) = T_1 = x (with C=0)
        a = chebyshev_polynomial_t(torch.tensor([1.0]))  # 1 = T_0
        ia = chebyshev_polynomial_t_antiderivative(a)
        # Result: 0 + T_1 = [0, 1]
        torch.testing.assert_close(ia, torch.tensor([0.0, 1.0]))

    def test_antiderivative_with_constant(self):
        """Antiderivative with integration constant."""
        a = chebyshev_polynomial_t(torch.tensor([1.0]))  # constant 1
        ia = chebyshev_polynomial_t_antiderivative(a, constant=2.0)
        # integral(1) + 2 = x + 2 = 2*T_0 + T_1
        torch.testing.assert_close(ia, torch.tensor([2.0, 1.0]))

    def test_antiderivative_derivative_inverse(self):
        """Derivative of antiderivative recovers original."""
        a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        ia = chebyshev_polynomial_t_antiderivative(a)
        dia = chebyshev_polynomial_t_derivative(ia)
        torch.testing.assert_close(dia, a, atol=1e-6, rtol=1e-6)

    def test_antiderivative_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebint."""
        coeffs = [1.0, 2.0, 3.0]

        a = chebyshev_polynomial_t(torch.tensor(coeffs))
        ia = chebyshev_polynomial_t_antiderivative(a, constant=0.0)

        ia_np = np_cheb.chebint(coeffs)

        np.testing.assert_allclose(ia.numpy(), ia_np, rtol=1e-6)

    def test_antiderivative_order_2(self):
        """Second antiderivative."""
        coeffs = [1.0, 2.0, 3.0]

        a = chebyshev_polynomial_t(torch.tensor(coeffs))
        i2a = chebyshev_polynomial_t_antiderivative(a, order=2)

        i2a_np = np_cheb.chebint(coeffs, m=2)

        np.testing.assert_allclose(i2a.numpy(), i2a_np, rtol=1e-5)

    def test_antiderivative_t1_vs_numpy(self):
        """Antiderivative of T_1 matches numpy."""
        coeffs = [0.0, 1.0]  # T_1

        a = chebyshev_polynomial_t(torch.tensor(coeffs))
        ia = chebyshev_polynomial_t_antiderivative(a, constant=0.0)

        ia_np = np_cheb.chebint(coeffs)

        np.testing.assert_allclose(ia.numpy(), ia_np, rtol=1e-6)


class TestChebyshevPolynomialTIntegral:
    """Tests for chebyshev_polynomial_t_integral (definite integral)."""

    def test_integral_constant(self):
        """Integral of constant over [-1, 1]."""
        # integral_{-1}^{1} 1 dx = 2
        a = chebyshev_polynomial_t(torch.tensor([1.0]))
        result = chebyshev_polynomial_t_integral(
            a, torch.tensor(-1.0), torch.tensor(1.0)
        )
        torch.testing.assert_close(result, torch.tensor(2.0))

    def test_integral_t1(self):
        """Integral of T_1 = x over [-1, 1]."""
        # integral_{-1}^{1} x dx = 0 (odd function)
        a = chebyshev_polynomial_t(torch.tensor([0.0, 1.0]))
        result = chebyshev_polynomial_t_integral(
            a, torch.tensor(-1.0), torch.tensor(1.0)
        )
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )

    def test_integral_t2(self):
        """Integral of T_2 = 2x^2 - 1 over [-1, 1]."""
        # integral_{-1}^{1} (2x^2 - 1) dx = [2x^3/3 - x]_{-1}^{1}
        # = (2/3 - 1) - (-2/3 + 1) = -1/3 - 1/3 = -2/3
        a = chebyshev_polynomial_t(torch.tensor([0.0, 0.0, 1.0]))
        result = chebyshev_polynomial_t_integral(
            a, torch.tensor(-1.0), torch.tensor(1.0)
        )
        torch.testing.assert_close(
            result, torch.tensor(-2.0 / 3.0), atol=1e-6, rtol=1e-6
        )

    def test_integral_custom_limits(self):
        """Integral over [0, 1]."""
        # integral_{0}^{1} 1 dx = 1
        a = chebyshev_polynomial_t(torch.tensor([1.0]))
        result = chebyshev_polynomial_t_integral(
            a, torch.tensor(0.0), torch.tensor(1.0)
        )
        torch.testing.assert_close(result, torch.tensor(1.0))

    def test_integral_quadratic(self):
        """Integral of 1 + x + x^2 over [0, 1]."""
        # Convert to Chebyshev: 1 + x + x^2 = 1 + T_1 + (T_2+1)/2 = 1.5 + T_1 + 0.5*T_2
        # integral_{0}^{1} (1 + x + x^2) dx = [x + x^2/2 + x^3/3]_{0}^{1}
        # = 1 + 0.5 + 1/3 = 11/6
        a = chebyshev_polynomial_t(torch.tensor([1.5, 1.0, 0.5]))
        result = chebyshev_polynomial_t_integral(
            a, torch.tensor(0.0), torch.tensor(1.0)
        )
        torch.testing.assert_close(
            result, torch.tensor(11.0 / 6.0), atol=1e-5, rtol=1e-5
        )

    def test_integral_vs_numpy(self):
        """Compare with numerical integration."""
        coeffs = [1.0, 2.0, 3.0, 4.0]
        a = chebyshev_polynomial_t(torch.tensor(coeffs))

        # Compute using our integral
        result = chebyshev_polynomial_t_integral(
            a, torch.tensor(-1.0), torch.tensor(1.0)
        )

        # Compute using numpy antiderivative and evaluate
        ia_np = np_cheb.chebint(coeffs)
        result_np = np_cheb.chebval(1.0, ia_np) - np_cheb.chebval(-1.0, ia_np)

        np.testing.assert_allclose(result.item(), result_np, rtol=1e-6)


class TestChebyshevPolynomialTCalculusAutograd:
    """Tests for autograd support in calculus operations."""

    def test_derivative_gradcheck(self):
        """Gradcheck for derivative."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            result = chebyshev_polynomial_t_derivative(
                chebyshev_polynomial_t(c)
            )
            return result.as_subclass(torch.Tensor)

        assert torch.autograd.gradcheck(fn, (coeffs,), raise_exception=True)

    def test_antiderivative_gradcheck(self):
        """Gradcheck for antiderivative."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            result = chebyshev_polynomial_t_antiderivative(
                chebyshev_polynomial_t(c), constant=0.0
            )
            return result.as_subclass(torch.Tensor)

        assert torch.autograd.gradcheck(fn, (coeffs,), raise_exception=True)

    def test_integral_gradcheck_coeffs(self):
        """Gradcheck for integral w.r.t. coefficients."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            return chebyshev_polynomial_t_integral(
                chebyshev_polynomial_t(c),
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
            return chebyshev_polynomial_t_integral(
                chebyshev_polynomial_t(coeffs), lo, hi
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
            result = chebyshev_polynomial_t_derivative(
                chebyshev_polynomial_t(c)
            )
            return result.as_subclass(torch.Tensor).sum()

        assert torch.autograd.gradgradcheck(
            fn, (coeffs,), raise_exception=True
        )

    def test_integral_gradgradcheck(self):
        """Second-order gradients for integral."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            return chebyshev_polynomial_t_integral(
                chebyshev_polynomial_t(c),
                torch.tensor(-1.0, dtype=torch.float64),
                torch.tensor(1.0, dtype=torch.float64),
            )

        assert torch.autograd.gradgradcheck(
            fn, (coeffs,), raise_exception=True
        )
