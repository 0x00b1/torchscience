"""Tests for ChebyshevPolynomialT tensorclass."""

import pytest
import torch

from torchscience.polynomial import chebyshev_polynomial_t


class TestChebyshevPolynomialTConstructor:
    """Tests for chebyshev_polynomial_t() constructor."""

    def test_single_coefficient(self):
        """Constant Chebyshev series."""
        c = chebyshev_polynomial_t(torch.tensor([3.0]))
        assert c.coeffs.shape == (1,)
        assert c.coeffs[0] == 3.0

    def test_multiple_coefficients(self):
        """Standard Chebyshev series."""
        c = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        assert c.coeffs.shape == (3,)
        torch.testing.assert_close(c.coeffs, torch.tensor([1.0, 2.0, 3.0]))

    def test_empty_raises(self):
        """Empty coefficients raise error."""
        from torchscience.polynomial import PolynomialError

        with pytest.raises(PolynomialError):
            chebyshev_polynomial_t(torch.tensor([]))

    def test_preserves_dtype(self):
        """Dtype is preserved."""
        c = chebyshev_polynomial_t(
            torch.tensor([1.0, 2.0], dtype=torch.float64)
        )
        assert c.coeffs.dtype == torch.float64

    def test_preserves_device(self):
        """Device is preserved."""
        coeffs = torch.tensor([1.0, 2.0])
        c = chebyshev_polynomial_t(coeffs)
        assert c.coeffs.device == coeffs.device


from torchscience.polynomial import chebyshev_polynomial_t_evaluate


class TestChebyshevPolynomialTEvaluate:
    """Tests for chebyshev_polynomial_t_evaluate using Clenshaw algorithm."""

    def test_evaluate_constant(self):
        """T_0(x) = 1 for all x."""
        c = chebyshev_polynomial_t(torch.tensor([3.0]))
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        y = chebyshev_polynomial_t_evaluate(c, x)
        torch.testing.assert_close(y, torch.tensor([3.0, 3.0, 3.0, 3.0]))

    def test_evaluate_t1(self):
        """T_1(x) = x."""
        c = chebyshev_polynomial_t(torch.tensor([0.0, 1.0]))  # T_1
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        y = chebyshev_polynomial_t_evaluate(c, x)
        torch.testing.assert_close(y, x)

    def test_evaluate_t2(self):
        """T_2(x) = 2x^2 - 1."""
        c = chebyshev_polynomial_t(torch.tensor([0.0, 0.0, 1.0]))  # T_2
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        expected = 2 * x**2 - 1  # [-1+2, -1, -0.5, 1] = [1, -1, -0.5, 1]
        y = chebyshev_polynomial_t_evaluate(c, x)
        torch.testing.assert_close(y, expected)

    def test_evaluate_linear_combination(self):
        """1 + 2*T_1 + 3*T_2 at x=0.5."""
        # T_0(0.5) = 1, T_1(0.5) = 0.5, T_2(0.5) = 2*0.25 - 1 = -0.5
        # Result: 1 + 2*0.5 + 3*(-0.5) = 1 + 1 - 1.5 = 0.5
        c = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
        x = torch.tensor([0.5])
        y = chebyshev_polynomial_t_evaluate(c, x)
        torch.testing.assert_close(y, torch.tensor([0.5]))

    def test_evaluate_at_chebyshev_points(self):
        """T_n(cos(k*pi/n)) = cos(k*pi)."""
        import math

        # T_3 at cos(k*pi/3) for k=0,1,2,3
        c = chebyshev_polynomial_t(torch.tensor([0.0, 0.0, 0.0, 1.0]))  # T_3
        k = torch.tensor([0.0, 1.0, 2.0, 3.0])
        x = torch.cos(k * math.pi / 3)
        y = chebyshev_polynomial_t_evaluate(c, x)
        expected = torch.cos(k * math.pi)  # [1, -1, 1, -1]
        torch.testing.assert_close(y, expected, atol=1e-6, rtol=1e-6)

    def test_evaluate_call_operator(self):
        """Test __call__ operator."""
        c = chebyshev_polynomial_t(torch.tensor([1.0, 2.0]))
        x = torch.tensor([0.0, 1.0])
        y = c(x)
        # T_0=1, T_1=x: 1 + 2*0=1, 1 + 2*1=3
        torch.testing.assert_close(y, torch.tensor([1.0, 3.0]))


class TestChebyshevPolynomialTEvaluateAutograd:
    """Tests for autograd support in chebyshev_polynomial_t_evaluate."""

    def test_grad_wrt_coeffs(self):
        """Gradient through evaluation w.r.t. coefficients."""
        coeffs = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = chebyshev_polynomial_t(coeffs)
        x = torch.tensor([0.5])
        y = chebyshev_polynomial_t_evaluate(c, x)
        y.sum().backward()

        assert coeffs.grad is not None
        assert coeffs.grad.shape == coeffs.shape

    def test_grad_wrt_x(self):
        """Gradient through evaluation w.r.t. x."""
        coeffs = torch.tensor([1.0, 2.0, 3.0])
        c = chebyshev_polynomial_t(coeffs)
        x = torch.tensor([0.5], requires_grad=True)
        y = chebyshev_polynomial_t_evaluate(c, x)
        y.sum().backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradcheck_coeffs(self):
        """torch.autograd.gradcheck for coefficients."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            return chebyshev_polynomial_t_evaluate(
                chebyshev_polynomial_t(c),
                torch.tensor([0.3, 0.7], dtype=torch.float64),
            )

        assert torch.autograd.gradcheck(fn, (coeffs,), raise_exception=True)

    def test_gradcheck_x(self):
        """torch.autograd.gradcheck for x."""
        x = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
        coeffs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        def fn(x_):
            return chebyshev_polynomial_t_evaluate(
                chebyshev_polynomial_t(coeffs), x_
            )

        assert torch.autograd.gradcheck(fn, (x,), raise_exception=True)

    def test_gradgradcheck_coeffs(self):
        """Second-order gradients w.r.t. coefficients."""
        coeffs = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(c):
            return chebyshev_polynomial_t_evaluate(
                chebyshev_polynomial_t(c),
                torch.tensor([0.3, 0.7], dtype=torch.float64),
            )

        assert torch.autograd.gradgradcheck(
            fn, (coeffs,), raise_exception=True
        )

    def test_gradgradcheck_x(self):
        """Second-order gradients w.r.t. x."""
        x = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
        coeffs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        def fn(x_):
            return chebyshev_polynomial_t_evaluate(
                chebyshev_polynomial_t(coeffs), x_
            )

        assert torch.autograd.gradgradcheck(fn, (x,), raise_exception=True)


import numpy as np
from numpy.polynomial import chebyshev as np_cheb


class TestChebyshevPolynomialTVsNumPy:
    """Tests comparing against NumPy's chebyshev module."""

    def test_evaluate_vs_numpy(self):
        """Compare evaluation against numpy.polynomial.chebyshev.chebval."""
        coeffs = [1.0, -2.0, 3.0, -4.0, 5.0]
        x = np.linspace(-1, 1, 20)

        c_torch = chebyshev_polynomial_t(torch.tensor(coeffs))
        y_torch = chebyshev_polynomial_t_evaluate(
            c_torch, torch.tensor(x)
        ).numpy()

        y_np = np_cheb.chebval(x, coeffs)

        np.testing.assert_allclose(y_torch, y_np, rtol=1e-6)

    def test_evaluate_high_degree(self):
        """High degree polynomial evaluation matches NumPy."""
        coeffs = np.random.randn(20).tolist()
        x = np.linspace(-1, 1, 50)

        c_torch = chebyshev_polynomial_t(torch.tensor(coeffs))
        y_torch = chebyshev_polynomial_t_evaluate(
            c_torch, torch.tensor(x)
        ).numpy()

        y_np = np_cheb.chebval(x, coeffs)

        np.testing.assert_allclose(y_torch, y_np, rtol=1e-5)

    def test_evaluate_at_endpoints(self):
        """Evaluation at domain endpoints matches NumPy."""
        coeffs = [1.0, 2.0, 3.0]
        x = np.array([-1.0, 1.0])

        c_torch = chebyshev_polynomial_t(torch.tensor(coeffs))
        y_torch = chebyshev_polynomial_t_evaluate(
            c_torch, torch.tensor(x)
        ).numpy()

        y_np = np_cheb.chebval(x, coeffs)

        np.testing.assert_allclose(y_torch, y_np, rtol=1e-6)


class TestChebyshevPolynomialTBatched:
    """Tests for batched Chebyshev series operations."""

    def test_batched_constructor(self):
        """Batched Chebyshev series."""
        coeffs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        c = chebyshev_polynomial_t(coeffs)
        assert c.coeffs.shape == (2, 2)

    def test_batched_evaluate(self):
        """Evaluate batched series at multiple points."""
        # Two series: 1 + T_1 and 2 + 3*T_1
        coeffs = torch.tensor([[1.0, 1.0], [2.0, 3.0]])
        c = chebyshev_polynomial_t(coeffs)
        x = torch.tensor([0.0, 0.5, 1.0])
        y = chebyshev_polynomial_t_evaluate(c, x)

        # Result shape: (2, 3) - 2 series x 3 points
        assert y.shape == (2, 3)

        # c[0] = 1 + x: at [0, 0.5, 1] -> [1, 1.5, 2]
        # c[1] = 2 + 3x: at [0, 0.5, 1] -> [2, 3.5, 5]
        expected = torch.tensor([[1.0, 1.5, 2.0], [2.0, 3.5, 5.0]])
        torch.testing.assert_close(y, expected)

    def test_batched_multidim(self):
        """Multi-dimensional batch."""
        coeffs = torch.randn(2, 3, 4)  # 2x3 batch of degree-3 polynomials
        c = chebyshev_polynomial_t(coeffs)
        x = torch.tensor([0.0, 0.5])
        y = chebyshev_polynomial_t_evaluate(c, x)

        assert y.shape == (2, 3, 2)  # (batch, points)
