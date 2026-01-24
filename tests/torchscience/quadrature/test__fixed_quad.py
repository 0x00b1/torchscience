import numpy as np
import pytest
import scipy.integrate
import torch


class TestFixedQuad:
    def test_basic_integration(self):
        """Integrate sin(x) from 0 to pi"""
        from torchscience.quadrature import fixed_quad

        result = fixed_quad(torch.sin, 0, torch.pi, n=32)

        assert torch.allclose(
            result, torch.tensor(2.0, dtype=result.dtype), rtol=1e-10
        )

    def test_matches_scipy(self):
        """Compare with scipy.integrate.fixed_quad"""
        from torchscience.quadrature import fixed_quad

        result = fixed_quad(lambda x: torch.exp(-(x**2)), -1, 1, n=10)
        expected, _ = scipy.integrate.fixed_quad(
            lambda x: np.exp(-(x**2)), -1, 1, n=10
        )

        assert torch.allclose(result, torch.tensor(expected), rtol=1e-10)

    def test_batched_limits(self):
        """Test with batched upper limit"""
        from torchscience.quadrature import fixed_quad

        b = torch.linspace(1, 5, 10)
        result = fixed_quad(lambda x: x**2, 0, b, n=32)

        assert result.shape == (10,)
        expected = b**3 / 3
        assert torch.allclose(result, expected, rtol=1e-6)

    def test_polynomial_exact(self):
        """Fixed quad should be exact for polynomials of degree <= 2n-1"""
        from torchscience.quadrature import fixed_quad

        # n=5 => exact for degree <= 9
        result = fixed_quad(lambda x: x**8, 0, 1, n=5)

        assert torch.allclose(
            result, torch.tensor(1 / 9, dtype=result.dtype), rtol=1e-10
        )


class TestFixedQuadGradients:
    def test_gradient_closure_param(self):
        """Gradient flows through closure parameters"""
        from torchscience.quadrature import fixed_quad

        theta = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)

        # integral of theta * x^2 dx from 0 to 1 = theta / 3
        result = fixed_quad(lambda x: theta * x**2, 0, 1, n=32)
        result.backward()

        assert theta.grad is not None
        assert torch.allclose(
            theta.grad, torch.tensor(1 / 3, dtype=torch.float64), rtol=1e-6
        )

    def test_gradient_upper_limit(self):
        """Gradient flows through upper limit via Leibniz rule"""
        from torchscience.quadrature import fixed_quad

        b = torch.tensor(torch.pi, requires_grad=True, dtype=torch.float64)

        # d/db integral_0^b sin(x) dx = sin(b) = sin(pi) approximately 0
        result = fixed_quad(torch.sin, 0, b, n=32)
        result.backward()

        assert b.grad is not None
        assert torch.allclose(b.grad, torch.sin(b).detach(), atol=1e-6)

    def test_gradient_lower_limit(self):
        """Gradient flows through lower limit via Leibniz rule"""
        from torchscience.quadrature import fixed_quad

        a = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)

        # d/da integral_a^pi sin(x) dx = -sin(a) = -sin(0) = 0
        result = fixed_quad(torch.sin, a, torch.pi, n=32)
        result.backward()

        assert a.grad is not None
        assert torch.allclose(a.grad, -torch.sin(a).detach(), atol=1e-6)

    def test_gradcheck_limits(self):
        """Numerical gradient check for limits"""
        from torchscience.quadrature import fixed_quad

        a = torch.tensor(0.5, requires_grad=True, dtype=torch.float64)
        b = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)

        def fn(a_, b_):
            return fixed_quad(torch.sin, a_, b_, n=32)

        assert torch.autograd.gradcheck(fn, (a, b), raise_exception=True)

    def test_gradcheck_closure(self):
        """Numerical gradient check for closure parameters"""
        from torchscience.quadrature import fixed_quad

        theta = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)

        def fn(theta_):
            return fixed_quad(lambda x: theta_ * x**2, 0, 1, n=16)

        assert torch.autograd.gradcheck(fn, (theta,), raise_exception=True)


class TestFixedQuadDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_from_limits(self, dtype):
        """Output dtype matches limit dtype"""
        from torchscience.quadrature import fixed_quad

        a = torch.tensor(0.0, dtype=dtype)
        b = torch.tensor(1.0, dtype=dtype)

        result = fixed_quad(lambda x: x, a, b, n=10)

        assert result.dtype == dtype
