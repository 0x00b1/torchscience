import numpy as np
import pytest
import scipy.integrate
import torch


class TestGaussLegendre:
    def test_init(self):
        from torchscience.integration.quadrature import GaussLegendre

        rule = GaussLegendre(32)
        assert rule.n == 32

    def test_invalid_n_raises(self):
        from torchscience.integration.quadrature import GaussLegendre

        with pytest.raises(ValueError, match="at least 1"):
            GaussLegendre(0)

    def test_nodes_and_weights_default(self):
        """Default interval is [-1, 1]"""
        from torchscience.integration.quadrature import GaussLegendre

        rule = GaussLegendre(10)
        nodes, weights = rule.nodes_and_weights()

        assert nodes.shape == (10,)
        assert weights.shape == (10,)
        assert (nodes >= -1).all()
        assert (nodes <= 1).all()

    def test_nodes_and_weights_scaled(self):
        """Scale to [a, b]"""
        from torchscience.integration.quadrature import GaussLegendre

        rule = GaussLegendre(10)
        nodes, weights = rule.nodes_and_weights(a=0, b=1)

        assert (nodes >= 0).all()
        assert (nodes <= 1).all()
        # Weights should sum to (b - a) = 1
        assert torch.allclose(
            weights.sum(), torch.tensor(1.0, dtype=weights.dtype)
        )

    def test_integrate_sin(self):
        """Integrate sin(x) from 0 to pi"""
        from torchscience.integration.quadrature import GaussLegendre

        rule = GaussLegendre(32)
        result = rule.integrate(torch.sin, 0, torch.pi)

        assert torch.allclose(
            result, torch.tensor(2.0, dtype=result.dtype), rtol=1e-10
        )

    def test_integrate_polynomial_exact(self):
        """Should be exact for polynomials of degree <= 2n-1"""
        from torchscience.integration.quadrature import GaussLegendre

        rule = GaussLegendre(5)
        # Integrate x^8 from 0 to 1 = 1/9
        result = rule.integrate(lambda x: x**8, 0, 1)

        assert torch.allclose(
            result, torch.tensor(1 / 9, dtype=result.dtype), rtol=1e-10
        )

    def test_integrate_matches_scipy(self):
        """Compare with scipy.integrate.fixed_quad"""
        from torchscience.integration.quadrature import GaussLegendre

        rule = GaussLegendre(10)
        result = rule.integrate(lambda x: torch.exp(-(x**2)), -1, 1)

        expected, _ = scipy.integrate.fixed_quad(
            lambda x: np.exp(-(x**2)), -1, 1, n=10
        )

        assert torch.allclose(result, torch.tensor(expected), rtol=1e-10)


class TestGaussLegendreBatched:
    def test_batched_limits(self):
        """Test with batched integration limits"""
        from torchscience.integration.quadrature import GaussLegendre

        rule = GaussLegendre(32)
        b = torch.linspace(1, 10, 5)

        # Integrate x^2 from 0 to b
        result = rule.integrate(lambda x: x**2, 0, b)

        assert result.shape == (5,)
        # integral of x^2 from 0 to b = b^3 / 3
        expected = b**3 / 3
        assert torch.allclose(result, expected, rtol=1e-6)

    def test_batched_both_limits(self):
        """Test with both limits batched"""
        from torchscience.integration.quadrature import GaussLegendre

        rule = GaussLegendre(32)
        a = torch.tensor([0.0, 1.0, 2.0])
        b = torch.tensor([1.0, 2.0, 3.0])

        # Integrate x from a to b = (b^2 - a^2) / 2
        result = rule.integrate(lambda x: x, a, b)

        expected = (b**2 - a**2) / 2
        assert torch.allclose(result, expected, rtol=1e-6)


class TestGaussLegendreGradients:
    def test_gradient_through_closure(self):
        """Gradient flows through closure parameters"""
        from torchscience.integration.quadrature import GaussLegendre

        theta = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)
        rule = GaussLegendre(32)

        # integral of theta * x^2 dx from 0 to 1 = theta / 3
        result = rule.integrate(lambda x: theta * x**2, 0, 1)
        result.backward()

        assert theta.grad is not None
        assert torch.allclose(
            theta.grad, torch.tensor(1 / 3, dtype=torch.float64), rtol=1e-6
        )

    def test_gradcheck_closure(self):
        """Numerical gradient check for closure parameter"""
        from torchscience.integration.quadrature import GaussLegendre

        theta = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)
        rule = GaussLegendre(16)

        def fn(theta_):
            return rule.integrate(lambda x: theta_ * x**2, 0, 1)

        assert torch.autograd.gradcheck(fn, (theta,), raise_exception=True)


class TestGaussKronrod:
    def test_init(self):
        from torchscience.integration.quadrature import GaussKronrod

        rule = GaussKronrod(15)
        assert rule.order == 15

    def test_invalid_order(self):
        from torchscience.integration.quadrature import GaussKronrod

        with pytest.raises(ValueError, match="order must be"):
            GaussKronrod(20)

    def test_integrate(self):
        """Basic integration should work"""
        from torchscience.integration.quadrature import GaussKronrod

        rule = GaussKronrod(15)
        result = rule.integrate(torch.sin, 0, torch.pi)

        assert torch.allclose(
            result, torch.tensor(2.0, dtype=result.dtype), rtol=1e-10
        )

    def test_integrate_with_error(self):
        from torchscience.integration.quadrature import GaussKronrod

        rule = GaussKronrod(15)
        result, error = rule.integrate_with_error(torch.sin, 0, torch.pi)

        assert torch.allclose(
            result, torch.tensor(2.0, dtype=result.dtype), rtol=1e-10
        )
        assert error < 1e-10  # Very small error for smooth function

    def test_error_estimate(self):
        """Error estimate should be |Kronrod - Gauss|"""
        from torchscience.integration.quadrature import GaussKronrod

        rule = GaussKronrod(15)

        # Integrate something that's harder for low-order rules
        def oscillatory(x):
            return torch.sin(10 * x)

        result, error = rule.integrate_with_error(oscillatory, 0, torch.pi)

        # Error should be positive and reasonable
        assert error > 0
        assert error < 1  # Should still be somewhat accurate

    def test_gauss_vs_kronrod(self):
        """Gauss and Kronrod should agree for smooth functions"""
        from torchscience.integration.quadrature import GaussKronrod

        rule = GaussKronrod(21)

        # For a polynomial of low degree, both should be very accurate
        def poly(x):
            return x**3

        kronrod_result, error = rule.integrate_with_error(poly, 0, 1)
        # integral of x^3 from 0 to 1 = 1/4

        assert torch.allclose(
            kronrod_result,
            torch.tensor(0.25, dtype=kronrod_result.dtype),
            rtol=1e-10,
        )
        assert error < 1e-10


class TestGaussKronrodGradients:
    def test_gradient_closure(self):
        """Gradient flows through closure parameters"""
        from torchscience.integration.quadrature import GaussKronrod

        theta = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)
        rule = GaussKronrod(15)

        result = rule.integrate(lambda x: theta * x**2, 0, 1)
        result.backward()

        assert theta.grad is not None
        assert torch.allclose(
            theta.grad, torch.tensor(1 / 3, dtype=torch.float64), rtol=1e-6
        )
